# Qwen-Image-2.1 Prompt Rewrite（官方提示词改写流程）

移植自 [QwenLM/Qwen-Image-2.1](https://github.com/QwenLM/Qwen-Image-2.1) 的
`prompt_rewrite/` 流程，固定 commit `ab96caa944157ca28a4f36b695c3047549f48f98`。
上游调研记录见 `artifacts/qwen-image-rewrite/upstream/SPEC.md`。

官方 prompt enhancer 是两个 Qwen3.5-VL 9B 微调 checkpoint
（`Qwen/Qwen-Image-2.1-PE-T2I` / `Qwen/Qwen-Image-2.1-PE-I2I`），各自绑定独立的
系统提示词与回答契约：

- **t2i**：纯文本扩写，不接受图片。回答 JSON：`{"rewritten_prompt": "...", "wh_ratio": "16:9"}`。
- **edit**：编辑指令改写，必须带 1..N 张源图，模型用 `<image1>`…`<imageN>` 引用（顺序敏感）。
  回答 JSON 增加 `ratio_follow`，与 `wh_ratio` **互斥**。

## 当前状态

流程已完整落地：系统提示词资产（逐字节内嵌）、任务 profile、消息构造、严格 JSON
解析（含 `parse_ok=false` 回退）、比例决策、CLI 接线与单元测试。

本地改写后端已接入：crate feature **`rewrite`** 启用
`omni-qwen-image::rewrite::minicpm::MinicpmRewriter`，在本地 MiniCPM-V-4.7 checkpoint
上执行改写（默认 `models/MiniCPM-V-4.7-35B-A3B-int4`，`--rewrite-model <dir>` 覆盖，
如 `models/MiniCPM-V-4.7-1B`）。构建：

```sh
cargo build --release --features rewrite --manifest-path crates/qwen-image/Cargo.toml
```

未启用该 feature 时 `--rewrite` 仍会在加载生成模型**之前**清晰报错（不静默透传原始
提示词）；不带 `--rewrite` 时行为与之前完全一致。

## CLI 行为（`generate --rewrite`）

```sh
# t2i：默认 35B-A3B-int4 改写器
qwen-image generate --rewrite --prompt '一只柯基' --output corgi.png
# 换用小 checkpoint 快速迭代
qwen-image generate --rewrite --rewrite-model models/MiniCPM-V-4.7-1B --prompt '一只柯基'
# edit：1..N 张参考图，顺序即 <image1>…<imageN>
qwen-image generate --rewrite --image a.png --image b.png --prompt '把第一张图的猫放进第二张图的场景'
```

后端语义：

- **t2i**（无 `--image`）：`positive_prompt` 作为生成提示词；`wh_ratio` 经
  `nearest_preset` 映射到官方七组尺寸预设之一（对数比例距离最近，同距取表中靠前者）。
- **edit**（有 `--image`）：`positive_prompt` 作为生成提示词；`ratio_follow: <imageK>`
  时输出画布跟随第 K 张参考图的宽高比——按 `canvas_for_ratio` 以约 2048² 面积、
  32 对齐、不超侧长/像素上限解析（精确可表示的比例取精确画布，如 16:9 → 2560×1440）。
  `wh_ratio`（模型判定构图迁移）同样走七预设映射。
- **显式尺寸优先**：只要用户传了 `--aspect-ratio` 或 `--width`/`--height` 中任意一个，
  模型的比例决策不生效（对应系统提示词"用户给了比例就用用户的"）。
- **解析回退**：`parse_ok=false` 时 `positive_prompt` 保留模型原始回答全文、比例字段为空，
  stderr 打警告后继续生成（上游"宁可保留长文也不丢输出"语义）。
- `ratio_follow` 引用越界（如只有 1 张图却返回 `<image2>`）：警告并保持默认画布。
- 最终 JSON 输出新增 `rewrite` 字段：完整 `RewriteOutcome`（含 thinking、原始字段值、
  `raw_answer`）外加 `canvas_source`（`user`/`wh_ratio`/`ratio_follow`/`default`）与
  实际 `canvas`。未启用 `--rewrite` 时为 `null`。
- 用户的 `--negative-prompt` 不受影响；改写结果的 `negative_prompt` 恒为空（上游契约）。

## 模块 API（`omni_qwen_image::rewrite`）

后端接入点是 `Rewriter` trait——后续 MiniCPM-V 任务实现它：

```rust
pub trait Rewriter {
    fn rewrite(
        &mut self,
        profile: TaskProfile,
        prompt: &str,
        images: &[PathBuf],
    ) -> Result<RewriteOutcome>;
}
```

配套类型与函数：

- `TaskProfile::{T2i, Edit}`：`select(has_images)`、`system_prompt()`、`sampling()`
  （t2i `presence_penalty=1.5`、`max_new_tokens=16256`；edit `0`、`24000`；两者
  `temperature=1.0`、`top_p=0.95`、`top_k=20`、`min_p=0.0`、`seed=42`，thinking 必需）、
  `takes_images()`、`has_ratio_follow()`。
- `build_messages(profile, prompt, images) -> Result<Vec<Message>>`：一个 system 消息
  （官方系统提示词）+ 一个 user 消息（**图片按序在前、文字在后**）。t2i 带图、edit 无图、
  空提示词都会报错。
- `RewriteOutcome::from_decoded(profile, decoded)`：`split_thinking` → 严格 JSON 解析
  （`balanced_spans` 平衡括号扫描，候选**从后往前**试）→ `ratio_decision`。
  字段：`task`、`parse_ok`、`thinking`、`positive_prompt`、`negative_prompt`（恒空）、
  `ratio: RatioDecision`、原始 `wh_ratio`/`ratio_follow` 字符串、`raw_answer`。
- `RatioDecision::{None, WhRatio(u32, u32), FollowImage(usize)}`：`FollowImage` 为 1-based。
- `nearest_preset(w, h) -> (&str, usize, usize)`、`canvas_for_ratio(w, h) -> Result<(usize, usize)>`、
  `parse_wh_ratio`、`parse_image_tag`。
- `default_backend() -> Result<Box<dyn Rewriter>>`：仅 `rewrite` feature **关闭**时存在，
  返回清晰的"未启用后端"错误（CLI 的兜底）。

### MiniCPM-V 后端（`rewrite::minicpm`，feature `rewrite`）

```rust
pub const DEFAULT_MODEL_DIR: &str = "models/MiniCPM-V-4.7-35B-A3B-int4";
pub struct MinicpmRewriter<B: Backend> { /* owns TextModel + Multimodal + tokenizer */ }
impl<B: Backend> MinicpmRewriter<B> {
    pub fn load(model_dir: &Path, precision: Precision, device: &B::Device) -> Result<Self>;
}
/// 本 crate 内改写器实际运行的后端（裸 cubecl / NdArray；原因见「与上游的刻意差异」）。
pub type Native = …;
```

- t2i：消息经 `omni_minicpm_v::chat::render` 渲染为纯文本 prompt。
- edit：走 `omni_minicpm_v::multimodal::Multimodal::build_prompt_multi`，第 i 张图展开为
  `<image_id>i</image_id>` 前缀的 pad 占位（即官方系统提示词里的 `<image1>`…`<imageN>`，
  顺序敏感）；视觉特征经 canvas M-RoPE 位置表拼入 token 流。
- 每次改写在 stderr 打一行延迟：`rewrite {t2i|edit}: N tokens in Xs (first token Yms, finish=…)`。
- 采样直接映射 `profile.sampling()`；`repetition_penalty` 恒 1.0（上游无此旋钮）。

## 与上游的刻意差异

- **仅严格 JSON**：上游可选的 `json_repair` 修复未移植；近乎合法的坏 JSON 走
  `parse_ok=false` 回退，语义一致。
- **比例集合收窄**：t2i 系统提示词可能产出七预设之外的比例（21:9、4:5、3:1、1:2 等），
  统一经 `nearest_preset` 映射到最近的官方预设，不渲染自定义画布。
- **互斥违反的处理**：edit 回答同时给出 `wh_ratio` 与 `ratio_follow` 时 `ratio_follow`
  优先（保留输入构图的破坏性更小）。
- `rewritten_prompt` 字段存在但非法（非字符串或空）时继续尝试 `rewrited_prompt` 别名，
  比上游略宽松；两个键都非法才放弃该候选。

### 后端相关（MiniCPM-V 替代 Qwen3.5-VL-9B-PE）

- **模型替换**：改写器是本地 MiniCPM-V-4.7（默认 35B-A3B-int4），不是官方微调的
  Qwen-Image-2.1-PE checkpoint。输出语义质量由 `parse_ok` 率与样本人工检查兜底。
- **非 thinking**：MiniCPM-V 的生成提示用 `enable_thinking=Some(false)`（空 `<think>` 块），
  而上游 thinking 为必需。严格解析器同时容忍两种回答形态。
- **presence_penalty 语义**：t2i 的 1.5 映射到 `omni_minicpm_v::generate::Options`，其采样
  history **包含 prompt token**（vLLM 只罚已生成 token）——即系统/用户提示中出现过的
  token 也会被同样扣减。
- **min_p**：上游 `min_p=0.0` 本为空操作过滤器，`generate::Options` 没有该字段，不构成差异。
- **int4 加载**：Checkpoint 加载按 Backend 选定表示。改写器使用裸 Cube，所以 Metal 与 WebGPU
  构建都保持 int4 打包（W4A16 内核内反量化）。NdArray 在加载点反量化。
- **独立非 fusion 运行时**：qwen-image 的 `metal`/`webgpu` feature 为扩散模型开启
  `burn/fusion`，但 burn-fusion 的 IR 构造器要求二元算子两侧 dtype 完全一致，会把 int4
  内核内反量化 matmul 的输出记为量化参数的 F32 dtype，与外围 F16 激活冲突——35B int4 的
  MoE expert 路径（`silu(gate) * up`）在 fusion 下图构建即 panic `DTypeMismatch`。改写器因此
  固定实例化裸 cubecl 后端（`rewrite::minicpm::Native` =
  `CubeBackend<WgpuRuntime, f32, i32, u32>`，即 minicpm-v CLI/测试验证过的同一类型），与扩散
  模型的 fusion 运行时以两个 wgpu device 共存；扩散模型路径不受影响。

## 测试

单元测试全部无需模型，为纯逻辑测试（parser、profile 默认值、消息构造、比例→尺寸映射、
`Rewriter` seam 桩实现；feature `rewrite` 另加后端消息映射测试）。Feature 矩阵：

```sh
cargo test --manifest-path crates/qwen-image/Cargo.toml --target-dir target/qwen-image-test \
  --no-default-features --features cpu
cargo test --manifest-path crates/qwen-image/Cargo.toml --target-dir target/qwen-image-test \
  --no-default-features --features cpu,rewrite
cargo test --manifest-path crates/qwen-image/Cargo.toml --target-dir target/qwen-image
cargo test --features rewrite --manifest-path crates/qwen-image/Cargo.toml \
  --target-dir target/qwen-image-rewrite
```

## 验证证据

`artifacts/qwen-image-rewrite/minicpm-backend/`：真实 CLI 端到端矩阵（512×512，Metal FP16，
M5 Max），复现脚本 `run-matrix.zsh`。每次运行产出 `<name>.json`（最终 JSON，含完整
`rewrite` 字段）、`<name>.log`（stderr，含后端延迟行）、`<name>.png`。

（实测数据表见该目录 `REPORT.md`。）

## 许可证与归属

两份系统提示词文本属 **Qwen RESEARCH LICENSE AGREEMENT**（仅限研究/非商业用途）覆盖的
Materials，逐字节复制自上游并在 `crates/qwen-image/src/rewrite/THIRD_PARTY.md` 记录归属、
固定 commit 与 SHA-256。周边 Rust 代码保持 Apache-2.0。
