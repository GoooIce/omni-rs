# 可观察语气数据与反事实评估合同

仅供训练／开发分区使用的实际输入遮蔽与不读取 holdout 音频的接口，见 [语气开发输入准备](speech-tone-input.md)。该接口不执行模型，也不验证保留 PCM 独立性。

`omni_rsi::speech_tone` 提供数据准入和已有预测的离线汇总，不采集音频、不下载语料、不运行模型，也不决定是否发布模型。它记录的是可听见的表达标签；标签、标注者意见和预测都不能用来断言说话人的真实心理状态。合成语音的指令、音高、速度和音量只是控制条件，不是情绪真值；ASR 转写和音量也不是情绪裁判。模拟样本只能检查合同和指标，不能代表真实模型质量。

## 清单与准入

`ToneDataset` 可用 serde 从 JSON 读取，版本固定为 `1`，包含非空且不重复的 `labels` 与 `records`。每条 `ToneRecord` 包含：

- 全局唯一 `id`、`split`（`train`、`dev` 或 `holdout`）、原文 `text` 与 `text_family`。
- 匿名 `speaker_id`、`session_id` 和 `derived_family`。原始录音、裁剪、增强或合成变体的家族关系须由数据准备者如实标注；这些 ID 须在整个清单中使用同一命名空间。
- `audio` 中相对于数据集根目录的 WAV `path` 和小写十六进制文件 `sha256`；`source` 中的数据源名、许可说明及 `kind`（`human` 或 `synthetic`）。许可字段记录来源，不代表程序已验证许可或获得使用权。
- 非空 `annotations`：每位 `annotator_id` 在该样本只能出现一次，逐条记录 `source` 和 `decision`。决定是 `label` 加声明过的标签，或 `uncertain`、`abstain`。`annotation_summary` 返回各标签票数以及未知和弃判数，不自动消除分歧或生成参考标签。
- 可选 `reference`：用于评分的表达标签和独立的 `source`。它由数据准备者明确提供，不从多数票、合成参数、ASR 或响度推导。没有参考标签的题目仍可进入报告，但不参与召回或校准。

`ToneDataset::validate(root)` 会检查版本、必填字段、标签、ID、标注来源和文件完整性，拒绝不存在、逃出数据根目录、摘要不符、空、损坏、截断或含非有限浮点值的 WAV。训练、开发和保留分组不得共享说话人、会话、派生家族、文本家族或相同文件；规范化后相同的文本也不能跨组。音频还按采样率、声道、位深、样本格式与解码后的样本检查，防止仅改 WAV 容器字节绕过排重。

这些检查不能发现未如实登记的说话人或家族，也不能保证识别重采样、变速、裁剪、不同编码或近似文本造成的泄漏。数据准备者仍须先按真实来源分组，再生成派生样本并检查许可。`validate` 不证明音频确实获得授权，也不证明标签或听审质量。

以下只是字段形状示例，路径、摘要和来源均是占位符，不是一份可用语料：

```json
{
  "version": 1,
  "labels": ["steady", "emphatic"],
  "records": [{
    "id": "example-1", "split": "dev",
    "audio": {"path": "example.wav", "sha256": "<实际文件的 SHA-256>"},
    "text": "我们走吧", "text_family": "text-1",
    "speaker_id": "speaker-1", "session_id": "session-1", "derived_family": "family-1",
    "source": {"name": "占位来源", "license": "占位许可", "kind": "synthetic"},
    "annotations": [
      {"annotator_id": "a", "source": "占位听审-a", "decision": {"kind": "label", "label": "steady"}},
      {"annotator_id": "b", "source": "占位听审-b", "decision": {"kind": "label", "label": "emphatic"}},
      {"annotator_id": "c", "source": "占位听审-c", "decision": {"kind": "uncertain"}},
      {"annotator_id": "d", "source": "占位听审-d", "decision": {"kind": "abstain"}}
    ]
  }]
}
```

## 预测、指标与报告

从 JSON 读取 `ToneEvaluationPlan` 和 `Vec<TonePrediction>` 后调用 `plan.evaluate(&dataset, root, &predictions)`；它先重新校验清单，再返回可序列化的 `ToneReport`。一次运行只包含一个声明的分组。每题用 `id`、`record_id` 和 `input`（`full`、`audio_masked` 或 `text_masked`）标识；一个样本在同一输入条件下只能出现一次。每题必须且只能有一条预测：`label` 含声明的标签和 `[0, 1]` 内有限置信度，或 `abstain` 含原因。缺题、多余题、重复题、未知标签和非法置信度都会被拒绝。

例如，单题计划的 `cases` 可以是 `[{"id":"case-1","record_id":"example-1","input":"full"}]`，对应预测为 `[{"case_id":"case-1","outcome":{"kind":"label","label":"steady","confidence":0.8}}]`；弃判的 `outcome` 写成 `{"kind":"abstain","reason":"证据不足"}`。计划还须提供 `run_id`、`split` 和 `pairs`（无对照时为空数组）。上面的示例没有 `reference`，因此只能报告覆盖率，不能计算召回或校准。

报告的 `rows` 保留完整样本、每位标注者及来源、参考标签、题目输入条件和预测；`run_id` 由调用方指定，应绑定可追溯的模型、配置与运行记录。报告的 `by_input` 分别汇总完整、音频遮蔽及文本遮蔽条件；总 `metrics` 汇总所有行，如果同一样本在多个条件出现，总数也会重复计算，应优先比较各条件的指标。

各项指标口径：

- 逐类召回 = 该参考类预测正确数 / 该类有参考的题数；预测弃判计入分母且不算正确。宏平均只平均参考数非零的类别；没有参考的类和无法计算的宏平均为 `null`。
- 覆盖率 = 有标签预测数 / 全部题数；弃判率 = 弃判数 / 全部题数。无参考标签的题仍计入覆盖率。
- 校准使用有参考且没有弃判的题，将所预测标签的置信度放入 10 个等宽区间（左闭右开，1.0 在最后一区间）。每个区间报告边界，每个非空区间报告平均置信度与准确率；ECE 是两者绝对差按区间题数加权的平均。没有可评分的预测时 ECE 为 `null`。没有参考的题和弃判不参与 ECE。

## 对照

`pairs` 引用同一计划里的两道题；相同类型的同一对题不能重复登记，包括调换前后顺序。报告逐对的 `changed` 和 `both_correct`，以及按对照类型统计的总对数、可比较对数、变化、不变和双方正确的对数。如果任一预测弃判，变化为 `null`，不计入可比较对数；缺少参考时不能计算双方正确。

- `same_text_different_expression`：两题都是完整输入，原文逐字相同，音频样本不同，且独立参考表达标签不同。
- `same_expression_different_text`：两题都是完整输入，规范化文本不同，音频样本不同，且参考表达标签相同。
- `audio_masked`／`text_masked`：基准题为完整输入，对照题指向同一条样本且声明对应遮蔽条件。

接口汇总调用方提供的预测，并校验上述清单关系；它无法验证模型运行时是否真的应用了遮蔽，也无法单凭配对排除说话人、录制条件等混杂因素。应记录生成预测的方法并结合人工听审解释变化，不能仅凭预测发生变化就声称模型理解了语气。

基础设施测试不需要语料、权重或设备：

```sh
CARGO_TARGET_DIR=target CARGO_BUILD_JOBS=2 \
  cargo test --offline --locked --manifest-path crates/omni-rsi/Cargo.toml \
  --no-default-features --features cpu --test speech_tone
```

真实中文效果仍需获许可的人声、多名独立标注者、保留分歧、未见说话人／会话、不同噪声条件和人工听审；见[语音 RSI 研究](research/lfm3-omni-speech-rsi-learning.md)。
