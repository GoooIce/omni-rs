//! Grouped MoE kernels for a speculative verify block.
//!
//! The burn-level paths either read every expert of the layer (three dense
//! matmuls) or launch per expert after a router readback (syncs and a launch
//! storm). One layer's block forward is four launches here: the router
//! (k-split logit dots, top-k, renormalized softmax and the shared-expert
//! sigmoid gate in one cube per token), the packed gate/up projection over
//! the touched experts, the shared expert's gate/up, and a down projection
//! that folds the weighted combine, the shared expert's down projection and
//! the final add into one output. A workgroup for an untouched expert exits
//! after reading its mask bit, before touching the packed projections.
//! Layout contract: `gate`/`up` are packed `[E * inter, hidden]`, `down` is
//! packed `[hidden, E * inter]` (row-major, group 128, u32 words with f32
//! scales appended), exactly the tensors [`crate::quant`] concatenates for
//! [`crate::moe::SparseMoeBlock`].
use burn::tensor::{DType, Shape, Tensor as BurnTensor, TensorPrimitive, backend::Backend};
use burn_cubecl::tensor::CubeTensor;
use cubecl::{prelude::*, wgpu::WgpuRuntime};
use std::mem::{forget, size_of};

const LANES: u32 = 64;

#[cube]
fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

/// Extract the signed 4-bit value at nibble slot `nibble` (0..8) of a word:
/// shift the slot to the top and arithmetic-shift back down, which
/// sign-extends in two integer ops. The unpack runs once per weight in the
/// gate/up and down kernels, so its ALU cost is the dominant term there —
/// the old mask-and-correct form spent six ops per nibble.
#[cube]
fn word_bits(word: i32, nibble: i32) -> i32 {
    (word << (28 - nibble * 4)) >> 28
}

/// Router for one token per cube, one unit per expert. Each unit computes its
/// expert's logit, the cube shares them through cube address space, and every
/// unit redundantly derives the same top-k: argmax rounds where strict `>`
/// keeps the lowest index on a tie — the same rule as the stable CPU sort in
/// `route`, so the renormalized top-k softmax weights match the official
/// router — and each picked slot is masked to -infty after barriers, so later
/// rounds are plain scans without taken-check branches. Unit zero also
/// scores the shared expert's sigmoid gate. Measured alternatives that all
/// lost: a transposed coalesced read (~+200µs), a burn matmul for the logits
/// (F32 skinny matmuls run no faster), and a 4-way k-split of the dot (the
/// dot improves but 1024-unit barriers slow the scans by more).
#[cube(launch_unchecked)]
fn moe_router(
    x: &Tensor<f32>,               // (m, hidden)
    router_w: &Tensor<f32>,        // (experts, hidden)
    gate_vec: &Tensor<f32>,        // (hidden,) shared-expert sigmoid gate
    weights: &mut Tensor<f32>,     // (m, experts) top-k renormalized softmax
    shared_gate: &mut Tensor<f32>, // (m,) sigmoid(dot(x, gate_vec))
    #[comptime] top_k: u32,
    #[comptime] experts: u32,
    hidden: u32,
) {
    let token = CUBE_POS_X;
    let e = UNIT_POS_X;
    let x_base = (token * hidden) as usize;
    let w_row = (e * hidden) as usize;
    let mut acc = 0.0f32;
    for k in 0..hidden {
        acc += x[x_base + k as usize] * router_w[w_row + k as usize];
    }
    let mut logits = SharedMemory::<f32>::new(experts as usize);
    logits[e as usize] = acc;
    sync_cube();
    // A negative float literal (or constant-folded literal arithmetic) in a
    // let initializer breaks the cube macro; build the sentinel in a second
    // statement instead.
    let mut neg_inf = 0.0f32;
    neg_inf = neg_inf - 3.4e38;
    let mut sum = 0.0f32;
    let mut max = logits[0];
    let mut self_picked = false;
    #[unroll]
    for r in 0..top_k {
        let mut best = neg_inf;
        let mut best_i = 0u32;
        for i in 0..experts {
            let v = logits[i as usize];
            if v > best {
                best = v;
                best_i = i;
            }
        }
        // Everyone must finish scanning before anyone masks: a unit still
        // reading would see the sentinel and pick a different slot. And the
        // mask needs its own barrier so the next round's scan sees it.
        sync_cube();
        logits[best_i as usize] = neg_inf;
        sync_cube();
        if r == 0 {
            max = best;
        }
        sum += (best - max).exp();
        if best_i == e {
            self_picked = true;
        }
    }
    let mut weight = 0.0f32;
    if self_picked {
        weight = (acc - max).exp() / sum;
    }
    weights[(token * experts + e) as usize] = weight;
    if e == 0 {
        let mut gate = 0.0f32;
        for k in 0..hidden {
            gate += x[x_base + k as usize] * gate_vec[k as usize];
        }
        shared_gate[token as usize] = 1.0 / (1.0 + (-gate).exp());
    }
}

/// checkpoint values cast losslessly at load): one thread per (token, j).
#[cube(launch_unchecked)]
fn moe_shared_gate_up(
    x: &Tensor<f32>,             // (m, hidden)
    shared_gate_w: &Tensor<f32>, // (shared_inter, hidden)
    shared_up_w: &Tensor<f32>,   // (shared_inter, hidden)
    act: &mut Tensor<f32>,       // (m, shared_inter)
    hidden: u32,
    shared_inter: u32,
) {
    let token = CUBE_POS_X;
    let j = CUBE_POS_Y * CUBE_DIM_X + UNIT_POS_X;
    let x_base = (token * hidden) as usize;
    let row = (j * hidden) as usize;
    let mut acc_gate = 0.0f32;
    let mut acc_up = 0.0f32;
    for k in 0..hidden {
        let value = x[x_base + k as usize];
        acc_gate += value * shared_gate_w[row + k as usize];
        acc_up += value * shared_up_w[row + k as usize];
    }
    act[(token * shared_inter + j) as usize] = silu(acc_gate) * acc_up;
}

/// Fused tail of the block: for each (token, h) output, the routed sum over
/// the token's touched experts (weight applied here, once), plus the shared
/// expert's down projection scaled by its sigmoid gate. Replaces the separate
/// down, combine, shared down, gate multiply and final add.
#[cube(launch_unchecked)]
fn moe_down_total(
    act: &Tensor<f32>,         // (m, experts * inter) row-major
    act_shared: &Tensor<f32>,  // (m, shared_inter)
    shared_gate: &Tensor<f32>, // (m,)
    weights: &Tensor<f32>,     // (m, experts)
    down_words: &Tensor<i32>,  // packed (hidden, experts * inter)
    down_scales: &Tensor<f32>,
    shared_down_w: &Tensor<f32>, // (hidden, shared_inter)
    y: &mut Tensor<f32>,         // (m, hidden)
    hidden: u32,
    inter: u32,
    #[comptime] experts: u32,
    shared_inter: u32,
) {
    let m = CUBE_POS_X;
    let h = CUBE_POS_Y * CUBE_DIM_X + UNIT_POS_X;
    let w_base = (m * experts) as usize;
    let act_base = (m * experts * inter) as usize;
    let mut acc = 0.0f32;
    for e in 0..experts {
        let weight = weights[w_base + e as usize];
        if weight != 0.0 {
            let row_base = h * (experts * inter) + e * inter;
            let row_scale = row_base / 128;
            let a_base = act_base + (e * inter) as usize;
            let mut part = 0.0f32;
            // Down words are k-major — `(expert_col_word) * hidden + h`. As in
            // gate/up, one scale per word covers all eight nibbles.
            for word_index in 0..inter / 8 {
                let word = down_words[((e * (inter / 8) + word_index) * hidden + h) as usize];
                let scale = down_scales[(row_scale + word_index / 16) as usize];
                let mut w = 0.0f32;
                #[unroll]
                for nibble in 0..8i32 {
                    let j = word_index * 8 + u32::cast_from(nibble);
                    w += act[a_base + j as usize] * f32::cast_from(word_bits(word, nibble));
                }
                part += w * scale;
            }
            acc += weight * part;
        }
    }
    let s_base = (m * shared_inter) as usize;
    let d_row = (h * shared_inter) as usize;
    let mut shared_acc = 0.0f32;
    for j in 0..shared_inter {
        shared_acc += act_shared[s_base + j as usize] * shared_down_w[d_row + j as usize];
    }
    acc += shared_gate[m as usize] * shared_acc;
    y[(m * hidden + h) as usize] = acc;
}

/// Token masks per expert: bit m is set when token m routed to expert e
/// (nonzero router weight). The sparse gate/up kernel reads these instead of
/// rescanning the weight matrix.
#[cube(launch_unchecked)]
fn moe_masks(
    weights: &Tensor<f32>,   // (m, experts)
    masks: &mut Tensor<u32>, // (experts,)
    m_len: u32,
    experts: u32,
) {
    let e = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;
    let mut mask = 0u32;
    for m in 0..m_len {
        if weights[(m * experts + e) as usize] != 0.0 {
            mask += 1u32 << m;
        }
    }
    masks[e as usize] = mask;
}

/// Sparse gate/up: each (expert, row) thread projects only the tokens whose
/// bit is set in the expert's mask. With dispersed routing the 16-token
/// verify block touches ~100 of 256 experts for ~1.2 tokens each, so
/// computing every touched expert for all sixteen tokens — the dense and
/// vector forms — paid ~13x the necessary MACs. Unrouted act slots stay
/// unwritten; the down kernel skips them by the same zero weights.
#[cube(launch_unchecked)]
fn moe_gate_up_sparse(
    x: &Tensor<f32>,     // (m, hidden) row-major
    masks: &Tensor<u32>, // (experts,)
    gate_words: &Tensor<i32>,
    gate_scales: &Tensor<f32>,
    up_words: &Tensor<i32>,
    up_scales: &Tensor<f32>,
    act: &mut Tensor<f32>, // (m, experts * inter) row-major
    m_len: u32,
    hidden: u32,
    inter: u32,
    #[comptime] experts: u32,
) {
    let e = CUBE_POS_X;
    let j = CUBE_POS_Y * CUBE_DIM_X + UNIT_POS_X;
    let mask = masks[e as usize];
    if mask != 0u32 {
        let row = e * inter + j;
        let row_scale = (row * hidden) / 128;
        let rows = experts * inter;
        for m in 0..m_len {
            if (mask >> m) % 2 == 1u32 {
                let x_base = (m * hidden) as usize;
                let mut acc_gate = 0.0f32;
                let mut acc_up = 0.0f32;
                for word_index in 0..hidden / 8 {
                    let word = gate_words[(word_index * rows + row) as usize];
                    let up_word = up_words[(word_index * rows + row) as usize];
                    // A word's eight nibbles sit inside one 128-wide scale
                    // group, so the scale multiplies the word's partial dot
                    // once instead of every weight.
                    let scale = gate_scales[(row_scale + word_index / 16) as usize];
                    let up_scale = up_scales[(row_scale + word_index / 16) as usize];
                    let mut g = 0.0f32;
                    let mut u = 0.0f32;
                    #[unroll]
                    for nibble in 0..8i32 {
                        let k = word_index * 8 + u32::cast_from(nibble);
                        let x_value = x[x_base + k as usize];
                        g += x_value * f32::cast_from(word_bits(word, nibble));
                        u += x_value * f32::cast_from(word_bits(up_word, nibble));
                    }
                    acc_gate += g * scale;
                    acc_up += u * up_scale;
                }
                act[(m * rows + row) as usize] = silu(acc_gate) * acc_up;
            }
        }
    }
}

/// Four-launch grouped forward over the concatenated expert payloads: router,
/// packed gate/up, shared gate/up, fused down/combine/shared. The word and
/// scale tensors are flat and read linearly. Returns `None` off Metal or on
/// shapes the kernels do not cover; the output is the full block output
/// `[m, hidden]` (routed sum plus gated shared expert) in FP32.
pub(crate) fn try_dense_moe<B: Backend>(
    x: BurnTensor<B, 2>,
    dense: &crate::moe::DenseExperts<B>,
) -> Option<BurnTensor<B, 2>> {
    use burn::backend::wgpu::Metal;
    if std::any::TypeId::of::<B>() != std::any::TypeId::of::<Metal>()
        || size_of::<BurnTensor<B, 2>>() != size_of::<BurnTensor<Metal, 2>>()
    {
        return None;
    }
    let (m, hidden) = (x.dims()[0], x.dims()[1]);
    let experts = dense.num_experts;
    if m == 0 || hidden == 0 || dense.top_k > experts || experts > 1024 {
        return None;
    }

    let router_w = cube_float2d(dense.router.clone())?;
    let gate_vec = cube_float2d(dense.shared_gate_vec.clone())?;
    let x: BurnTensor<Metal, 2> = unsafe { take_as(x) };
    // The kernels read FP32; the model keeps activations in its device dtype.
    let x_row = cube_float(x.clone().cast(burn::tensor::FloatDType::F32))?;
    let client = x_row.client.clone();
    let device = x_row.device.clone();
    let weights = alloc_f32_2d(&client, &device, [m, experts]);
    let shared_gate = alloc_f32_2d(&client, &device, [m, 1]);
    let (m_len, hidden, experts) = (m as u32, hidden as u32, experts as u32);
    unsafe {
        moe_router::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(m_len, 1, 1),
            CubeDim::new_1d(experts),
            x_row.into_tensor_arg(),
            router_w.into_tensor_arg(),
            gate_vec.into_tensor_arg(),
            weights.clone().into_tensor_arg(),
            shared_gate.clone().into_tensor_arg(),
            dense.top_k as u32,
            experts,
            hidden,
        );
    }
    let weights = BurnTensor::<Metal, 2>::from_primitive(TensorPrimitive::Float(weights));
    let shared_gate = BurnTensor::<Metal, 2>::from_primitive(TensorPrimitive::Float(shared_gate));
    let weights: BurnTensor<B, 2> = unsafe { take_as(weights) };
    let shared_gate: BurnTensor<B, 2> = unsafe { take_as(shared_gate) };
    let x: BurnTensor<B, 2> = unsafe { take_as(x) };
    try_dense_experts(x, dense, weights, shared_gate)
}

/// The expert half of the block: packed gate/up over the touched experts,
/// the shared expert's gate/up, and the fused down that combines the routed
/// sum with the gated shared expert. `weights` is the router output
/// `(m, experts)` with exact zeros outside each token's top-k; `shared_gate`
/// is `(m,)`. Split out so tests can drive the expert kernels with exact
/// weights, isolating them from the router's f32 dot noise.
pub(crate) fn try_dense_experts<B: Backend>(
    x: BurnTensor<B, 2>,
    dense: &crate::moe::DenseExperts<B>,
    weights: BurnTensor<B, 2>,
    shared_gate: BurnTensor<B, 2>,
) -> Option<BurnTensor<B, 2>> {
    use burn::backend::wgpu::Metal;
    if std::any::TypeId::of::<B>() != std::any::TypeId::of::<Metal>()
        || size_of::<BurnTensor<B, 2>>() != size_of::<BurnTensor<Metal, 2>>()
    {
        return None;
    }
    let (m, hidden) = (x.dims()[0], x.dims()[1]);
    let experts = dense.num_experts;
    let inter = dense.expert_intermediate;
    let shared_inter = dense.shared_intermediate;
    if m == 0
        || m > u32::MAX as usize
        || inter % LANES as usize != 0
        || shared_inter % LANES as usize != 0
        || hidden % LANES as usize != 0
    {
        return None;
    }
    let gate_words = cube_int(dense.gate_words.clone())?;
    let gate_scales = cube_f32_1d(dense.gate_scales.clone())?;
    let up_words = cube_int(dense.up_words.clone())?;
    let up_scales = cube_f32_1d(dense.up_scales.clone())?;
    let down_words = cube_int(dense.down_words.clone())?;
    let down_scales = cube_f32_1d(dense.down_scales.clone())?;
    let shared_gate_w = cube_float2d(dense.shared_gate_w.clone())?;
    let shared_up_w = cube_float2d(dense.shared_up_w.clone())?;
    let shared_down_w = cube_float2d(dense.shared_down_w.clone())?;
    let x: BurnTensor<Metal, 2> = unsafe { take_as(x) };
    // The kernels read FP32; the model keeps activations in its device dtype.
    let x_f32 = x.cast(burn::tensor::FloatDType::F32);
    let x_row = cube_float(x_f32)?;
    let weights = cube_float2d(weights)?;
    let shared_gate = cube_float2d(shared_gate)?;
    let client = x_row.client.clone();
    let device = x_row.device.clone();
    let act = alloc_f32(&client, &device, m * experts * inter);
    let act_shared = alloc_f32(&client, &device, m * shared_inter);
    let masks = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::from([experts]),
        client.empty(experts * size_of::<u32>()),
        DType::U32,
    );
    let y = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::from([m, hidden]),
        client.empty(m * hidden * size_of::<f32>()),
        DType::F32,
    );
    let (m_len, hidden, inter, experts) = (m as u32, hidden as u32, inter as u32, experts as u32);
    let shared_inter = shared_inter as u32;
    unsafe {
        moe_masks::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(experts.div_ceil(LANES), 1, 1),
            CubeDim::new_1d(LANES),
            weights.clone().into_tensor_arg(),
            masks.clone().into_tensor_arg(),
            m_len,
            experts,
        );
        // Unrouted act slots stay unwritten; the down skips them by the same
        // zero weights, so no buffer init is needed.
        moe_gate_up_sparse::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(experts, inter / LANES, 1),
            CubeDim::new_1d(LANES),
            x_row.clone().into_tensor_arg(),
            masks.into_tensor_arg(),
            gate_words.into_tensor_arg(),
            gate_scales.into_tensor_arg(),
            up_words.into_tensor_arg(),
            up_scales.into_tensor_arg(),
            act.clone().into_tensor_arg(),
            m_len,
            hidden,
            inter,
            experts,
        );
        moe_shared_gate_up::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(m_len, shared_inter / LANES, 1),
            CubeDim::new_1d(LANES),
            x_row.into_tensor_arg(),
            shared_gate_w.into_tensor_arg(),
            shared_up_w.into_tensor_arg(),
            act_shared.clone().into_tensor_arg(),
            hidden,
            shared_inter,
        );
        moe_down_total::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(m_len, hidden / LANES, 1),
            CubeDim::new_1d(LANES),
            act.into_tensor_arg(),
            act_shared.into_tensor_arg(),
            shared_gate.into_tensor_arg(),
            weights.into_tensor_arg(),
            down_words.into_tensor_arg(),
            down_scales.into_tensor_arg(),
            shared_down_w.into_tensor_arg(),
            y.clone().into_tensor_arg(),
            hidden,
            inter,
            experts,
            shared_inter,
        );
    }
    let y = BurnTensor::<Metal, 2>::from_primitive(TensorPrimitive::Float(y));
    Some(unsafe { take_as(y) })
}

/// Router-only entry for tests: launches [`moe_router`] alone and returns
/// `(weights, shared_gate)` so the fused selection/softmax can be compared
/// with the reference without the expert kernels.
#[cfg(test)]
pub(crate) fn try_dense_router<B: Backend>(
    x: BurnTensor<B, 2>,
    router: BurnTensor<B, 2>,
    gate_vec: BurnTensor<B, 2>,
    top_k: usize,
) -> Option<(BurnTensor<B, 2>, BurnTensor<B, 2>)> {
    use burn::backend::wgpu::Metal;
    if std::any::TypeId::of::<B>() != std::any::TypeId::of::<Metal>()
        || size_of::<BurnTensor<B, 2>>() != size_of::<BurnTensor<Metal, 2>>()
    {
        return None;
    }
    let (m, hidden) = (x.dims()[0], x.dims()[1]);
    let experts = router.dims()[0];
    if m == 0 || experts > 1024 || top_k > experts || hidden == 0 {
        return None;
    }
    let router_w = cube_float2d(router.clone())?;
    let gate_vec = cube_float2d(gate_vec.clone())?;
    let x: BurnTensor<Metal, 2> = unsafe { take_as(x) };
    let x_row = cube_float(x.clone().cast(burn::tensor::FloatDType::F32))?;
    let client = x_row.client.clone();
    let device = x_row.device.clone();
    let weights = alloc_f32_2d(&client, &device, [m, experts]);
    let shared_gate = alloc_f32_2d(&client, &device, [m, 1]);
    let (m_len, hidden, experts) = (m as u32, hidden as u32, experts as u32);
    unsafe {
        moe_router::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(m_len, 1, 1),
            CubeDim::new_1d(experts),
            x_row.into_tensor_arg(),
            router_w.into_tensor_arg(),
            gate_vec.into_tensor_arg(),
            weights.clone().into_tensor_arg(),
            shared_gate.clone().into_tensor_arg(),
            top_k as u32,
            experts,
            hidden,
        );
    }
    let weights = BurnTensor::<Metal, 2>::from_primitive(TensorPrimitive::Float(weights));
    let shared_gate = BurnTensor::<Metal, 2>::from_primitive(TensorPrimitive::Float(shared_gate));
    Some((unsafe { take_as(weights) }, unsafe { take_as(shared_gate) }))
}

fn cube_int<B: Backend>(
    tensor: BurnTensor<B, 1, burn::tensor::Int>,
) -> Option<CubeTensor<WgpuRuntime>> {
    use burn::backend::wgpu::Metal;
    let tensor: BurnTensor<Metal, 1, burn::tensor::Int> = unsafe { take_as(tensor) };
    Some(burn_cubecl::kernel::into_contiguous(
        tensor.into_primitive(),
    ))
}

fn cube_f32_1d<B: Backend>(tensor: BurnTensor<B, 1>) -> Option<CubeTensor<WgpuRuntime>> {
    use burn::backend::wgpu::Metal;
    let tensor: BurnTensor<Metal, 1> = unsafe { take_as(tensor) };
    let primitive = tensor.into_primitive();
    let TensorPrimitive::Float(tensor) = primitive else {
        return None;
    };
    Some(burn_cubecl::kernel::into_contiguous(tensor))
}

fn cube_float2d<B: Backend>(tensor: BurnTensor<B, 2>) -> Option<CubeTensor<WgpuRuntime>> {
    use burn::backend::wgpu::Metal;
    let tensor: BurnTensor<Metal, 2> = unsafe { take_as(tensor) };
    cube_float(tensor)
}

fn cube_float(
    tensor: BurnTensor<burn::backend::wgpu::Metal, 2>,
) -> Option<CubeTensor<WgpuRuntime>> {
    let primitive = tensor.into_primitive();
    let TensorPrimitive::Float(tensor) = primitive else {
        return None;
    };
    Some(burn_cubecl::kernel::into_contiguous(tensor))
}

fn alloc_f32(
    client: &cubecl::client::ComputeClient<WgpuRuntime>,
    device: &cubecl::wgpu::WgpuDevice,
    values: usize,
) -> CubeTensor<WgpuRuntime> {
    CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::from([values]),
        client.empty(values * size_of::<f32>()),
        DType::F32,
    )
}

/// Outputs that leave the raw world through [`crate::cube_bridge::wrap`] must
/// carry their true rank: the fusion stream analyzes shapes and strides, and a
/// flat `[values]` shape under a rank-2 burn tensor corrupts its traces.
fn alloc_f32_2d(
    client: &cubecl::client::ComputeClient<WgpuRuntime>,
    device: &cubecl::wgpu::WgpuDevice,
    shape: [usize; 2],
) -> CubeTensor<WgpuRuntime> {
    let values = shape[0] * shape[1];
    CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::from(shape),
        client.empty(values * size_of::<f32>()),
        DType::F32,
    )
}

/// # Safety
/// `T` and `U` are the same type: the caller checked `TypeId` and size.
unsafe fn take_as<T, U>(value: T) -> U {
    let out = unsafe { std::ptr::read((&value) as *const T as *const U) };
    forget(value);
    out
}

#[cfg(test)]
mod tests {
    use super::{try_dense_experts, try_dense_router};
    use crate::moe::DenseExperts;
    use crate::quant;
    use burn::backend::wgpu::{Metal, WgpuDevice};
    use burn::tensor::{Int, Tensor, TensorData};
    use half::bf16;
    use omni_r2t2::weights::Weights;

    /// Synthetic per-expert pack-quantized weights whose packed round trip is
    /// exact (integer nibble times scale), a plain-f32 shared expert, and an
    /// explicit routing matrix driving the expert kernels directly (the
    /// router has its own test). Everything runs through the three expert
    /// launches and is compared with an f64 host reference of the whole block
    /// output: the weighted routed sum plus the gated shared expert.
    /// `experts`/`inter`/`hidden` cover the toy shapes and the 35B layer
    /// shapes.
    fn run_grouped_reference(experts: usize, inter: usize, hidden: usize, m: usize) {
        run_grouped_reference_with(experts, inter, hidden, m, false, false, 128);
    }

    /// With `delta_down`, the routed expert's down projection is the identity
    /// (down[h, j] = 1 iff j == h), so `y[h]` reads out `act[h]` directly and
    /// the failing stage (gate/up versus down/combine) is identifiable.
    #[allow(clippy::too_many_arguments)]
    fn run_grouped_reference_with(
        experts: usize,
        inter: usize,
        hidden: usize,
        m: usize,
        delta_down: bool,
        single_full_weight: bool,
        shared_inter: usize,
    ) {
        type B = Metal;
        let device = WgpuDevice::default();
        let group = quant::GROUP_SIZE;
        // Scales are decided once per (seed, row, group) and shared by the
        // packing and the reference, so the packed tensor reproduces the
        // reference values exactly. The quantized integer stays in i32:
        // `% 15 - 7` on usize would underflow for the negative half.
        // The checkpoint stores bf16 scales; loaders round-trip them, so the
        // packing and the reference must both use the rounded value.
        let scale = |seed: usize, row: usize, g: usize| {
            bf16::from_f32(0.006 + ((seed + row * 31 + g * 7) % 19) as f32 * 0.0009).to_f32()
        };
        let quantized = |seed: usize, a: usize, b: usize, c: usize| -> i32 {
            ((seed + a * 13 + b * 31 + c * 5) % 15) as i32 - 7
        };
        let gate = |e: usize, j: usize, k: usize| {
            quantized(e * 3, e, j, k) as f64 * scale(e * 3, j, k / group) as f64
        };
        let up = |e: usize, j: usize, k: usize| {
            quantized(e * 5, e, j, k) as f64 * scale(e * 5 + 1, j, k / group) as f64
        };
        let down = |e: usize, h: usize, j: usize| {
            if delta_down {
                // +1 times its group scale, or exact zero: both representable.
                if h < inter && j == h {
                    scale(e * 7 + 2, h, j / group) as f64
                } else {
                    0.0
                }
            } else {
                quantized(e * 7, e, h, j) as f64 * scale(e * 7 + 2, h, j / group) as f64
            }
        };
        // Shared expert: plain f32 weights; the reference reads the same f32
        // values promoted to f64.
        let shared_scale = |seed: usize, i: usize| (((seed + i * 13) % 11) as f32 - 5.0) * 0.05f32;
        let sgate =
            |j: usize, k: usize| (shared_scale(1, j * 3 + k % 7) * shared_scale(2, k)) as f64;
        let sup = |j: usize, k: usize| (shared_scale(3, j + k) * shared_scale(4, j * 5 + k)) as f64;
        let sdown =
            |h: usize, j: usize| (shared_scale(5, h + j * 3) * shared_scale(6, h * 7 + j)) as f64;
        // Build a safetensors buffer with the checkpoint layout per expert.
        let mut entries: Vec<(String, String, Vec<usize>, Vec<u8>)> = Vec::new();
        for e in 0..experts {
            let pack = |value: &dyn Fn(usize, usize) -> f64,
                        scale_at: &dyn Fn(usize, usize) -> f32,
                        rows: usize,
                        cols: usize|
             -> Vec<u8> {
                let mut bytes = Vec::new();
                for r in 0..rows {
                    for c_idx in 0..cols / 2 {
                        let nibble = |index: usize| {
                            ((value(r, index) / scale_at(r, index / group) as f64).round() + 8.0)
                                as u8
                        };
                        bytes
                            .push((nibble(c_idx * 2) & 0xF) | ((nibble(c_idx * 2 + 1) & 0xF) << 4));
                    }
                }
                bytes
            };
            let scale_bytes = |seed: usize, rows: usize, cols: usize| -> Vec<u8> {
                (0..rows * cols / group)
                    .flat_map(|i| {
                        bf16::from_f32(scale(seed, i / (cols / group), i % (cols / group)))
                            .to_bits()
                            .to_le_bytes()
                            .to_vec()
                    })
                    .collect()
            };
            entries.push((
                format!("e{e}.gate.weight_packed"),
                "U8".into(),
                vec![inter, hidden / 2],
                pack(
                    &|j, k| gate(e, j, k),
                    &|r, g| scale(e * 3, r, g),
                    inter,
                    hidden,
                ),
            ));
            entries.push((
                format!("e{e}.gate.weight_scale"),
                "BF16".into(),
                vec![inter, hidden / group],
                scale_bytes(e * 3, inter, hidden),
            ));
            entries.push((
                format!("e{e}.up.weight_packed"),
                "U8".into(),
                vec![inter, hidden / 2],
                pack(
                    &|j, k| up(e, j, k),
                    &|r, g| scale(e * 5 + 1, r, g),
                    inter,
                    hidden,
                ),
            ));
            entries.push((
                format!("e{e}.up.weight_scale"),
                "BF16".into(),
                vec![inter, hidden / group],
                scale_bytes(e * 5 + 1, inter, hidden),
            ));
            entries.push((
                format!("e{e}.down.weight_packed"),
                "U8".into(),
                vec![hidden, inter / 2],
                pack(
                    &|h, j| down(e, h, j),
                    &|r, g| scale(e * 7 + 2, r, g),
                    hidden,
                    inter,
                ),
            ));
            entries.push((
                format!("e{e}.down.weight_scale"),
                "BF16".into(),
                vec![hidden, inter / group],
                scale_bytes(e * 7 + 2, hidden, inter),
            ));
        }
        let mut header = String::from("{");
        let mut offset = 0usize;
        for (name, dtype, shape, payload) in &entries {
            header.push_str(&format!(
                "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{},{}],\"data_offsets\":[{offset},{}]}},",
                shape[0],
                shape[1],
                offset + payload.len()
            ));
            offset += payload.len();
        }
        header.pop();
        header.push('}');
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        for (_, _, _, payload) in &entries {
            bytes.extend_from_slice(payload);
        }
        let weights = Weights::from_bytes(vec![bytes]).unwrap();
        let upload = |parts: quant::PackedParts| {
            (
                Tensor::<B, 1, Int>::from_data(
                    TensorData::new(
                        parts.words.iter().map(|word| *word as i32).collect(),
                        [parts.words.len()],
                    ),
                    burn::tensor::TensorCreationOptions::<B> {
                        device: device.clone(),
                        dtype: Some(burn::tensor::DType::I32),
                    },
                ),
                Tensor::<B, 1>::from_data(
                    TensorData::new(parts.scales.clone(), [parts.scales.len()]),
                    burn::tensor::TensorCreationOptions::<B> {
                        device: device.clone(),
                        dtype: Some(burn::tensor::DType::F32),
                    },
                ),
            )
        };
        let gate_names: Vec<String> = (0..experts)
            .map(|e| format!("e{e}.gate.weight_packed"))
            .collect();
        let gate_scales: Vec<String> = (0..experts)
            .map(|e| format!("e{e}.gate.weight_scale"))
            .collect();
        let up_names: Vec<String> = (0..experts)
            .map(|e| format!("e{e}.up.weight_packed"))
            .collect();
        let up_scales: Vec<String> = (0..experts)
            .map(|e| format!("e{e}.up.weight_scale"))
            .collect();
        let down_names: Vec<String> = (0..experts)
            .map(|e| format!("e{e}.down.weight_packed"))
            .collect();
        let down_scales: Vec<String> = (0..experts)
            .map(|e| format!("e{e}.down.weight_scale"))
            .collect();
        let gate_parts = quant::packed_vconcat_kmajor_parts::<B>(
            &weights,
            &gate_names,
            &gate_scales,
            [inter, hidden],
        )
        .unwrap();
        let up_parts = quant::packed_vconcat_kmajor_parts::<B>(
            &weights,
            &up_names,
            &up_scales,
            [inter, hidden],
        )
        .unwrap();
        let (gate_words, gate_scales) = upload(gate_parts);
        let (up_words, up_scales) = upload(up_parts);
        let (down_words, down_scales) = upload(
            quant::packed_hconcat_kmajor_parts::<B>(
                &weights,
                &down_names,
                &down_scales,
                [hidden, inter],
            )
            .unwrap(),
        );
        let upload2d = |values: Vec<f32>, shape: [usize; 2]| {
            Tensor::<B, 2>::from_data(
                TensorData::new(values, shape),
                burn::tensor::TensorCreationOptions::<B> {
                    device: device.clone(),
                    dtype: Some(burn::tensor::DType::F32),
                },
            )
        };
        let shared_gate_w = upload2d(
            (0..shared_inter * hidden)
                .map(|i| sgate(i / hidden, i % hidden) as f32)
                .collect(),
            [shared_inter, hidden],
        );
        let shared_up_w = upload2d(
            (0..shared_inter * hidden)
                .map(|i| sup(i / hidden, i % hidden) as f32)
                .collect(),
            [shared_inter, hidden],
        );
        let shared_down_w = upload2d(
            (0..hidden * shared_inter)
                .map(|i| sdown(i / shared_inter, i % shared_inter) as f32)
                .collect(),
            [hidden, shared_inter],
        );
        // The router row and gate vector are unused by the expert kernels;
        // fill them with small placeholders.
        let router_w = upload2d(vec![0.01f32; experts * hidden], [experts, hidden]);
        let gate_vec_w = upload2d(vec![0.0f32; hidden], [1, hidden]);
        let dense = DenseExperts {
            gate_words,
            gate_scales,
            up_words,
            up_scales,
            down_words,
            down_scales,
            router: router_w,
            shared_gate_w,
            shared_up_w,
            shared_down_w,
            shared_gate_vec: gate_vec_w,
            num_experts: experts,
            top_k: 2.min(experts),
            expert_intermediate: inter,
            shared_intermediate: shared_inter,
        };
        let x_values: Vec<f32> = (0..m * hidden)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.25)
            .collect();
        let x = Tensor::<B, 2>::from_data(
            TensorData::new(x_values.clone(), [m, hidden]),
            burn::tensor::TensorCreationOptions::<B> {
                device: device.clone(),
                dtype: Some(burn::tensor::DType::F32),
            },
        );
        // Token m routes to two of the first few experts so the f64 reference
        // stays cheap at model scale.
        let routed = experts.min(4);
        let mut router = vec![0f32; m * experts];
        for token in 0..m {
            if single_full_weight {
                router[token * experts] = 1.0;
                continue;
            }
            let a = token % routed;
            let b = (token + 3) % routed;
            router[token * experts + a] = 0.7;
            if b != a {
                router[token * experts + b] = 0.3;
            }
        }
        let weights_matrix = upload2d(router.clone(), [m, experts]);
        // The sigmoid gate values are inputs here; the reference reads the
        // same f32 numbers.
        let gate_values: Vec<f32> = (0..m).map(|t| 0.25 + (t % 3) as f32 * 0.2).collect();
        let shared_gate_in = upload2d(gate_values.clone(), [m, 1]);
        let y = try_dense_experts(x.clone(), &dense, weights_matrix, shared_gate_in)
            .expect("metal dense moe");
        let got = y
            .into_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        let mut worst = 0f64;
        // The same reference with the router weight squared: what a kernel
        // that weights both the activation and the combine would produce. The
        // gap proves the test can see that mistake.
        let mut worst_squared = 0f64;
        for token in 0..m {
            let row = &x_values[token * hidden..(token + 1) * hidden];
            // Shared expert activations and gate for this token.
            let mut act_s = vec![0f64; shared_inter];
            for j in 0..shared_inter {
                let mut g = 0f64;
                let mut u = 0f64;
                for k in 0..hidden {
                    let xk = row[k] as f64;
                    g += xk * sgate(j, k);
                    u += xk * sup(j, k);
                }
                act_s[j] = (g / (1.0 + (-g).exp())) * u;
            }
            let shared_gate_value = gate_values[token] as f64;
            let mut expected = vec![0f64; hidden];
            let mut squared = vec![0f64; hidden];
            for e in 0..experts {
                let weight = router[token * experts + e] as f64;
                if weight == 0.0 {
                    continue;
                }
                let mut act = vec![0f64; inter];
                for j in 0..inter {
                    let mut g = 0f64;
                    let mut u = 0f64;
                    for k in 0..hidden {
                        let xk = row[k] as f64;
                        g += xk * gate(e, j, k);
                        u += xk * up(e, j, k);
                    }
                    act[j] = (g / (1.0 + (-g).exp())) * u;
                }
                for h in 0..hidden {
                    let mut acc = 0f64;
                    for j in 0..inter {
                        acc += act[j] * down(e, h, j);
                    }
                    expected[h] += weight * acc;
                    squared[h] += weight * weight * acc;
                }
            }
            for h in 0..hidden {
                let mut shared_acc = 0f64;
                for j in 0..shared_inter {
                    shared_acc += act_s[j] * sdown(h, j);
                }
                expected[h] += shared_gate_value * shared_acc;
                squared[h] += shared_gate_value * shared_acc;
                let value = got[token * hidden + h] as f64;
                worst = worst.max((value - expected[h]).abs());
                worst_squared = worst_squared.max((value - squared[h]).abs());
            }
        }
        // FP32 accumulation in the kernels against this f64 reference; the
        // same budget the packed-matmul fixtures use.
        assert!(worst < 5e-2, "worst {worst}");
        // Meaningless at weight 1.0, where squaring changes nothing.
        if router.iter().any(|weight| *weight > 0.0 && *weight != 1.0) {
            assert!(
                worst_squared > 5.0 * worst.max(1e-6),
                "reference cannot distinguish a doubled router weight: {worst_squared} vs {worst}"
            );
        }
    }

    #[test]
    fn grouped_kernels_match_routed_reference() {
        run_grouped_reference(8, 128, 256, 4);
    }

    /// The 35B layer shapes: index overflow and layout mistakes only show up
    /// at these sizes.
    #[test]
    fn grouped_kernels_match_routed_reference_at_model_scale() {
        run_grouped_reference_with(256, 512, 2048, 19, false, false, 1024);
    }

    /// Isolated wall timing of the grouped kernels at the 35B layer shapes
    /// with realistic dispersed routing (16 tokens, top-8 random experts
    /// each, ~110 distinct experts touched). The cubecl per-launch profile
    /// log charges queue-drain time to whichever kernel a sync lands in, so
    /// its per-kernel numbers are not comparable across code variants; this
    /// bench times each entry point alone with a single trailing sync.
    /// Values are synthetic — only timing matters.
    ///
    /// ```text
    /// cargo test --release --features metal --lib moe_bench -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "diagnostic bench: model-scale synthetic weights"]
    fn moe_bench_model_scale() {
        use burn::tensor::backend::Backend as _;
        type B = Metal;
        let device = WgpuDevice::default();
        let (experts, inter, hidden, shared_inter, m) =
            (256usize, 512usize, 2048usize, 1024usize, 16usize);
        let word_count = experts * inter * hidden / 8;
        let scale_count = experts * inter * hidden / 128;
        // Multiplicative hashing keeps the 100M-word generation to seconds.
        let words = |seed: u32| -> Vec<i32> {
            (0..word_count)
                .map(|i| (i as u32).wrapping_mul(2654_435_761).wrapping_add(seed) as i32)
                .collect()
        };
        let scales = |seed: u32| -> Vec<f32> {
            (0..scale_count)
                .map(|i: usize| {
                    ((i as u32).wrapping_mul(2654_435_761).wrapping_add(seed) % 97) as f32 * 1e-4
                        + 1e-4
                })
                .collect()
        };
        let int_opts = burn::tensor::TensorCreationOptions::<B> {
            device: device.clone(),
            dtype: Some(burn::tensor::DType::I32),
        };
        let float_opts = burn::tensor::TensorCreationOptions::<B> {
            device: device.clone(),
            dtype: Some(burn::tensor::DType::F32),
        };
        let int_1d = |values: Vec<i32>| {
            Tensor::<B, 1, Int>::from_data(TensorData::new(values, [word_count]), int_opts.clone())
        };
        let f32_1d = |values: Vec<f32>| {
            Tensor::<B, 1>::from_data(TensorData::new(values, [scale_count]), float_opts.clone())
        };
        let f32_2d = |values: Vec<f32>, shape: [usize; 2]| {
            Tensor::<B, 2>::from_data(TensorData::new(values, shape), float_opts.clone())
        };
        let shared_pattern = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 7) % 13) as f32 - 6.0) * 0.03)
                .collect()
        };
        let dense = DenseExperts {
            gate_words: int_1d(words(1)),
            gate_scales: f32_1d(scales(2)),
            up_words: int_1d(words(3)),
            up_scales: f32_1d(scales(4)),
            down_words: int_1d(words(5)),
            down_scales: f32_1d(scales(6)),
            router: f32_2d(
                (0..experts * hidden)
                    .map(|i| (((i * 11) % 17) as f32 - 8.0) * 0.01)
                    .collect(),
                [experts, hidden],
            ),
            shared_gate_w: f32_2d(
                shared_pattern(shared_inter * hidden),
                [shared_inter, hidden],
            ),
            shared_up_w: f32_2d(
                shared_pattern(shared_inter * hidden),
                [shared_inter, hidden],
            ),
            shared_down_w: f32_2d(
                shared_pattern(hidden * shared_inter),
                [hidden, shared_inter],
            ),
            shared_gate_vec: f32_2d(
                (0..hidden)
                    .map(|k| (((k * 5) % 9) as f32 - 4.0) * 0.05)
                    .collect(),
                [1, hidden],
            ),
            num_experts: experts,
            top_k: 8,
            expert_intermediate: inter,
            shared_intermediate: shared_inter,
        };
        let x = f32_2d(
            (0..m * hidden)
                .map(|i| ((i % 23) as f32 - 11.0) * 0.25)
                .collect(),
            [m, hidden],
        );
        // Dispersed routing: token t picks eight strided experts, like the
        // real verify block's ~110 distinct experts over sixteen tokens.
        let mut routing = vec![0f32; m * experts];
        for t in 0..m {
            for k in 0..8 {
                let e = (t * 37 + k * 31 + t * k) % experts;
                routing[t * experts + e] = 0.125;
            }
        }
        let weights_matrix = f32_2d(routing, [m, experts]);
        let shared_gate = f32_2d(vec![0.5f32; m], [m, 1]);
        let bench = |label: &str, iters: usize, run: &mut dyn FnMut()| {
            for _ in 0..3 {
                run();
            }
            B::sync(&device).unwrap();
            let start = std::time::Instant::now();
            for _ in 0..iters {
                run();
            }
            B::sync(&device).unwrap();
            let per = start.elapsed().as_secs_f64() / iters as f64;
            eprintln!("{label}: {per:.6?} s/iter");
        };
        bench("router (moe_router)", 100, &mut || {
            let _ = super::try_dense_router(
                x.clone(),
                dense.router.clone(),
                dense.shared_gate_vec.clone(),
                8,
            );
        });
        // top_k is comptime: sweeping it splits the launch's cost between the
        // logit dots/reads (top_k-independent) and the argmax scan machinery.
        bench("router top_k=1 (dot+read only)", 100, &mut || {
            let _ = super::try_dense_router(
                x.clone(),
                dense.router.clone(),
                dense.shared_gate_vec.clone(),
                1,
            );
        });
        // The verify block's projection matmuls for comparison: a qkv-shaped
        // and an o/gdn-projection-shaped F16 burn matmul, the shapes the
        // dspark verify path issues hundreds of times per forward.
        let proj_opts = burn::tensor::TensorCreationOptions::<B> {
            device: device.clone(),
            dtype: Some(burn::tensor::DType::F16),
        };
        let a16 = |n: usize| {
            Tensor::<B, 2>::from_data(
                TensorData::new(
                    (0..m * n)
                        .map(|i| half::f16::from_f32((i % 17) as f32 * 0.1))
                        .collect(),
                    [m, n],
                ),
                proj_opts.clone(),
            )
        };
        let w16 = |rows: usize, cols: usize| {
            Tensor::<B, 2>::from_data(
                TensorData::new(
                    (0..rows * cols)
                        .map(|i| half::f16::from_f32(((i * 7) % 13) as f32 * 0.1))
                        .collect(),
                    [rows, cols],
                ),
                proj_opts.clone(),
            )
        };
        let x16 = a16(hidden);
        let qkv_w = w16(hidden, 6144);
        let o_w = w16(hidden, hidden);
        bench("matmul qkv [16,2048]x[2048,6144] f16", 200, &mut || {
            let _ = x16.clone().matmul(qkv_w.clone());
        });
        bench("matmul proj [16,2048]x[2048,2048] f16", 200, &mut || {
            let _ = x16.clone().matmul(o_w.clone());
        });
        bench(
            "experts (masks+gate_up_sparse+shared+down)",
            30,
            &mut || {
                let _ = try_dense_experts(
                    x.clone(),
                    &dense,
                    weights_matrix.clone(),
                    shared_gate.clone(),
                );
            },
        );
        eprintln!("done");
    }

    #[test]
    fn grouped_kernels_single_token() {
        run_grouped_reference(8, 128, 256, 1);
    }

    /// The 16-token verify block takes the vector gate/up kernel.
    #[test]
    fn grouped_kernels_verify_block_vector_path() {
        run_grouped_reference(8, 128, 256, 16);
    }

    #[test]
    fn grouped_kernels_delta_down_reads_act() {
        run_grouped_reference_with(8, 128, 128, 1, true, false, 128);
    }

    #[test]
    fn grouped_kernels_delta_down_reads_act_wide() {
        run_grouped_reference_with(8, 128, 256, 1, true, false, 128);
    }

    #[test]
    fn grouped_kernels_delta_e16() {
        run_grouped_reference_with(16, 128, 128, 1, true, false, 128);
    }

    #[test]
    fn grouped_kernels_delta_inter256() {
        run_grouped_reference_with(8, 256, 128, 1, true, false, 128);
    }

    /// Real down weights, one routed expert at weight 1.0: the combine is a
    /// copy, so a mismatch isolates the down kernel.
    #[test]
    fn grouped_kernels_single_expert_full_weight() {
        run_grouped_reference_with(8, 128, 256, 1, false, true, 128);
    }

    /// The fused router against an f64 mirror of the same arithmetic:
    /// constant-row router weights give logits `c_e * sum(x_row)`, so the
    /// top-k set is known analytically and tie-free. The weight tolerance is
    /// the router dot's own f32 accumulation noise (the x bias walks the
    /// partial sums to ~1000, worth ~3e-4 in the softmax); the selection set,
    /// the exact zeros outside it and the sigmoid gate are checked tightly.
    #[test]
    fn router_kernel_weights_match_reference_at_model_scale() {
        type B = Metal;
        let device = WgpuDevice::default();
        let (experts, hidden, m, top_k) = (256usize, 2048usize, 16usize, 8usize);
        let c = |e: usize| 1.0 + e as f32 * 0.0001;
        // The +0.5 bias keeps every row sum well away from zero so the
        // mirrored selection cannot flip on an ulp.
        let x_values: Vec<f32> = (0..m * hidden)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.25 + 0.5)
            .collect();
        let opts = burn::tensor::TensorCreationOptions::<B> {
            device: device.clone(),
            dtype: Some(burn::tensor::DType::F32),
        };
        let x =
            Tensor::<B, 2>::from_data(TensorData::new(x_values.clone(), [m, hidden]), opts.clone());
        let router_w = Tensor::<B, 2>::from_data(
            TensorData::new(
                (0..experts * hidden)
                    .map(|i| c(i / hidden))
                    .collect::<Vec<f32>>(),
                [experts, hidden],
            ),
            opts.clone(),
        );
        let gate_vec_values: Vec<f32> = (0..hidden)
            .map(|k| (((k * 7) % 9) as f32 - 4.0) * 0.06)
            .collect();
        let gate_vec =
            Tensor::<B, 2>::from_data(TensorData::new(gate_vec_values.clone(), [1, hidden]), opts);
        let (weights, shared_gate) =
            try_dense_router(x.clone(), router_w, gate_vec, top_k).expect("metal router");
        let weights = weights
            .into_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        let shared_gate = shared_gate
            .into_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        let mut worst_weight = 0f64;
        let mut worst_gate = 0f64;
        for token in 0..m {
            let row = &x_values[token * hidden..(token + 1) * hidden];
            let row_sum: f64 = row.iter().map(|v| *v as f64).sum();
            let logits: Vec<f64> = (0..experts).map(|e| c(e) as f64 * row_sum).collect();
            let mut order: Vec<usize> = (0..experts).collect();
            order.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]).then(a.cmp(b)));
            let top = &order[..top_k];
            let max = logits[top[0]];
            let exps: Vec<f64> = top.iter().map(|e| (logits[*e] - max).exp()).collect();
            let exp_sum: f64 = exps.iter().sum();
            for e in top {
                let slot = top.iter().position(|t| *t == *e).unwrap();
                let expected = exps[slot] / exp_sum;
                let got = weights[token * experts + e] as f64;
                worst_weight = worst_weight.max((got - expected).abs());
            }
            let outside_sum: f64 = (0..experts)
                .filter(|e| !top.contains(e))
                .map(|e| weights[token * experts + e] as f64)
                .sum();
            assert!(outside_sum == 0.0, "outside weights not zero");
            let mut gate_acc = 0f64;
            for k in 0..hidden {
                gate_acc += row[k] as f64 * gate_vec_values[k] as f64;
            }
            let expected_gate = 1.0 / (1.0 + (-gate_acc).exp());
            worst_gate = worst_gate.max((shared_gate[token] as f64 - expected_gate).abs());
        }
        assert!(worst_weight < 1e-3, "worst weight {worst_weight}");
        assert!(worst_gate < 1e-4, "worst gate {worst_gate}");
    }
}
