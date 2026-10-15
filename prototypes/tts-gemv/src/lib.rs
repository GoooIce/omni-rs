use burn::tensor::f16;
use burn::tensor::{DType, Shape, Tensor as BurnTensor, TensorPrimitive};
use burn_cubecl::{CubeBackend, tensor::CubeTensor};
use burn_fusion::{
    Fusion, FusionBackend, FusionRuntime,
    stream::{Operation, OperationStreams},
};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};
use cubecl::{prelude::*, wgpu::WgpuRuntime};

pub type Raw = CubeBackend<WgpuRuntime, f32, i32, u8>;
pub type B = Fusion<Raw>;
type Runtime = <Raw as FusionBackend>::FusionRuntime;

// One SIMD group reduces one output channel. The original Linear's transposed
// view still stores each output row contiguously. No weight rearrange or copy.
#[cube(launch_unchecked)]
fn gemv_kernel(
    x: &Tensor<f16>,
    w: &Tensor<f16>,
    out: &mut Tensor<f16>,
    #[comptime] k: usize,
    #[comptime] n: usize,
) {
    let lane = UNIT_POS % PLANE_DIM;
    let row = (CUBE_POS_X * 4 + UNIT_POS / PLANE_DIM) as usize;
    let batch = CUBE_POS_Y as usize;
    let mut sum = 0.0f32;
    if row < n {
        for i in 0..k / 32 {
            let col = i * 32 + lane as usize;
            sum += f32::cast_from(x[batch * k + col]) * f32::cast_from(w[row * k + col]);
        }
    }
    sum = plane_sum(sum);
    if lane == 0 && row < n {
        out[batch * n + row] = f16::cast_from(sum);
    }
}

pub fn raw_gemv(x: CubeTensor<WgpuRuntime>, w: CubeTensor<WgpuRuntime>) -> CubeTensor<WgpuRuntime> {
    let m = x.meta.shape()[0];
    let k = x.meta.shape()[1];
    let n = w.meta.shape()[1];
    assert_eq!(x.dtype, DType::F16);
    assert_eq!(w.dtype, DType::F16);
    assert_eq!(x.meta.strides()[1], 1);
    assert_eq!(x.meta.strides()[0], k);
    assert_eq!(w.meta.strides()[0], 1);
    assert_eq!(w.meta.strides()[1], k);
    assert_eq!(k % 32, 0);
    let out = CubeTensor::new_contiguous(
        x.client.clone(),
        x.device.clone(),
        Shape::new([m, n]),
        x.client.empty(m * n * 2),
        DType::F16,
    );
    unsafe {
        gemv_kernel::launch_unchecked::<WgpuRuntime>(
            &x.client,
            CubeCount::Static(n.div_ceil(4) as u32, m as u32, 1),
            CubeDim::new_1d(128),
            x.clone().into_tensor_arg(),
            w.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            k,
            n,
        );
    }
    out
}

#[derive(Clone, Debug)]
struct GemvOp(CustomOpIr);
impl Operation<Runtime> for GemvOp {
    fn execute(&self, handles: &mut HandleContainer<<Runtime as FusionRuntime>::FusionHandle>) {
        let ([x, w], [out]) = self.0.as_fixed();
        let output = raw_gemv(
            handles.get_float_tensor::<Raw>(x),
            handles.get_float_tensor::<Raw>(w),
        );
        handles.register_float_tensor::<Raw>(&out.id, output);
    }
}

pub fn gemv(x: BurnTensor<B, 2>, w: BurnTensor<B, 2>) -> BurnTensor<B, 2> {
    let x = x.into_primitive().tensor();
    let w = w.into_primitive().tensor();
    let client = x.client.clone();
    let streams = OperationStreams::with_inputs([&x, &w]);
    let out = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([x.shape[0], w.shape[1]]),
        x.dtype,
    );
    let desc = CustomOpIr::new("breeze_f16_gemv", &[x.into_ir(), w.into_ir()], &[out]);
    BurnTensor::from_primitive(TensorPrimitive::Float(
        client
            .register(streams, OperationIr::Custom(desc.clone()), GemvOp(desc))
            .output(),
    ))
}
