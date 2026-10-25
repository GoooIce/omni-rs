use super::init_matmul_output;
use crate::{CubeRuntime, kernel::quantization::dequantize, tensor::CubeTensor};
use burn_backend::{DType, QTensorPrimitive};
use burn_std::QuantLevel;
use cubecl::features::TypeUsage;
use cubecl::std::tensor::{MatrixBatchLayout, matrix_batch_layout};
use cubek::matmul::components::tile::TileMatmulKind;
use cubek::matmul::definition::{AvailableVectorSizes, MatmulProblem};
use cubek::matmul::launch::{TensorArgs, launch_kernel_concrete};
use cubek::matmul::routines::simple::{SimpleAlgorithm, SimpleArgs};
use cubek::matmul::routines::{BlueprintStrategy, Routine};
use cubek::{
    matmul::{
        definition::{MatmulElems, MatmulGlobalElems, MatmulSetupError},
        launch::Strategy,
    },
    std::InputBinding,
};
use std::cell::Cell;

#[cfg(feature = "autotune")]
use super::matmul_autotune;

/// The strategy to be used when launching a matmul kernel.
pub enum MatmulStrategy {
    #[cfg(feature = "autotune")]
    /// Using autotune to choose the best kernel based on runtime information.
    Autotune,
    /// Cube implementation of matmul.
    Cube,
}

impl Default for MatmulStrategy {
    fn default() -> Self {
        // if autotune is enabled, default to autotune
        #[cfg(feature = "autotune")]
        return MatmulStrategy::Autotune;

        #[cfg(not(feature = "autotune"))]
        MatmulStrategy::Cube
    }
}

/// Launch a matmul kernel using the given strategy.
pub fn matmul<R: CubeRuntime>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: Option<CubeTensor<R>>,
    strategy: MatmulStrategy,
    out_dtype: DType,
) -> Result<CubeTensor<R>, MatmulSetupError> {
    match strategy {
        MatmulStrategy::Cube => {
            let out = out.unwrap_or_else(|| init_matmul_output(&lhs, &rhs, out_dtype));
            launch_matmul(&Default::default(), lhs, rhs, out.clone())?;
            Ok(out)
        }
        #[cfg(feature = "autotune")]
        MatmulStrategy::Autotune => Ok(matmul_autotune(lhs, rhs, out, out_dtype)),
    }
}

pub(crate) fn launch_matmul_naive<R: CubeRuntime>(
    strategy: &Strategy,
    mut lhs: CubeTensor<R>,
    mut rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), MatmulSetupError> {
    // Naive has very specific layout requirements for block scaled tensors, so we need to manually
    // dequantize if it fails to launch normally. This is because naive is assumed to always work.
    if lhs.qparams.is_some() || rhs.qparams.is_some() {
        match launch_matmul(strategy, lhs.clone(), rhs.clone(), out.clone()) {
            Err(_) => {
                if lhs.qparams.is_some() {
                    lhs = dequantize(lhs, out.dtype);
                }
                if rhs.qparams.is_some() {
                    rhs = dequantize(rhs, out.dtype);
                }
                launch_matmul(strategy, lhs, rhs, out)
            }
            Ok(_) => Ok(()),
        }
    } else {
        launch_matmul(strategy, lhs, rhs, out)
    }
}

pub(crate) fn launch_matmul<R: CubeRuntime>(
    strategy: &Strategy,
    lhs: CubeTensor<R>,
    mut rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), MatmulSetupError> {
    let client = &out.client;

    let lhs_quant_handles = lhs.quantized_handles();
    let out_dtype: DType = out.dtype;

    let (lhs_dtype, lhs_handle) = match lhs_quant_handles {
        None => {
            let lhs_dtype = lhs.dtype;
            (
                lhs_dtype,
                InputBinding::new(lhs.binding(), lhs_dtype.into()),
            )
        }
        Some((data, scale)) => {
            let scheme = *lhs.scheme();
            let data_dtype = data.dtype;
            let scale_dtype = scale.dtype;
            (
                out_dtype,
                InputBinding::quantized(
                    data.binding(),
                    scale.binding(),
                    lhs.meta.shape().clone(),
                    scheme,
                    data_dtype.into(),
                    scale_dtype.into(),
                ),
            )
        }
    };

    let rhs_quant_handles = rhs.quantized_handles();

    let (rhs_dtype, rhs_handle) = match rhs_quant_handles {
        None => (
            lhs_dtype,
            InputBinding::new(rhs.binding(), lhs_dtype.into()),
        ),
        Some((data, scale)) => {
            // Extremely hacky fix to ensure naive can run in every case
            if matches!(strategy, Strategy::Naive)
                && matches!(rhs.scheme().level, QuantLevel::Block(_))
            {
                rhs = dequantize(rhs.clone(), lhs_dtype);
                let rhs_dtype = rhs.dtype;
                (
                    lhs_dtype,
                    InputBinding::new(rhs.binding(), rhs_dtype.into()),
                )
            } else {
                let scheme = *rhs.scheme();
                let data_dtype = data.dtype;
                let scale_dtype = scale.dtype;
                (
                    out_dtype,
                    InputBinding::quantized(
                        data.binding(),
                        scale.binding(),
                        rhs.meta.shape().clone(),
                        scheme,
                        data_dtype.into(),
                        scale_dtype.into(),
                    ),
                )
            }
        }
    };

    let mut dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: lhs_dtype.into(),
        rhs: rhs_dtype.into(),
        out: out_dtype.into(),
    });

    cubek::matmul::launch::launch_ref(
        strategy,
        client,
        lhs_handle,
        rhs_handle,
        out.clone().binding(),
        &mut dtypes,
    )?;

    Ok(())
}

thread_local! {
    static DECODE_TILE_ROWS: Cell<bool> = const { Cell::new(false) };
}

/// While this guard is alive, a short matmul uses the CMMA tile selected for
/// one row, so a verify block and a single decode share that dot product.
pub struct DecodeTileRows {
    previous: bool,
}

impl DecodeTileRows {
    /// Score a short verify block with the decode matmul tile.
    pub fn enter() -> Self {
        let previous = DECODE_TILE_ROWS.with(|flag| flag.replace(true));
        Self { previous }
    }
}

impl Drop for DecodeTileRows {
    fn drop(&mut self) {
        DECODE_TILE_ROWS.with(|flag| flag.set(self.previous));
    }
}

pub(crate) fn decode_tile_rows() -> bool {
    DECODE_TILE_ROWS.with(|flag| flag.get())
}

/// One quantized matmul whose tile was chosen as if `m = 1`.
pub(crate) fn try_launch_decode_tile<R: CubeRuntime>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), MatmulSetupError> {
    let client = out.client.clone();
    let out_dtype = out.dtype;
    let lhs_dtype = if lhs.qparams.is_some() {
        out_dtype
    } else {
        lhs.dtype
    };
    let rhs_dtype = if rhs.qparams.is_some() {
        out_dtype
    } else {
        lhs_dtype
    };

    let lhs_handle = quantized_binding(&lhs);
    let rhs_handle = quantized_binding(&rhs);
    let lhs_handle = match matrix_batch_layout(&lhs_handle.data().strides, lhs_handle.scheme()) {
        MatrixBatchLayout::HighlyPermuted => lhs_handle.into_contiguous(&client)?,
        _ => lhs_handle,
    };
    let rhs_handle = match matrix_batch_layout(&rhs_handle.data().strides, rhs_handle.scheme()) {
        MatrixBatchLayout::HighlyPermuted => rhs_handle.into_contiguous(&client)?,
        _ => rhs_handle,
    };

    let mut dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: lhs_dtype.into(),
        rhs: rhs_dtype.into(),
        out: out_dtype.into(),
    });
    if !client
        .properties()
        .features
        .type_usage(dtypes.lhs_global)
        .contains(TypeUsage::Conversion)
        || !client
            .properties()
            .features
            .type_usage(dtypes.rhs_global)
            .contains(TypeUsage::Conversion)
        || !client
            .properties()
            .features
            .type_usage(dtypes.acc_global)
            .contains(TypeUsage::Conversion)
    {
        return Err(MatmulSetupError::Unavailable(
            cubek::matmul::definition::MatmulAvailabilityError::TypesUnavailable {
                lhs: dtypes.lhs_global,
                rhs: dtypes.rhs_global,
                output: dtypes.acc_global,
            },
        ));
    }

    let out_binding = out.binding();
    let address_type = lhs_handle
        .required_address_type()
        .max(rhs_handle.required_address_type())
        .max(out_binding.required_address_type(dtypes.acc_global.size()));
    let problem = MatmulProblem::from_shapes_and_strides(
        lhs_handle.shape().clone(),
        rhs_handle.shape().clone(),
        out_binding.shape.clone(),
        lhs_handle.data().strides.clone(),
        rhs_handle.data().strides.clone(),
        out_binding.strides.clone(),
        dtypes.as_global_elems(),
        address_type,
        lhs_handle.scheme(),
        rhs_handle.scheme(),
    )?;

    let available = AvailableVectorSizes::from_type_sizes(
        &client,
        lhs_handle.data_elem_size(),
        rhs_handle.data_elem_size(),
        dtypes.acc_global.size(),
    );
    let mut vector_sizes = available
        .filter_lhs_with_tensor(&problem.lhs_strides, &problem.lhs_shape, problem.lhs_layout)
        .filter_rhs_with_tensor(&problem.rhs_strides, &problem.rhs_shape, problem.rhs_layout)
        .filter_out_with_tensor(&problem.out_strides, &problem.out_shape)
        .pick_max()?;
    if lhs_handle.scale().is_some() {
        vector_sizes.lhs = 1;
    }
    if rhs_handle.scale().is_some() {
        vector_sizes.rhs = 1;
    }

    let mut view_vector_sizes = vector_sizes;
    if let InputBinding::Quantized { scheme, .. } = &lhs_handle {
        view_vector_sizes.lhs *= scheme.num_quants();
    }
    if let InputBinding::Quantized { scheme, .. } = &rhs_handle {
        view_vector_sizes.rhs *= scheme.num_quants();
    }
    let mut tile_problem = problem.clone();
    tile_problem.m = 1;
    let device_settings =
        <SimpleAlgorithm as Routine<()>>::device_settings(&client, view_vector_sizes);
    let expand = <SimpleAlgorithm as Routine<()>>::expand_blueprint(
        &tile_problem,
        &device_settings,
        &BlueprintStrategy::Inferred(SimpleArgs {
            multi_rows: false,
            tile_matmul: TileMatmulKind::Cmma,
        }),
    )?;
    launch_kernel_concrete::<TensorArgs, R, SimpleAlgorithm>(
        &client,
        lhs_handle,
        rhs_handle,
        out_binding,
        problem,
        vector_sizes,
        &BlueprintStrategy::Forced(expand.blueprint),
        &mut dtypes,
    )
}

fn quantized_binding<R: CubeRuntime>(tensor: &CubeTensor<R>) -> InputBinding<R> {
    match tensor.quantized_handles() {
        None => InputBinding::new(tensor.clone().binding(), tensor.dtype.into()),
        Some((data, scale)) => {
            let data_dtype = data.dtype;
            let scale_dtype = scale.dtype;
            InputBinding::quantized(
                data.binding(),
                scale.binding(),
                tensor.meta.shape().clone(),
                *tensor.scheme(),
                data_dtype.into(),
                scale_dtype.into(),
            )
        }
    }
}
