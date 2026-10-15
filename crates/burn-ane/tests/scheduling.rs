use burn_ane::{Ane, AneDevice, AneOptions, FallbackPolicy};
use burn_tensor::{Tensor, backend::Backend};

fn execute(device: &AneDevice) -> Vec<f32> {
    (Tensor::<Ane, 1>::from_floats([1.0, 2.0, 3.0], device) + 2.0)
        .into_data()
        .to_vec()
        .unwrap()
}

#[test]
fn fallback_scopes_are_thread_and_device_local_and_restore_after_unwind() {
    let first = AneDevice::default();
    let second = AneDevice::default();
    let scope = first.fallback_scope().unwrap();
    assert_eq!(execute(&first), [3., 4., 5.]);
    let initial = first.stats().fallback_scope_operations;
    assert!(initial > 0);
    std::thread::scope(|threads| {
        threads
            .spawn(|| assert_eq!(execute(&first), [3., 4., 5.]))
            .join()
            .unwrap();
    });
    assert_eq!(first.stats().fallback_scope_operations, initial);
    execute(&second);
    assert_eq!(second.stats().fallback_scope_operations, 0);
    let inner = first.fallback_scope().unwrap();
    execute(&first);
    drop(inner);
    execute(&first);
    assert_eq!(first.stats().fallback_scope_operations, initial + 2);
    drop(scope);
    execute(&first);
    assert_eq!(first.stats().fallback_scope_operations, initial + 2);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _scope = first.fallback_scope().unwrap();
        panic!("test unwind");
    }));
    assert!(result.is_err());
    execute(&first);
    assert_eq!(first.stats().fallback_scope_operations, initial + 2);
}

#[test]
fn custom_fallback_rejects_cross_session_and_strict_calls_without_consuming_live_inputs() {
    let first = AneDevice::default();
    let second = AneDevice::default();
    let x = Tensor::<Ane, 1>::from_floats([1.0, 2.0], &first);
    let y = Tensor::<Ane, 1>::from_floats([3.0, 4.0], &second);
    let xp = x.clone().into_primitive().tensor();
    let yp = y.clone().into_primitive().tensor();
    assert!(
        xp.client
            .clone()
            .fallback_float("wrong session", [xp, yp], |_| panic!("must not run"))
            .is_err()
    );
    assert_eq!(x.into_data().to_vec::<f32>().unwrap(), [1., 2.]);
    assert_eq!(y.into_data().to_vec::<f32>().unwrap(), [3., 4.]);
    let integers = Tensor::<Ane, 1, burn_tensor::Int>::from_ints([1, 2], &first);
    let primitive = integers.clone().into_primitive();
    assert!(
        primitive
            .client
            .clone()
            .fallback_float("wrong dtype", [primitive], |_| panic!("must not run"))
            .is_err()
    );
    assert_eq!(integers.into_data().to_vec::<i64>().unwrap(), [1, 2]);
    let strict = AneDevice::new(
        Default::default(),
        AneOptions {
            fallback: FallbackPolicy::Error,
            ..Default::default()
        },
    );
    assert!(strict.fallback_scope().is_err());
    let x = Tensor::<Ane, 1>::from_floats([1.0, 2.0], &strict);
    let xp = x.clone().into_primitive().tensor();
    assert!(
        xp.client
            .clone()
            .fallback_float("strict", [xp], |_| panic!("must not run"))
            .is_err()
    );
    assert_eq!(x.into_data().to_vec::<f32>().unwrap(), [1., 2.]);
    Ane::sync(&strict).unwrap();
}
