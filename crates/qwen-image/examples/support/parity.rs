//! Numerical checks shared by the attention benchmark and its CPU-only tests.

#[derive(Debug)]
pub struct Error {
    pub relative_l2: f64,
    pub max_abs: f64,
}

impl Error {
    pub fn passes(&self, relative_limit: f64, absolute_limit: f64) -> bool {
        self.relative_l2 < relative_limit && self.max_abs < absolute_limit
    }
}

pub fn compare(reference: &[f64], candidate: &[f64]) -> Result<Error, &'static str> {
    if reference.is_empty() || reference.len() != candidate.len() {
        return Err("nonempty equal-length tensors required");
    }
    let (mut squared_error, mut squared_norm, mut max_abs) = (0.0, 0.0, 0.0_f64);
    for (&expected, &actual) in reference.iter().zip(candidate) {
        if !expected.is_finite() || !actual.is_finite() {
            return Err("nonfinite tensor value");
        }
        let error = actual - expected;
        squared_error += error * error;
        squared_norm += expected * expected;
        max_abs = max_abs.max(error.abs());
    }
    let relative_l2 = (squared_error / squared_norm.max(1e-30)).sqrt();
    if !relative_l2.is_finite() || !max_abs.is_finite() {
        return Err("nonfinite error metric");
    }
    Ok(Error {
        relative_l2,
        max_abs,
    })
}

#[cfg(test)]
mod tests {
    use super::compare;

    #[test]
    fn accepts_rounding_error_but_rejects_a_perturbed_candidate() {
        let reference = [0.25, -0.5, 1.0, -2.0];
        let mut candidate = reference;
        candidate[1] += 0.0001;
        assert!(compare(&reference, &candidate).unwrap().passes(0.003, 0.04));
        candidate[1] += 0.2;
        assert!(!compare(&reference, &candidate).unwrap().passes(0.003, 0.04));
        assert_eq!(reference, [0.25, -0.5, 1.0, -2.0]);
    }

    #[test]
    fn rejects_nonfinite_truncated_and_empty_results() {
        assert!(compare(&[], &[]).is_err());
        assert!(compare(&[1.0, 2.0], &[1.0]).is_err());
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(compare(&[1.0], &[invalid]).is_err());
            assert!(compare(&[invalid], &[1.0]).is_err());
        }
        assert!(!compare(&[0.0], &[1.0]).unwrap().passes(0.003, 0.04));
    }
}
