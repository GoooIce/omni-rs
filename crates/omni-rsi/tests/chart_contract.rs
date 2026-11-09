use omni_rsi::chart::{ChartProgram, ChartSpec, Split, render, verify};

#[test]
fn only_matching_programs_with_matching_pixels_become_training_data() {
    let spec = ChartSpec::new(3, 7).unwrap();
    let program = ChartProgram::parse(r#"{"left":3,"right":7}"#).unwrap();
    let image = render(&program).unwrap();
    assert!(verify(&spec, &program, &image).is_ok());
    let wrong = ChartProgram::parse(r#"{"left":7,"right":3}"#).unwrap();
    assert!(verify(&spec, &wrong, &image).is_err());
    let mut damaged = image.clone();
    damaged.put_pixel(55, 150, image::Rgb([0, 0, 0]));
    assert!(verify(&spec, &program, &damaged).is_err());
    assert!(ChartProgram::parse(r#"{"left":3,"right":7,"execute":"rm -rf /"}"#).is_err());
}

#[test]
fn mirrored_counterfactuals_never_cross_the_holdout_boundary() {
    for left in 1..=9 {
        for right in 1..=9 {
            if left != right {
                let a = ChartSpec::new(left, right).unwrap();
                let b = ChartSpec::new(right, left).unwrap();
                assert_eq!(a.split(), b.split());
                assert_ne!(a.label(), b.label());
            }
        }
    }
    assert!(ChartSpec::new(2, 2).is_err());
    assert!(matches!(
        ChartSpec::new(1, 2).unwrap().split(),
        Split::Train | Split::Dev | Split::Test
    ));
}
