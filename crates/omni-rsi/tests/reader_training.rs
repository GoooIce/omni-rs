use omni_rsi::reader::{Example, ReaderHead};

#[test]
fn real_gradient_updates_learn_a_separable_reader_without_changing_the_base_score() {
    let examples = vec![
        Example {
            features: vec![1.0, 0.0],
            base_logit: -1.0,
            label: true,
        },
        Example {
            features: vec![0.8, 0.2],
            base_logit: -1.0,
            label: true,
        },
        Example {
            features: vec![-1.0, 0.0],
            base_logit: 1.0,
            label: false,
        },
        Example {
            features: vec![-0.8, -0.2],
            base_logit: 1.0,
            label: false,
        },
    ];
    let mut head = ReaderHead::new(2).unwrap();
    assert!(head.probability(&examples[0]).unwrap() < 0.5);
    for _ in 0..200 {
        head.train_epoch(&examples, 0.1, 0.0).unwrap();
    }
    for example in &examples {
        assert_eq!(head.probability(example).unwrap() >= 0.5, example.label);
    }
    assert!(head.weights.iter().any(|weight| *weight != 0.0));
    assert_eq!(examples[0].base_logit, -1.0);
}

#[test]
fn malformed_or_nonfinite_examples_cannot_update_a_reader() {
    let mut head = ReaderHead::new(2).unwrap();
    let bad = Example {
        features: vec![f32::NAN, 0.0],
        base_logit: 0.0,
        label: true,
    };
    assert!(head.train_epoch(&[bad], 0.1, 0.0).is_err());
    assert_eq!(head.weights, vec![0.0, 0.0]);
    assert!(
        head.probability(&Example {
            features: vec![1.0],
            base_logit: 0.0,
            label: false
        })
        .is_err()
    );
}
