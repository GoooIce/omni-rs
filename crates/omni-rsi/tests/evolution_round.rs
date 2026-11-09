use image::RgbImage;
use omni_rsi::{
    chart::{ChartProgram, ChartSpec, Question},
    evolution::{Oracle, RoundOptions, run_round},
    reader::Example,
    storage::Store,
};
use std::{
    cell::Cell,
    sync::atomic::{AtomicBool, Ordering},
};

struct PixelOracle {
    wrong_program: bool,
}

struct InterruptingOracle(Cell<usize>);
impl Oracle for InterruptingOracle {
    fn write_program(
        &self,
        spec: ChartSpec,
        stop: &AtomicBool,
    ) -> Result<Option<ChartProgram>, String> {
        PixelOracle {
            wrong_program: false,
        }
        .write_program(spec, stop)
    }
    fn features(
        &self,
        image: &RgbImage,
        question: Question,
        label: bool,
        stop: &AtomicBool,
    ) -> Result<Example, String> {
        let result = PixelOracle {
            wrong_program: false,
        }
        .features(image, question, label, stop);
        self.0.set(self.0.get() + 1);
        if self.0.get() == 2 {
            stop.store(true, Ordering::Release);
        }
        result
    }
}
impl Oracle for PixelOracle {
    fn write_program(
        &self,
        spec: ChartSpec,
        _: &AtomicBool,
    ) -> Result<Option<ChartProgram>, String> {
        Ok(Some(if self.wrong_program {
            ChartProgram {
                left: spec.right,
                right: spec.left,
            }
        } else {
            ChartProgram {
                left: spec.left,
                right: spec.right,
            }
        }))
    }
    fn features(
        &self,
        image: &RgbImage,
        question: Question,
        label: bool,
        _: &AtomicBool,
    ) -> Result<Example, String> {
        let count = |start: u32, end: u32| -> f32 {
            (start..end)
                .flat_map(|x| (0..image.height()).map(move |y| (x, y)))
                .filter(|(x, y)| image.get_pixel(*x, *y).0 != [255, 255, 255])
                .count() as f32
        };
        let (left, right) = (count(40, 70), count(122, 152));
        let values = match question {
            Question::Taller => vec![(left - right) / 3000.0, 0.0],
            Question::Twice => vec![0.0, (left - 2.0 * right + 200.0) / 3000.0],
        };
        Ok(Example {
            features: values,
            base_logit: 0.0,
            label,
        })
    }
}

#[test]
fn verified_counterfactuals_train_and_promote_a_real_reader_update() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::new(dir.path());
    let options = RoundOptions {
        families: 4,
        epochs: 200,
        learning_rate: 0.2,
        seed: 31,
    };
    let report = run_round(
        &PixelOracle {
            wrong_program: false,
        },
        &store,
        "fixture-model",
        &options,
        &AtomicBool::new(false),
    )
    .unwrap();
    assert!(report.promoted && report.promotion_finalized && report.train_examples >= 8);
    assert!(report.dev.combined.correct() > report.baseline.combined.correct());
    let active = store.load_active("fixture-model").unwrap().unwrap();
    assert!(active.head.updates > 0 && active.head.weights.iter().any(|x| *x != 0.0));
    // A resume must not undo an explicit rollback by re-promoting the last checkpoint.
    let mut prior = active.clone();
    prior.round = 0;
    prior.dev = prior.baseline.clone();
    let prior_id = store.stage(&prior).unwrap();
    std::fs::write(
        dir.path().join("previous.json"),
        serde_json::to_vec(&prior_id).unwrap(),
    )
    .unwrap();
    assert_eq!(store.rollback("fixture-model").unwrap(), prior_id);
    let progress = std::fs::read(dir.path().join("progress.json")).unwrap();
    let interrupted = run_round(
        &InterruptingOracle(Cell::new(0)),
        &store,
        "fixture-model",
        &options,
        &AtomicBool::new(false),
    );
    assert!(interrupted.unwrap_err().contains("interrupted"));
    assert_eq!(
        std::fs::read(dir.path().join("progress.json")).unwrap(),
        progress
    );
    assert_eq!(
        store.load_active("fixture-model").unwrap().unwrap().round,
        prior.round
    );
    assert_eq!(store.rollback("fixture-model").unwrap(), report.candidate);
    let result = run_round(
        &PixelOracle {
            wrong_program: true,
        },
        &store,
        "fixture-model",
        &options,
        &AtomicBool::new(false),
    );
    assert!(
        result.is_err(),
        "unverified programs must never train a candidate"
    );
    assert_eq!(
        store.load_active("fixture-model").unwrap().unwrap().round,
        active.round
    );

    let resumed = run_round(
        &PixelOracle {
            wrong_program: false,
        },
        &store,
        "fixture-model",
        &options,
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_eq!(resumed.round, 2);
    assert!(
        resumed.train_examples >= report.train_examples,
        "verified replay is retained"
    );

    let progress_path = dir.path().join("progress.json");
    let mut progress: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&progress_path).unwrap()).unwrap();
    progress["replay"][0]["example"]["base_logit"] = 123.into();
    std::fs::write(&progress_path, serde_json::to_vec(&progress).unwrap()).unwrap();
    let error = run_round(
        &PixelOracle {
            wrong_program: false,
        },
        &store,
        "fixture-model",
        &options,
        &AtomicBool::new(false),
    )
    .unwrap_err();
    assert!(error.contains("checkpoint digest mismatch"), "{error}");
}
