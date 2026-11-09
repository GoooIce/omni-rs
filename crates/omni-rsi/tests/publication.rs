use omni_rsi::{
    reader::ReaderHead,
    storage::{Evaluation, Metrics, Package, Store, digest},
};

fn metrics(yes_correct: usize, no_correct: usize) -> Metrics {
    Metrics {
        yes_total: 3,
        no_total: 3,
        yes_correct,
        no_correct,
    }
}

fn evaluation(first: Metrics, second: Metrics) -> Evaluation {
    Evaluation {
        combined: Metrics {
            yes_total: first.yes_total + second.yes_total,
            no_total: first.no_total + second.no_total,
            yes_correct: first.yes_correct + second.yes_correct,
            no_correct: first.no_correct + second.no_correct,
        },
        by_question: [first, second],
    }
}

#[test]
fn a_candidate_must_improve_dev_before_atomic_promotion_and_must_reload_intact() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::new(dir.path());
    let mut head = ReaderHead::new(2).unwrap();
    head.weights = vec![0.2, -0.1];
    let package = Package::new(
        "base-hash".into(),
        1,
        head,
        evaluation(metrics(2, 2), metrics(2, 2)),
        evaluation(metrics(1, 1), metrics(1, 1)),
        digest(b"fixture checkpoint"),
    )
    .unwrap();
    let id = store.stage(&package).unwrap();
    assert!(store.promote(&id, "base-hash").unwrap());
    assert_eq!(
        store
            .load_active("base-hash")
            .unwrap()
            .unwrap()
            .head
            .weights,
        vec![0.2, -0.1]
    );
    assert!(store.load_active("other-model").is_err());

    let mut worse = package.clone();
    worse.round = 2;
    worse.dev = evaluation(metrics(1, 1), metrics(1, 2));
    let rejected = store.stage(&worse).unwrap();
    assert!(!store.promote(&rejected, "base-hash").unwrap());
    assert_eq!(store.load_active("base-hash").unwrap().unwrap().round, 1);

    let mut task_regression = package.clone();
    task_regression.round = 3;
    task_regression.dev = evaluation(metrics(1, 2), metrics(3, 3));
    let rejected = store.stage(&task_regression).unwrap();
    assert!(
        !store.promote(&rejected, "base-hash").unwrap(),
        "an overall gain must not hide a task regression"
    );
    assert_eq!(store.load_active("base-hash").unwrap().unwrap().round, 1);

    let mut improved = package.clone();
    improved.round = 4;
    improved.dev = evaluation(metrics(3, 3), metrics(3, 3));
    let improved_id = store.stage(&improved).unwrap();
    assert!(store.promote(&improved_id, "base-hash").unwrap());
    assert_eq!(store.load_active("base-hash").unwrap().unwrap().round, 4);
    assert!(store.rollback("other-model").is_err());
    assert_eq!(store.rollback("base-hash").unwrap(), id);
    assert_eq!(store.load_active("base-hash").unwrap().unwrap().round, 1);
    let previous: String =
        serde_json::from_slice(&std::fs::read(dir.path().join("previous.json")).unwrap()).unwrap();
    assert_eq!(previous, improved_id);

    std::fs::write(
        dir.path().join("candidates").join(format!("{id}.json")),
        b"{}\n",
    )
    .unwrap();
    assert!(store.load_active("base-hash").is_err());
}
