use omni_lfm3::stable::ConfirmedText;

#[test]
fn confirmed_text_only_appends_after_two_consistent_hypotheses() {
    let mut text = ConfirmedText::default();
    assert_eq!(text.observe("之前。").unwrap(), "");
    assert_eq!(text.observe("之前有顾客。").unwrap(), "之前");
    assert_eq!(text.text(), "之前");
    assert_eq!(
        text.observe("之前，有顾客自己带酒水。").unwrap(),
        "，有顾客"
    );
    assert_eq!(text.text(), "之前，有顾客");
    assert_eq!(
        text.finish("之前，有顾客自己带酒水。").unwrap(),
        Some("自己带酒水。".into())
    );
    assert_eq!(text.text(), "之前，有顾客自己带酒水。");
    assert!(text.observe("之后").is_err());
}

#[test]
fn a_final_disagreement_keeps_committed_text_and_allows_a_forced_suffix() {
    let mut text = ConfirmedText::default();
    text.observe("天气不错。").unwrap();
    text.observe("天气不错，去公园。").unwrap();
    assert_eq!(text.text(), "天气不错");
    assert_eq!(text.finish("天气糟糕，去公园。").unwrap(), None);
    assert_eq!(text.text(), "天气不错");
    assert_eq!(text.finish_with_suffix("，去公园。").unwrap(), "，去公园。");
    assert_eq!(text.text(), "天气不错，去公园。");
    assert!(text.finish("天气糟糕").is_err());
}

#[test]
fn numerical_punctuation_is_not_ignored_when_confirming_text() {
    let mut text = ConfirmedText::default();
    text.observe("3.1。�").unwrap();
    assert_eq!(text.observe("31度。").unwrap(), "3");
    assert_eq!(text.text(), "3");
}
