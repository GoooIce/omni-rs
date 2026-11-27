#![cfg(all(feature = "native", any(target_os = "linux", target_os = "macos")))]

use std::{fs, process::Command};

fn single_interval(tier: &str, mark: &str) -> String {
    format!(
        "File type = \"ooTextFile\"\nObject class = \"TextGrid\"\nxmin = 0\nxmax = 1\n\
         tiers? <exists>\nsize = 1\nitem []:\nitem [1]:\nclass = \"IntervalTier\"\n\
         name = \"{tier}\"\nxmin = 0\nxmax = 1\nintervals: size = 1\nintervals [1]:\n\
         xmin = 0\nxmax = 1\ntext = \"{mark}\"\n"
    )
}

fn run_cli(input: &std::path::Path, output: &std::path::Path) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_alimeeting-screen"))
        .arg("--textgrid-dir")
        .arg(input)
        .arg("--output")
        .arg(output)
        .output()
        .unwrap()
}

fn utf16(text: &str, little_endian: bool) -> Vec<u8> {
    let mut bytes = if little_endian {
        vec![0xff, 0xfe]
    } else {
        vec![0xfe, 0xff]
    };
    for word in text.encode_utf16() {
        bytes.extend_from_slice(&if little_endian {
            word.to_le_bytes()
        } else {
            word.to_be_bytes()
        });
    }
    bytes
}

#[test]
fn cli_writes_reviewable_exploratory_candidates_from_a_long_textgrid() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let input = root.join("textgrid_dir");
    fs::create_dir(&input).unwrap();
    fs::write(
        input.join("R0001_M0001_F_SPK0001.TextGrid"),
        r#"File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 2
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "SPK0001"
        xmin = 0
        xmax = 2
        intervals: size = 3
        intervals [1]:
            xmin = 0
            xmax = 0.5
            text = ""
        intervals [2]:
            xmin = 0.5
            xmax = 1.5
            text = "十二"
        intervals [3]:
            xmin = 1.5
            xmax = 2
            text = ""
"#,
    )
    .unwrap();
    let output_path = root.join("candidates.json");
    let binary = env!("CARGO_BIN_EXE_alimeeting-screen");
    let result = Command::new(binary)
        .arg("--textgrid-dir")
        .arg(&input)
        .arg("--output")
        .arg(&output_path)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(output_path).unwrap()).unwrap();
    assert_eq!(report["kind"], "exploratory_text_candidates");
    assert_eq!(report["summary"]["candidate_count"], 1);
    assert_eq!(report["summary"]["distinct_tier_ids"], 1);
    let candidate = &report["candidates"][0];
    assert_eq!(candidate["recording_id"], "R0001_M0001_F_SPK0001");
    assert_eq!(candidate["tier_name"], "SPK0001");
    assert_eq!(candidate["start_seconds"], 0.5);
    assert_eq!(candidate["end_seconds"], 1.5);
    assert_eq!(candidate["mark"], "十二");
    assert_eq!(candidate["candidate_number"], 12);
}

#[test]
fn multiline_mark_cannot_bypass_the_string_limit() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let input = root.join("textgrid_dir");
    fs::create_dir(&input).unwrap();
    let excessive = "\n".repeat(65_537);
    let grid = format!(
        "File type = \"ooTextFile\"\nObject class = \"TextGrid\"\nxmin = 0\nxmax = 1\n\
         tiers? <exists>\nsize = 1\nitem []:\nitem [1]:\nclass = \"IntervalTier\"\n\
         name = \"SPK0001\"\nxmin = 0\nxmax = 1\nintervals: size = 1\nintervals [1]:\n\
         xmin = 0\nxmax = 1\ntext = \"{excessive}\"\n"
    );
    fs::write(input.join("R0001_SPK0001.TextGrid"), grid).unwrap();
    let output_path = root.join("candidates.json");
    let result = Command::new(env!("CARGO_BIN_EXE_alimeeting-screen"))
        .arg("--textgrid-dir")
        .arg(&input)
        .arg("--output")
        .arg(&output_path)
        .output()
        .unwrap();
    assert!(
        !result.status.success(),
        "oversized multiline mark must be rejected"
    );
    assert!(!output_path.exists());
}

#[test]
fn implausibly_long_times_fail_without_publishing_a_report() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let input = root.join("textgrid_dir");
    fs::create_dir(&input).unwrap();
    let grid = "File type = \"ooTextFile\"\nObject class = \"TextGrid\"\nxmin = 0\nxmax = 86401\n\
        tiers? <exists>\nsize = 1\nitem []:\nitem [1]:\nclass = \"IntervalTier\"\n\
        name = \"SPK0001\"\nxmin = 0\nxmax = 86401\nintervals: size = 1\nintervals [1]:\n\
        xmin = 0\nxmax = 86401\ntext = \"十二\"\n";
    fs::write(input.join("R0001_SPK0001.TextGrid"), grid).unwrap();
    let output_path = root.join("candidates.json");
    let result = Command::new(env!("CARGO_BIN_EXE_alimeeting-screen"))
        .arg("--textgrid-dir")
        .arg(&input)
        .arg("--output")
        .arg(&output_path)
        .output()
        .unwrap();
    assert!(
        !result.status.success(),
        "times beyond the 24-hour limit must fail"
    );
    assert!(!output_path.exists());
}

#[cfg(unix)]
#[test]
fn rejects_output_directory_writable_by_other_users() {
    use std::os::unix::fs::PermissionsExt;

    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let input = root.join("textgrid_dir");
    let output_dir = root.join("shared");
    fs::create_dir(&input).unwrap();
    fs::create_dir(&output_dir).unwrap();
    fs::set_permissions(&output_dir, fs::Permissions::from_mode(0o777)).unwrap();
    let grid = "File type = \"ooTextFile\"\nObject class = \"TextGrid\"\nxmin = 0\nxmax = 1\n\
        tiers? <exists>\nsize = 1\nitem []:\nitem [1]:\nclass = \"IntervalTier\"\n\
        name = \"SPK0001\"\nxmin = 0\nxmax = 1\nintervals: size = 1\nintervals [1]:\n\
        xmin = 0\nxmax = 1\ntext = \"十二\"\n";
    fs::write(input.join("R0001_SPK0001.TextGrid"), grid).unwrap();
    let output = output_dir.join("candidates.json");
    let result = Command::new(env!("CARGO_BIN_EXE_alimeeting-screen"))
        .arg("--textgrid-dir")
        .arg(&input)
        .arg("--output")
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        !result.status.success(),
        "untrusted output directory must be rejected"
    );
    assert!(!output.exists());
}

#[test]
fn records_source_and_tool_identities_for_reproducible_candidates() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let input = root.join("textgrid_dir");
    fs::create_dir(&input).unwrap();
    let source = input.join("R0001_SPK0001.TextGrid");
    fs::write(&source, single_interval("SPK0001", "十二")).unwrap();
    let first_path = root.join("first.json");
    assert!(run_cli(&input, &first_path).status.success());
    let first: serde_json::Value = serde_json::from_slice(&fs::read(first_path).unwrap()).unwrap();
    fs::write(&source, single_interval("SPK0001", "十三")).unwrap();
    let second_path = root.join("second.json");
    assert!(run_cli(&input, &second_path).status.success());
    let second: serde_json::Value =
        serde_json::from_slice(&fs::read(second_path).unwrap()).unwrap();
    assert_ne!(first["source_sha256"], second["source_sha256"]);
    let tool = first["tool_sha256"]
        .as_str()
        .expect("tool identity is required");
    assert_eq!(tool.len(), 64);
    assert!(tool.bytes().all(|byte| byte.is_ascii_hexdigit()));
    assert_eq!(first["tool_sha256"], second["tool_sha256"]);
}

#[test]
fn duplicate_recording_stems_are_rejected_as_ambiguous() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let input = root.join("textgrid_dir");
    fs::create_dir_all(input.join("first")).unwrap();
    fs::create_dir_all(input.join("second")).unwrap();
    let name = "R0001_SPK0001.TextGrid";
    fs::write(
        input.join("first").join(name),
        single_interval("SPK0001", "十二"),
    )
    .unwrap();
    fs::write(
        input.join("second").join(name),
        single_interval("SPK0001", "十三"),
    )
    .unwrap();
    let output = root.join("candidates.json");
    let result = run_cli(&input, &output);
    assert!(
        !result.status.success(),
        "a recording stem must identify one TextGrid"
    );
    assert!(!output.exists());
}

#[test]
fn whole_number_rule_excludes_dates_digit_strings_and_context() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let input = root.join("textgrid_dir");
    fs::create_dir(&input).unwrap();
    let cases = [
        ("零", Some(0)),
        ("〇", Some(0)),
        ("两", Some(2)),
        ("十", Some(10)),
        ("一十", Some(10)),
        ("十一", Some(11)),
        ("一十九", Some(19)),
        ("二十", Some(20)),
        ("九十九", Some(99)),
        ("99", Some(99)),
        ("０", Some(0)),
        ("１２", Some(12)),
        ("01", None),
        ("二零", None),
        ("一二", None),
        ("一个", None),
        ("十二月", None),
        ("100", None),
        ("一百", None),
        ("答案是十二", None),
        ("1.2", None),
    ];
    for (index, (mark, _)) in cases.iter().enumerate() {
        fs::write(
            input.join(format!("R{index:04}_SPK0001.TextGrid")),
            single_interval("SPK0001", mark),
        )
        .unwrap();
    }
    let output = root.join("candidates.json");
    let result = run_cli(&input, &output);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&fs::read(output).unwrap()).unwrap();
    let actual: Vec<_> = report["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            (
                row["mark"].as_str().unwrap().to_owned(),
                row["candidate_number"].as_u64().unwrap(),
            )
        })
        .collect();
    let expected: Vec<_> = cases
        .iter()
        .filter_map(|(mark, value)| value.map(|number| (mark.to_string(), number)))
        .collect();
    assert_eq!(actual, expected);
    assert_eq!(report["summary"]["distinct_tier_ids"], 1);
}

#[test]
fn leading_period_before_chinese_number_is_deferred() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let input = root.join("textgrid_dir");
    fs::create_dir(&input).unwrap();
    for (index, mark) in [".一", "．一", "...十", "．两", ".〇"].iter().enumerate() {
        fs::write(
            input.join(format!("R{index:04}.TextGrid")),
            single_interval("c1", mark),
        )
        .unwrap();
    }
    let output = root.join("candidates.json");
    let result = run_cli(&input, &output);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&fs::read(output).unwrap()).unwrap();
    assert_eq!(report["summary"]["candidate_count"], 0);
    assert_eq!(report["summary"]["deferred_ambiguous_notation_count"], 5);
    let deferred = report["deferred"].as_array().unwrap();
    assert_eq!(deferred.len(), 5);
    for row in deferred {
        assert_eq!(row["reason"], "ambiguous_numeric_notation");
        assert!(row.get("candidate_number").is_none());
    }
}

#[test]
fn mixed_leading_punctuation_with_a_period_is_deferred() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let input = root.join("textgrid_dir");
    fs::create_dir(&input).unwrap();
    let ambiguous = [
        ".,一",
        "．，一",
        ".,1",
        "，．１",
        ",.一",
        "，.1",
        ".，十",
        ". , \t一",
        "！．、１２",
        "，．，两",
    ];
    let unambiguous = [
        (",一", 1),
        ("，１", 1),
        ("。一", 1),
        ("一.", 1),
        ("十二。", 12),
        (" １２， ", 12),
    ];
    let rejected = [".", "。", "1.2", "．，一二", "．，01", "正文.,一"];
    for (index, mark) in ambiguous
        .iter()
        .copied()
        .chain(unambiguous.iter().map(|(mark, _)| *mark))
        .chain(rejected)
        .enumerate()
    {
        fs::write(
            input.join(format!("R{index:04}.TextGrid")),
            single_interval("c1", mark),
        )
        .unwrap();
    }
    let output = root.join("candidates.json");
    let result = run_cli(&input, &output);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&fs::read(output).unwrap()).unwrap();
    assert_eq!(report["summary"]["candidate_count"], unambiguous.len());
    assert_eq!(
        report["summary"]["deferred_ambiguous_notation_count"],
        ambiguous.len()
    );
    let candidates = report["candidates"].as_array().unwrap();
    assert_eq!(candidates.len(), unambiguous.len());
    for (row, (mark, number)) in candidates.iter().zip(unambiguous) {
        assert_eq!(row["mark"], mark);
        assert_eq!(row["candidate_number"], number);
    }
    let deferred = report["deferred"].as_array().unwrap();
    assert_eq!(deferred.len(), ambiguous.len());
    for (row, mark) in deferred.iter().zip(ambiguous) {
        assert_eq!(row["mark"], mark);
        assert_eq!(row["reason"], "ambiguous_numeric_notation");
        assert_eq!(
            row["reasons"],
            serde_json::json!(["ambiguous_numeric_notation"])
        );
        assert!(row.get("candidate_number").is_none());
    }
}

#[test]
fn compatibility_digits_and_leading_decimal_are_deferred_for_review() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let input = root.join("textgrid_dir");
    fs::create_dir(&input).unwrap();
    for (index, mark) in ["①③", "1²", ".1", "...1"].iter().enumerate() {
        fs::write(
            input.join(format!("R{index:04}.TextGrid")),
            single_interval("SPK0001", mark),
        )
        .unwrap();
    }
    let output = root.join("candidates.json");
    let result = run_cli(&input, &output);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&fs::read(output).unwrap()).unwrap();
    assert_eq!(report["summary"]["candidate_count"], 0);
    assert_eq!(report["summary"]["deferred_ambiguous_notation_count"], 4);
    for row in report["deferred"].as_array().unwrap() {
        assert_eq!(row["reason"], "ambiguous_numeric_notation");
        assert!(row.get("candidate_number").is_none());
    }
}

#[test]
fn ambiguous_notation_and_neighbor_context_are_both_reported() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let input = root.join("textgrid_dir");
    fs::create_dir(&input).unwrap();
    let marks = [".一", ".,一", "．，１"];
    for (index, mark) in marks.iter().enumerate() {
        let grid = format!(
            "File type = \"ooTextFile\"\nObject class = \"TextGrid\"\nxmin = 0\nxmax = 2\n\
             tiers? <exists>\nsize = 1\nitem []:\nitem [1]:\nclass = \"IntervalTier\"\n\
             name = \"c1\"\nxmin = 0\nxmax = 2\nintervals: size = 2\n\
             intervals [1]:\nxmin = 0\nxmax = 1\ntext = \"{mark}\"\n\
             intervals [2]:\nxmin = 1\nxmax = 2\ntext = \"个\"\n"
        );
        fs::write(input.join(format!("R{index:04}.TextGrid")), grid).unwrap();
    }
    let output = root.join("candidates.json");
    let result = run_cli(&input, &output);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&fs::read(output).unwrap()).unwrap();
    assert_eq!(report["summary"]["candidate_count"], 0);
    assert_eq!(report["summary"]["deferred_adjacent_count"], marks.len());
    assert_eq!(
        report["summary"]["deferred_ambiguous_notation_count"],
        marks.len()
    );
    let deferred = report["deferred"].as_array().unwrap();
    assert_eq!(deferred.len(), marks.len());
    for (row, mark) in deferred.iter().zip(marks) {
        assert_eq!(row["mark"], mark);
        assert_eq!(row["reason"], "ambiguous_numeric_notation");
        assert_eq!(
            row["reasons"],
            serde_json::json!(["ambiguous_numeric_notation", "adjacent_nonempty_interval"])
        );
        assert!(row.get("candidate_number").is_none());
    }
}

#[test]
fn a_digit_adjacent_to_other_text_is_deferred_instead_of_labeled() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let input = root.join("textgrid_dir");
    fs::create_dir(&input).unwrap();
    let grid = "File type = \"ooTextFile\"\nObject class = \"TextGrid\"\nxmin = 0\nxmax = 4\n\
        tiers? <exists>\nsize = 1\nitem []:\nitem [1]:\nclass = \"IntervalTier\"\n\
        name = \"SPK0001\"\nxmin = 0\nxmax = 4\nintervals: size = 4\n\
        intervals [1]:\nxmin = 0\nxmax = 1\ntext = \"\"\n\
        intervals [2]:\nxmin = 1\nxmax = 2\ntext = \"一\"\n\
        intervals [3]:\nxmin = 2\nxmax = 3\ntext = \"个\"\n\
        intervals [4]:\nxmin = 3\nxmax = 4\ntext = \"\"\n";
    fs::write(input.join("R0001_SPK0001.TextGrid"), grid).unwrap();
    let output = root.join("candidates.json");
    let result = run_cli(&input, &output);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&fs::read(output).unwrap()).unwrap();
    assert_eq!(report["summary"]["candidate_count"], 0);
    assert_eq!(report["summary"]["deferred_adjacent_count"], 1);
    assert_eq!(report["deferred"][0]["mark"], "一");
    assert_eq!(
        report["deferred"][0]["reason"],
        "adjacent_nonempty_interval"
    );
    assert!(report["deferred"][0].get("candidate_number").is_none());
}

#[test]
fn multiline_crlf_mark_is_preserved_in_the_report() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let input = root.join("textgrid_dir");
    fs::create_dir(&input).unwrap();
    let grid = single_interval("c1", "\n十二\n").replace('\n', "\r\n");
    fs::write(input.join("R0001.TextGrid"), grid.as_bytes()).unwrap();
    fs::write(input.join("R0002.TextGrid"), utf16(&grid, true)).unwrap();
    fs::write(input.join("R0003.TextGrid"), utf16(&grid, false)).unwrap();
    let output = root.join("candidates.json");
    let result = run_cli(&input, &output);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&fs::read(output).unwrap()).unwrap();
    let candidates = report["candidates"].as_array().unwrap();
    assert_eq!(candidates.len(), 3);
    for candidate in candidates {
        assert_eq!(candidate["mark"], "\r\n十二\r\n");
        assert_eq!(candidate["candidate_number"], 12);
    }
}

#[test]
fn supports_utf8_utf16_quotes_multiline_and_empty_point_tiers() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let input = root.join("textgrid_dir");
    fs::create_dir(&input).unwrap();
    let quoted_mark = single_interval("SPK0000", "他说 \"\"十二\"\"");
    let mut utf8 = vec![0xef, 0xbb, 0xbf];
    utf8.extend_from_slice(quoted_mark.as_bytes());
    fs::write(input.join("R0000.TextGrid"), utf8).unwrap();
    let multiline = single_interval("SPK\"\"0001", " \n十二\n");
    fs::write(input.join("R0001.TextGrid"), utf16(&multiline, true)).unwrap();
    let multiple_tiers = "File type = \"ooTextFile\"\nObject class = \"TextGrid\"\nxmin = 0\nxmax = 1\n\
        tiers? <exists>\nsize = 3\nitem []:\n\
        item [1]:\nclass = \"TextTier\"\nname = \"empty\"\nxmin = 0\nxmax = 1\npoints: size = 0\n\
        item [2]:\nclass = \"IntervalTier\"\nname = \"SPK0002\"\nxmin = 0\nxmax = 1\n\
        intervals: size = 1\nintervals [1]:\nxmin = 0\nxmax = 1\ntext = \"九十九\"\n\
        item [3]:\nclass = \"IntervalTier\"\nname = \"\"\nxmin = 0\nxmax = 1\nintervals: size = 0\n";
    fs::write(input.join("R0002.TextGrid"), utf16(multiple_tiers, false)).unwrap();
    let output = root.join("candidates.json");
    let result = run_cli(&input, &output);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&fs::read(output).unwrap()).unwrap();
    assert_eq!(report["summary"]["source_files"], 3);
    assert_eq!(report["summary"]["candidate_count"], 2);
    assert_eq!(report["summary"]["distinct_tier_ids"], 2);
    assert_eq!(report["candidates"][0]["tier_name"], "SPK\"0001");
    assert_eq!(report["candidates"][0]["mark"], " \n十二\n");
    assert_eq!(report["candidates"][0]["candidate_number"], 12);
    assert_eq!(report["candidates"][1]["candidate_number"], 99);
    assert_eq!(report["candidates"][1]["tier_index"], 2);
}

#[test]
fn keeps_multiple_tiers_separate_without_claiming_real_people() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let input = root.join("textgrid_dir");
    fs::create_dir(&input).unwrap();
    let grid = "File type = \"ooTextFile\"\nObject class = \"TextGrid\"\nxmin = 0\nxmax = 1\n\
        tiers? <exists>\nsize = 2\nitem []:\n\
        item [1]:\nclass = \"IntervalTier\"\nname = \"SPK0001\"\nxmin = 0\nxmax = 1\n\
        intervals: size = 1\nintervals [1]:\nxmin = 0\nxmax = 1\ntext = \"十二\"\n\
        item [2]:\nclass = \"IntervalTier\"\nname = \"SPK0002\"\nxmin = 0\nxmax = 1\n\
        intervals: size = 1\nintervals [1]:\nxmin = 0\nxmax = 1\ntext = \"十五\"\n";
    fs::write(input.join("R0001.TextGrid"), grid).unwrap();
    let output = root.join("candidates.json");
    let result = run_cli(&input, &output);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&fs::read(output).unwrap()).unwrap();
    assert_eq!(report["summary"]["distinct_tier_ids"], 2);
    assert_eq!(report["summary"]["tier_id_counts"]["SPK0001"], 1);
    assert_eq!(report["summary"]["tier_id_counts"]["SPK0002"], 1);
    assert_eq!(report["candidates"][0]["recording_id"], "R0001");
    assert_eq!(report["candidates"][1]["recording_id"], "R0001");
    assert!(
        report["limitations"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item
                .as_str()
                .unwrap()
                .contains("not verified unique people"))
    );
}

#[cfg(unix)]
#[test]
fn rejects_symlinks_special_files_traversal_and_oversized_inputs() {
    use std::{
        ffi::CString,
        os::unix::{ffi::OsStrExt, fs::symlink},
    };

    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let input = root.join("textgrid_dir");
    fs::create_dir(&input).unwrap();
    let source = input.join("R0001.TextGrid");
    fs::write(&source, single_interval("SPK0001", "十二")).unwrap();
    let output = root.join("candidates.json");

    let input_link = root.join("input-link");
    symlink(&input, &input_link).unwrap();
    assert!(!run_cli(&input_link, &output).status.success());
    assert!(
        !run_cli(&input.join("..").join("textgrid_dir"), &output)
            .status
            .success()
    );
    fs::remove_file(input_link).unwrap();

    let link = input.join("R0002.TextGrid");
    symlink(&source, &link).unwrap();
    assert!(!run_cli(&input, &output).status.success());
    fs::remove_file(link).unwrap();

    let link_dir = input.join("linked-dir");
    symlink(&root, &link_dir).unwrap();
    assert!(!run_cli(&input, &output).status.success());
    fs::remove_file(link_dir).unwrap();

    let fifo = input.join("pipe.TextGrid");
    let fifo_name = CString::new(fifo.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(fifo_name.as_ptr(), 0o600) }, 0);
    assert!(!run_cli(&input, &output).status.success());
    fs::remove_file(fifo).unwrap();

    let large = input.join("large.TextGrid");
    fs::File::create(&large)
        .unwrap()
        .set_len(8 * 1024 * 1024 + 1)
        .unwrap();
    assert!(!run_cli(&input, &output).status.success());
    assert!(!output.exists());
}

#[test]
fn repeated_long_source_paths_hit_the_accumulation_budget_before_publication() {
    use std::fmt::Write;

    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let input = root.join("textgrid_dir");
    fs::create_dir(&input).unwrap();
    let mut nested = input.clone();
    for index in 0..3 {
        nested.push(format!("d{index}{}", "x".repeat(238)));
        fs::create_dir(&nested).unwrap();
    }
    let count = 38_000;
    let mut grid = format!(
        "File type = \"ooTextFile\"\nObject class = \"TextGrid\"\nxmin = 0\nxmax = {count}\n\
         tiers? <exists>\nsize = 1\nitem []:\nitem [1]:\nclass = \"IntervalTier\"\n\
         name = \"c1\"\nxmin = 0\nxmax = {count}\nintervals: size = {count}\n"
    );
    for index in 0..count {
        writeln!(
            grid,
            "intervals [{}]:\nxmin = {index}\nxmax = {}\ntext = \"1\"",
            index + 1,
            index + 1
        )
        .unwrap();
    }
    assert!(grid.len() < 8 * 1024 * 1024);
    fs::write(nested.join("R0001.TextGrid"), grid).unwrap();
    let output = root.join("candidates.json");
    let result = run_cli(&input, &output);
    assert!(
        !result.status.success(),
        "accumulated records must be bounded"
    );
    assert!(
        String::from_utf8_lossy(&result.stderr).contains("accumulated report data exceeds limit"),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(!output.exists());
}

#[test]
fn input_depth_and_unexpected_regular_files_are_not_silently_skipped() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let input = root.join("textgrid_dir");
    fs::create_dir(&input).unwrap();
    fs::write(
        input.join("R0001.TextGrid"),
        single_interval("SPK0001", "十二"),
    )
    .unwrap();
    let output = root.join("candidates.json");
    let unexpected = input.join("unexpected.txt");
    fs::write(&unexpected, "synthetic").unwrap();
    assert!(!run_cli(&input, &output).status.success());
    fs::remove_file(unexpected).unwrap();
    let mut deep = input.clone();
    for _ in 0..17 {
        deep.push("nested");
        fs::create_dir(&deep).unwrap();
    }
    fs::write(
        deep.join("R0002.TextGrid"),
        single_interval("SPK0002", "十五"),
    )
    .unwrap();
    assert!(!run_cli(&input, &output).status.success());
    assert!(!output.exists());
}

#[test]
fn malformed_or_unsupported_textgrids_fail_closed() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let input = root.join("textgrid_dir");
    fs::create_dir(&input).unwrap();
    let source = input.join("R0001.TextGrid");
    let output = root.join("candidates.json");
    let valid = single_interval("SPK0001", "十二");
    let short = "File type = \"ooTextFile\"\nObject class = \"TextGrid\"\n0\n1\n<exists>\n1\n\"IntervalTier\"\n\"SPK0001\"\n0\n1\n1\n0\n1\n\"十二\"\n";
    let point = "File type = \"ooTextFile\"\nObject class = \"TextGrid\"\nxmin = 0\nxmax = 1\n\
        tiers? <exists>\nsize = 1\nitem []:\nitem [1]:\nclass = \"TextTier\"\nname = \"SPK0001\"\n\
        xmin = 0\nxmax = 1\npoints: size = 1\npoints [1]:\nnumber = 0.5\nmark = \"十二\"\n";
    let cases: Vec<Vec<u8>> = vec![
        short.as_bytes().to_vec(),
        point.as_bytes().to_vec(),
        valid
            .replace("intervals: size = 1", "intervals: size = 2")
            .into_bytes(),
        valid
            .replace("xmin = 0\nxmax = 1\ntext", "xmin = 0.8\nxmax = 0.2\ntext")
            .into_bytes(),
        valid
            .replace("xmin = 0\nxmax = 1\ntext", "xmin = NaN\nxmax = 1\ntext")
            .into_bytes(),
        valid.replace("SPK0001", "../../outside").into_bytes(),
        valid.replace("SPK0001", "..").into_bytes(),
        valid
            .replace("text = \"十二\"", "text = \"十二")
            .into_bytes(),
        format!("{valid}unexpected extra data\n").into_bytes(),
        vec![0xff, 0xfe, 0x00],
        vec![0xff, 0xfe, 0x00, 0xd8],
    ];
    for (index, content) in cases.iter().enumerate() {
        fs::write(&source, content).unwrap();
        let result = run_cli(&input, &output);
        assert!(!result.status.success(), "malformed case {index} must fail");
        assert!(!output.exists(), "malformed case {index} published output");
    }
}

#[cfg(unix)]
#[test]
fn existing_output_is_rejected_before_reading_input_files() {
    use std::os::unix::fs::symlink;

    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let input = root.join("textgrid_dir");
    fs::create_dir(&input).unwrap();
    fs::write(input.join("R0001.TextGrid"), "invalid synthetic TextGrid").unwrap();
    let output = root.join("existing.json");
    fs::write(&output, "unchanged").unwrap();
    let result = run_cli(&input, &output);
    assert!(!result.status.success());
    assert!(
        String::from_utf8_lossy(&result.stderr).contains("output already exists"),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(fs::read(&output).unwrap(), b"unchanged");
    let missing_target = root.join("missing-target");
    let dangling = root.join("dangling.json");
    symlink(&missing_target, &dangling).unwrap();
    let result = run_cli(&input, &dangling);
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("output already exists"));
    assert!(
        fs::symlink_metadata(&dangling)
            .unwrap()
            .file_type()
            .is_symlink()
    );
    assert!(!missing_target.exists());
}

#[cfg(unix)]
#[test]
fn concurrent_publishers_do_not_replace_the_winning_report() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let left = root.join("left");
    let right = root.join("right");
    fs::create_dir(&left).unwrap();
    fs::create_dir(&right).unwrap();
    fs::write(left.join("R0001.TextGrid"), single_interval("c1", "十二")).unwrap();
    fs::write(right.join("R0001.TextGrid"), single_interval("c1", "十五")).unwrap();
    let output = root.join("race.json");
    let spawn = |input: &std::path::Path| {
        Command::new(env!("CARGO_BIN_EXE_alimeeting-screen"))
            .arg("--textgrid-dir")
            .arg(input)
            .arg("--output")
            .arg(&output)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap()
    };
    let first = spawn(&left);
    let second = spawn(&right);
    let first = first.wait_with_output().unwrap();
    let second = second.wait_with_output().unwrap();
    assert_ne!(first.status.success(), second.status.success());
    let report: serde_json::Value = serde_json::from_slice(&fs::read(output).unwrap()).unwrap();
    assert_eq!(
        report["candidates"][0]["candidate_number"],
        if first.status.success() { 12 } else { 15 }
    );
}

#[cfg(unix)]
#[test]
fn output_is_deterministic_and_never_replaces_existing_paths() {
    use std::os::unix::fs::symlink;

    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let left = root.join("left");
    let right = root.join("right");
    fs::create_dir(&left).unwrap();
    fs::create_dir(&right).unwrap();
    for (name, mark) in [("R0002.TextGrid", "十五"), ("R0001.TextGrid", "十二")] {
        fs::write(left.join(name), single_interval("SPK0001", mark)).unwrap();
    }
    for (name, mark) in [("R0001.TextGrid", "十二"), ("R0002.TextGrid", "十五")] {
        fs::write(right.join(name), single_interval("SPK0001", mark)).unwrap();
    }
    let left_report = root.join("left.json");
    let right_report = root.join("right.json");
    assert!(run_cli(&left, &left_report).status.success());
    assert!(run_cli(&right, &right_report).status.success());
    let saved = fs::read(&left_report).unwrap();
    assert_eq!(saved, fs::read(&right_report).unwrap());
    assert!(!run_cli(&left, &left_report).status.success());
    assert_eq!(saved, fs::read(&left_report).unwrap());
    let link = root.join("existing-link.json");
    symlink(&left_report, &link).unwrap();
    assert!(!run_cli(&left, &link).status.success());
    assert_eq!(saved, fs::read(&left_report).unwrap());
    let inside = left.join("new.json");
    assert!(!run_cli(&left, &inside).status.success());
    assert!(!inside.exists());
}

#[test]
fn hardlinked_input_aliases_are_not_counted_as_distinct_recordings() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let input = root.join("textgrid_dir");
    fs::create_dir(&input).unwrap();
    let original = input.join("R0001.TextGrid");
    fs::write(&original, single_interval("SPK0001", "十二")).unwrap();
    fs::hard_link(&original, input.join("R0002.TextGrid")).unwrap();
    let output = root.join("candidates.json");
    let result = run_cli(&input, &output);
    assert!(
        !result.status.success(),
        "hardlinks cannot create independent records"
    );
    assert!(!output.exists());
}

#[test]
fn a_single_hardlink_to_an_external_file_is_rejected() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let input = root.join("textgrid_dir");
    fs::create_dir(&input).unwrap();
    let external = root.join("outside.TextGrid");
    fs::write(&external, single_interval("SPK0001", "十二")).unwrap();
    fs::hard_link(&external, input.join("R0001.TextGrid")).unwrap();
    let output = root.join("candidates.json");
    let result = run_cli(&input, &output);
    assert!(
        !result.status.success(),
        "external hardlink must be rejected"
    );
    assert!(!output.exists());
}

#[test]
fn empty_input_path_is_not_implicitly_the_current_directory() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let input = root.join("textgrid_dir");
    fs::create_dir(&input).unwrap();
    fs::write(
        input.join("R0001.TextGrid"),
        single_interval("SPK0001", "十二"),
    )
    .unwrap();
    let output = root.join("candidates.json");
    let result = Command::new(env!("CARGO_BIN_EXE_alimeeting-screen"))
        .current_dir(&input)
        .arg("--textgrid-dir")
        .arg("")
        .arg("--output")
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        !result.status.success(),
        "empty source path must be rejected"
    );
    assert!(!output.exists());
}

#[test]
fn public_module_rejects_an_empty_source_path() {
    const CHILD_OUTPUT: &str = "OMNI_RSI_EMPTY_SOURCE_TEST_OUTPUT";
    if let Some(path) = std::env::var_os(CHILD_OUTPUT) {
        let output = std::path::PathBuf::from(path);
        assert!(omni_rsi::alimeeting_screen::screen(std::path::Path::new(""), &output).is_err());
        assert!(!output.exists());
        return;
    }
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let input = root.join("textgrid_dir");
    fs::create_dir(&input).unwrap();
    fs::write(
        input.join("R0001.TextGrid"),
        single_interval("SPK0001", "十二"),
    )
    .unwrap();
    let output = root.join("candidates.json");
    let child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "alimeeting_screen::public_module_rejects_an_empty_source_path",
            "--nocapture",
        ])
        .env(CHILD_OUTPUT, &output)
        .current_dir(&input)
        .output()
        .unwrap();
    assert!(
        child.status.success(),
        "{}",
        String::from_utf8_lossy(&child.stdout)
    );
    assert!(!output.exists());
}

#[test]
fn counts_matching_tier_labels_per_recording_without_assuming_person_identity() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let input = root.join("textgrid_dir");
    fs::create_dir(&input).unwrap();
    fs::write(input.join("R0002.TextGrid"), single_interval("SPK1", "十")).unwrap();
    fs::write(input.join("R0001.TextGrid"), single_interval("SPK1", "一")).unwrap();
    let output = root.join("candidates.json");
    let result = run_cli(&input, &output);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&fs::read(output).unwrap()).unwrap();
    assert_eq!(report["summary"]["distinct_tier_ids"], 1);
    assert_eq!(report["summary"]["distinct_recording_tier_ids"], 2);
    assert_eq!(
        report["summary"]["recording_tier_counts"],
        serde_json::json!([
            {"recording_id": "R0001", "tier_name": "SPK1", "candidate_count": 1},
            {"recording_id": "R0002", "tier_name": "SPK1", "candidate_count": 1}
        ])
    );
}

#[test]
fn reports_spk_filename_suffixes_across_recordings_and_unknown_names() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let input = root.join("textgrid_dir");
    fs::create_dir(&input).unwrap();
    for (name, mark) in [
        ("R0001_SPK001.TextGrid", "一"),
        ("R0002_SPK001.TextGrid", "二"),
        ("R0003_SPK002.TextGrid", "三"),
        ("R0004_SPKx.TextGrid", "四"),
        ("R0005_SPK999.TextGrid", "普通话"),
        ("R0006_SPK.TextGrid", "普通话"),
        ("R0007_SPK001extra.TextGrid", "普通话"),
        ("R0008_SPK01.TextGrid", "五"),
    ] {
        fs::write(input.join(name), single_interval("c1", mark)).unwrap();
    }
    let output = root.join("candidates.json");
    let result = run_cli(&input, &output);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&fs::read(output).unwrap()).unwrap();
    assert_eq!(report["summary"]["distinct_tier_ids"], 1);
    assert_eq!(report["summary"]["distinct_recording_tier_ids"], 5);
    assert_eq!(report["summary"]["distinct_source_spk_suffixes"], 4);
    assert_eq!(report["summary"]["source_unknown_spk_files"], 3);
    assert_eq!(
        report["summary"]["source_spk_file_counts"],
        serde_json::json!({"SPK001": 2, "SPK002": 1, "SPK01": 1, "SPK999": 1})
    );
    assert_eq!(report["summary"]["distinct_candidate_spk_suffixes"], 3);
    assert_eq!(report["summary"]["candidate_unknown_spk_count"], 1);
    assert_eq!(
        report["summary"]["candidate_spk_counts"],
        serde_json::json!({"SPK001": 2, "SPK002": 1, "SPK01": 1})
    );
    assert_eq!(report["source_files"][0]["spk_suffix"], "SPK001");
    assert_eq!(report["source_files"][3]["spk_suffix"], "unknown");
    assert_eq!(report["candidates"][0]["spk_suffix"], "SPK001");
    assert_eq!(report["candidates"][3]["spk_suffix"], "unknown");
}

#[test]
fn permits_nonoverlapping_gaps_but_defers_neighbor_text_across_them() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let input = root.join("textgrid_dir");
    fs::create_dir(&input).unwrap();
    let grid = "File type = \"ooTextFile\"\nObject class = \"TextGrid\"\nxmin = 0\nxmax = 100\n\
        tiers? <exists>\nsize = 1\nitem []:\nitem [1]:\nclass = \"IntervalTier\"\n\
        name = \"c1\"\nxmin = 0\nxmax = 100\nintervals: size = 2\n\
        intervals [1]:\nxmin = 82.53\nxmax = 83.34\ntext = \"一\"\n\
        intervals [2]:\nxmin = 84.32\nxmax = 85\ntext = \"个\"\n";
    fs::write(input.join("R0001_SPK01.TextGrid"), grid).unwrap();
    fs::write(input.join("R0002_unknown.TextGrid"), grid).unwrap();
    let output = root.join("candidates.json");
    let result = run_cli(&input, &output);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&fs::read(output).unwrap()).unwrap();
    assert_eq!(report["summary"]["interval_count"], 4);
    assert_eq!(report["summary"]["candidate_count"], 0);
    assert_eq!(report["summary"]["deferred_adjacent_count"], 2);
    assert_eq!(report["summary"]["distinct_deferred_spk_suffixes"], 1);
    assert_eq!(
        report["summary"]["deferred_spk_counts"],
        serde_json::json!({"SPK01": 1})
    );
    assert_eq!(report["summary"]["deferred_unknown_spk_count"], 1);
    assert_eq!(report["deferred"][0]["mark"], "一");
    assert_eq!(report["deferred"][0]["spk_suffix"], "SPK01");
    assert_eq!(
        report["deferred"][0]["reason"],
        "adjacent_nonempty_interval"
    );
    assert!(report["deferred"][0].get("candidate_number").is_none());
}

#[test]
fn rejects_overlapping_and_out_of_order_intervals() {
    for (start, end) in [("83.33", "84"), ("80", "81")] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        let input = root.join("textgrid_dir");
        fs::create_dir(&input).unwrap();
        let grid = format!(
            "File type = \"ooTextFile\"\nObject class = \"TextGrid\"\nxmin = 0\nxmax = 100\n\
             tiers? <exists>\nsize = 1\nitem []:\nitem [1]:\nclass = \"IntervalTier\"\n\
             name = \"c1\"\nxmin = 0\nxmax = 100\nintervals: size = 2\n\
             intervals [1]:\nxmin = 82.53\nxmax = 83.34\ntext = \"一\"\n\
             intervals [2]:\nxmin = {start}\nxmax = {end}\ntext = \"二\"\n"
        );
        fs::write(input.join("R0001_SPK01.TextGrid"), grid).unwrap();
        let output = root.join("candidates.json");
        let result = run_cli(&input, &output);
        assert!(
            !result.status.success(),
            "overlap or reverse order must fail"
        );
        assert!(!output.exists());
    }
}
