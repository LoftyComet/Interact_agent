from gesture_agent.knowledge import MechanismRegistry


def test_registry_loads_all_36_mechanisms_from_runtime_inventory() -> None:
    registry = MechanismRegistry.load("data/term_inventory.json")

    assert len(registry.entries) == 36
    assert registry.by_code["1-a"].label == "开关"
    assert registry.by_code["2-a"].label_en == "Drag"
    assert registry.by_code["4-i"].label == "捏合解耦"


def test_registry_accepts_matching_code_name_and_english() -> None:
    registry = MechanismRegistry.load("data/term_inventory.json")

    issues = registry.validate_answer("可以采用 2-a 拖拽（Drag）来连续改变位置。")

    assert issues == ()


def test_registry_rejects_missing_and_mismatched_codes() -> None:
    registry = MechanismRegistry.load("data/term_inventory.json")

    missing = registry.validate_answer("可以采用拖拽来改变位置。")
    mismatch = registry.validate_answer("可以采用 1-b 拖拽来改变位置。")

    assert any(issue.issue_type == "missing_mechanism_code" for issue in missing)
    assert any(issue.issue_type == "mechanism_code_mismatch" for issue in mismatch)
    assert any(issue.issue_type == "mechanism_name_mismatch" for issue in mismatch)


def test_longest_name_match_avoids_nested_open_switch_false_positive() -> None:
    registry = MechanismRegistry.load("data/term_inventory.json")

    issues = registry.validate_answer("1-g 多点开关（Multi-Switch）适用于多个触点。")

    assert issues == ()


def test_registry_rejects_code_outside_registry() -> None:
    registry = MechanismRegistry.load("data/term_inventory.json")

    issues = registry.validate_answer("采用 9-z 未知机制。")

    assert any(issue.issue_type == "unknown_mechanism_code" for issue in issues)


def test_registry_rejects_known_code_without_its_canonical_name() -> None:
    registry = MechanismRegistry.load("data/term_inventory.json")

    issues = registry.validate_answer("这里使用 4-i 含义识别来处理语音。")

    assert any(
        issue.issue_type == "mechanism_name_mismatch"
        and issue.expected.startswith("4-i 捏合解耦")
        for issue in issues
    )


def test_short_english_name_does_not_match_inside_another_word() -> None:
    registry = MechanismRegistry.load("data/term_inventory.json")

    issues = registry.validate_answer("Pressure is a physical property.")

    assert issues == ()


def test_ordinary_action_verb_is_not_forced_into_mechanism_format() -> None:
    registry = MechanismRegistry.load("data/term_inventory.json")

    issues = registry.validate_answer("用户同时按下两个按钮即可截屏。")

    assert issues == ()


def test_question_targeted_mechanism_requires_code_on_first_mention() -> None:
    registry = MechanismRegistry.load("data/term_inventory.json")

    issues = registry.validate_answer("长按需要保持一段时间。", required_labels=["长按"])

    assert any(issue.issue_type == "missing_mechanism_code" for issue in issues)


def test_only_first_formal_mention_requires_code() -> None:
    registry = MechanismRegistry.load("data/term_inventory.json")

    issues = registry.validate_answer(
        "2-a 拖拽（Drag）是交互机制。后文再简称拖拽。",
        required_labels=["拖拽"],
    )

    assert issues == ()


def test_normalizer_inserts_missing_code_for_formal_first_mention() -> None:
    registry = MechanismRegistry.load("data/term_inventory.json")

    normalized = registry.normalize_answer(
        "常用交互机制包括单击、拖拽和捏合缩放。后文简称拖拽。"
    )

    assert "1-b 单击" in normalized
    assert "2-a 拖拽" in normalized
    assert "2-g 捏合缩放" in normalized
    assert normalized.count("2-a") == 1


def test_normalizer_repairs_mismatched_existing_code() -> None:
    registry = MechanismRegistry.load("data/term_inventory.json")

    normalized = registry.normalize_answer("采用 1-b 拖拽来移动对象。")

    assert normalized == "采用 2-a 拖拽来移动对象。"


def test_normalizer_expands_known_bare_code_and_removes_unknown_code() -> None:
    registry = MechanismRegistry.load("data/term_inventory.json")

    normalized = registry.normalize_answer(
        "它属于 2-b 条目，也可参考起始选择机制（6-b）。"
    )

    assert "2-b 甩动（Flick） 条目" in normalized
    assert "起始选择机制。" in normalized
    assert "6-b" not in normalized
    assert registry.validate_answer(normalized) == ()


def test_normalizer_repairs_table_code_and_duplicate_parenthetical_code() -> None:
    registry = MechanismRegistry.load("data/term_inventory.json")
    answer = (
        "| 翻动（Swipe） | 2-c | 锚点归位 |\n"
        "可使用 **1-i 多点同时点击**（1-i）组合输入。"
    )

    normalized = registry.normalize_answer(answer)

    assert "2-c 翻动（Swipe）" in normalized
    assert normalized.count("1-i") == 1
    assert registry.validate_answer(normalized) == ()


def test_normalizer_repairs_noncanonical_double_press_name_and_code() -> None:
    registry = MechanismRegistry.load("data/term_inventory.json")

    normalized = registry.normalize_answer(
        "避免双按（1-i），但可以研究双按拖拽（3-d）。"
    )

    assert normalized == "避免双击（1-f），但可以研究双按拖拽（3-c）。"
    assert registry.validate_answer(normalized) == ()


def test_normalizer_repairs_comma_separated_candidate_codes() -> None:
    registry = MechanismRegistry.load("data/term_inventory.json")
    answer = "手势包括“单击”（1-a，1-b Single Tap）和“按下”（1-b，1-c Press）。"

    normalized = registry.normalize_answer(answer, required_labels=["单击", "按下"])

    assert "单击”（1-b Single Tap）" in normalized
    assert "按下”（1-c Press）" in normalized
    assert registry.validate_answer(normalized, required_labels=["单击", "按下"]) == ()


def test_other_mechanism_code_in_same_clause_does_not_create_false_mismatch() -> None:
    registry = MechanismRegistry.load("data/term_inventory.json")

    normalized = registry.normalize_answer(
        "3-b 长按拖拽与 2-a 拖拽不同，内部是先长按再拖拽。",
        required_labels=["长按拖拽", "拖拽", "长按"],
    )
    issues = registry.validate_answer(
        normalized,
        required_labels=["长按拖拽", "拖拽", "长按"],
    )

    assert "1-e 长按" in normalized
    assert "2-a 拖拽" in normalized
    assert not any(issue.issue_type == "mechanism_code_mismatch" for issue in issues)


def test_ordinary_press_is_not_linked_to_distant_switch_code() -> None:
    registry = MechanismRegistry.load("data/term_inventory.json")

    issues = registry.validate_answer(
        "书中把 2-d 滑动切换误用为越界切换，触发时机会根据按下的位置变化。"
    )

    assert not any(issue.mention == "按下" for issue in issues)
