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


def test_normalizer_does_not_rewrite_mismatched_existing_code() -> None:
    registry = MechanismRegistry.load("data/term_inventory.json")

    normalized = registry.normalize_answer("采用 1-b 拖拽来移动对象。")

    assert normalized == "采用 1-b 拖拽来移动对象。"
