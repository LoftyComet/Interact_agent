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
