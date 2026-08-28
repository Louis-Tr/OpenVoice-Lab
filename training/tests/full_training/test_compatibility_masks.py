from __future__ import annotations

import ast
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("module_name", "forward_targets"),
    [
        ("lora_compatibility.py", {"wrapped", "reloaded", "merged"}),
        ("reduction_factor_compatibility.py", {"model"}),
    ],
)
def test_guided_attention_compatibility_forwards_include_attention_mask(
    module_name: str, forward_targets: set[str]
) -> None:
    source = (
        Path(__file__).parents[2] / "full_training" / module_name
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in forward_targets
        and any(keyword.arg == "labels" for keyword in node.keywords)
    ]

    assert {call.func.id for call in calls} == forward_targets
    for call in calls:
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        assert "attention_mask" in keywords
        assert isinstance(keywords["attention_mask"], ast.Name)
        assert keywords["attention_mask"].id == "attention_mask"


@pytest.mark.parametrize(
    "module_name",
    ["lora_compatibility.py", "reduction_factor_compatibility.py"],
)
def test_compatibility_attention_mask_matches_input_shape(module_name: str) -> None:
    source = (
        Path(__file__).parents[2] / "full_training" / module_name
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "attention_mask"
            for target in node.targets
        )
    ]

    assert len(assignments) == 1
    value = assignments[0].value
    assert isinstance(value, ast.Call)
    assert isinstance(value.func, ast.Attribute)
    assert value.func.attr == "ones_like"
    assert len(value.args) == 1
    assert isinstance(value.args[0], ast.Name)
    assert value.args[0].id == "input_ids"
