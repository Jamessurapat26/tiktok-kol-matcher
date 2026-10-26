import json
from typing import Any

from app.models import BrandProfile


def assert_strict_object(schema: dict[str, Any]) -> None:
    """Every property is required, strict, and has no default."""
    properties = schema["properties"]
    assert set(schema["required"]) == set(properties)
    assert schema["additionalProperties"] is False
    for prop in properties.values():
        assert "default" not in prop


def assert_no_defaults(node: Any) -> None:
    if isinstance(node, dict):
        assert "default" not in node
        for value in node.values():
            assert_no_defaults(value)
    elif isinstance(node, list):
        for item in node:
            assert_no_defaults(item)


def test_brand_profile_schema_is_strict_with_no_defaults():
    schema = BrandProfile.model_json_schema()

    assert_strict_object(schema)

    # Nested models are emitted under $defs and must be strict as well.
    assert "TargetAudience" in schema["$defs"]
    assert_strict_object(schema["$defs"]["TargetAudience"])

    # No defaults anywhere in the schema (or its $defs).
    assert_no_defaults(schema)
    assert '"default"' not in json.dumps(schema)
