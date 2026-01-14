"""Build Pydantic models from YAML entity definitions."""

from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from pydantic import BaseModel, Field, create_model

# Supported field types mapping
# Using Any for values since Union types are not compatible with type[T]
SUPPORTED_TYPES: dict[str, Any] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "str | None": str | None,
    "int | None": int | None,
    "float | None": float | None,
    "bool | None": bool | None,
}

# Default path for entities YAML
DEFAULT_ENTITIES_PATH = Path("entities.yaml")


def load_entities_from_yaml(
    yaml_path: Path | None = None,
) -> dict[str, type[BaseModel]]:
    """
    Load entity definitions from YAML and create Pydantic models.

    Args:
        yaml_path: Path to the entities YAML file. Defaults to entities.yaml

    Returns:
        Dictionary mapping entity names to Pydantic model classes

    """
    yaml_path = yaml_path or DEFAULT_ENTITIES_PATH

    if not yaml_path.exists():
        logger.debug(f"No custom entities file found at {yaml_path}")
        return {}

    try:
        with Path.open(yaml_path) as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse entities YAML: {e}")
        return {}

    if not config or "entities" not in config:
        logger.debug("No entities defined in YAML")
        return {}

    entities_config = config.get("entities") or {}
    if not entities_config:
        return {}

    result: dict[str, type[BaseModel]] = {}

    for entity_name, entity_def in entities_config.items():
        try:
            models = _build_entity(entity_name, entity_def)
            result.update(models)
            logger.info(f"Loaded custom entity: {entity_name}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to build entity '{entity_name}': {e}")

    return result


def _build_entity(
    name: str,
    definition: dict[str, Any],
) -> dict[str, type[BaseModel]]:
    """
    Build a Pydantic model from an entity definition.

    Args:
        name: The entity name
        definition: The entity definition from YAML

    Returns:
        Dictionary with the entity model (and list wrapper if requested)

    """
    if not definition:
        error_msg = f"Empty definition for entity '{name}'"
        raise ValueError(error_msg)

    description = definition.get("description", f"Custom entity: {name}")
    fields_config = definition.get("fields", {})
    create_list = definition.get("create_list", False)

    if not fields_config:
        error_msg = f"No fields defined for entity '{name}'"
        raise ValueError(error_msg)

    # Build field definitions for create_model
    field_definitions: dict[str, Any] = {}

    for field_name, field_def in fields_config.items():
        if isinstance(field_def, str):
            # Simple format: field_name: "str | None"
            field_type = SUPPORTED_TYPES.get(field_def, str | None)
            field_definitions[field_name] = (field_type, Field(None))
        elif isinstance(field_def, dict):
            # Full format with description
            type_str = field_def.get("type", "str | None")
            field_type = SUPPORTED_TYPES.get(type_str, str | None)
            field_desc = field_def.get("description", "")
            is_required = field_def.get("required", False)

            if is_required:
                # Required field - use ... as default
                base_type_str = type_str.replace(" | None", "")
                field_type = SUPPORTED_TYPES.get(base_type_str, str)
                field_definitions[field_name] = (
                    field_type,
                    Field(..., description=field_desc),
                )
            else:
                field_definitions[field_name] = (
                    field_type,
                    Field(None, description=field_desc),
                )
        else:
            error_msg = (
                f"Invalid field definition for '{field_name}' in entity '{name}'"
            )
            raise TypeError(error_msg)

    # Create the base entity model
    entity_model = create_model(
        name,
        __doc__=description,
        **field_definitions,
    )

    result = {name: entity_model}

    # Optionally create list wrapper
    if create_list:
        list_name = f"{name}List"
        # Create plural field name (simple pluralisation)
        items_field = name[0].lower() + name[1:]
        if items_field.endswith("y"):
            items_field = items_field[:-1] + "ies"
        elif items_field.endswith("Entity"):
            items_field = items_field[:-6] + "Entities"
        else:
            items_field += "s"

        # Build list field definition - use tuple format for create_model
        list_field_type: Any = list[entity_model]  # type: ignore[valid-type]
        list_field_def = {
            items_field: (
                list_field_type,
                Field(..., description=f"List of extracted {name} entities"),
            )
        }
        list_model = create_model(
            list_name,
            __doc__=f"List of {name} entities",
            **list_field_def,  # type: ignore[call-overload]
        )
        result[list_name] = list_model

    return result


def validate_entities_yaml(yaml_path: Path | None = None) -> list[str]:
    """
    Validate an entities YAML file and return any errors.

    Args:
        yaml_path: Path to the entities YAML file

    Returns:
        List of error messages (empty if valid)

    """
    yaml_path = yaml_path or DEFAULT_ENTITIES_PATH
    errors: list[str] = []

    if not yaml_path.exists():
        return [f"File not found: {yaml_path}"]

    try:
        with Path.open(yaml_path) as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return [f"YAML parse error: {e}"]

    if not config:
        return ["Empty YAML file"]

    if "entities" not in config:
        return ["Missing 'entities' key in YAML"]

    entities_config = config.get("entities") or {}

    for entity_name, entity_def in entities_config.items():
        # Validate entity name
        if not entity_name[0].isupper():
            errors.append(f"Entity '{entity_name}': name should start with uppercase")

        if not entity_def:
            errors.append(f"Entity '{entity_name}': empty definition")
            continue

        fields = entity_def.get("fields", {})
        if not fields:
            errors.append(f"Entity '{entity_name}': no fields defined")
            continue

        for field_name, field_def in fields.items():
            if not field_name.isidentifier():
                errors.append(
                    f"Entity '{entity_name}': invalid field name '{field_name}'"
                )

            if isinstance(field_def, dict):
                type_str = field_def.get("type", "str | None")
                if type_str not in SUPPORTED_TYPES:
                    errors.append(
                        f"Entity '{entity_name}.{field_name}': "
                        f"unsupported type '{type_str}'"
                    )

    return errors
