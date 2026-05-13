"""Entities data models."""

from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field

from .entity_builder import load_entities_from_yaml


class AddressEntity(BaseModel):
    """Data model for an address entity."""

    model_config = {
        "json_schema_extra": {
            "description": (
                "An address entity. If the source text indicates 'same' or 'same address', "
                "resolve it to the full address it refers to elsewhere in the text."
            )
        }
    }

    street: str | None = Field(None, description="Street address")
    city: str | None = Field(None, description="City name")
    state: str | None = Field(None, description="State or province")
    postal_code: str | None = Field(None, description="Postal or ZIP code")
    country: str | None = Field(None, description="Country name")
    raw_text: str = Field(
        ...,
        description=(
            "Raw string of the address without formatting. "
            "If the text just says 'same' or 'same address', copy the full address "
            "it refers to from elsewhere in the document."
        ),
    )
    address_type: Literal["place of birth", "place of residence"] | None = Field(
        None,
        description="Type of address: 'place of birth' or 'place of residence'",
    )


class AddressEntityList(BaseModel):
    """Data model for a list of address entities."""

    addresses: list[AddressEntity] = Field(
        ..., description="List of extracted address entities"
    )


# Built-in entities (always available)
_BUILTIN_ENTITIES: dict[str, type[BaseModel]] = {
    "AddressEntity": AddressEntity,
    "AddressEntityList": AddressEntityList,
}

# Load custom entities from YAML
_custom_entities = load_entities_from_yaml()
if _custom_entities:
    logger.info(f"Loaded {len(_custom_entities)} custom entities from entities.yaml")

# Combined registry
ENTITY_REGISTRY: dict[str, type[BaseModel]] = {
    **_BUILTIN_ENTITIES,
    **_custom_entities,
}
