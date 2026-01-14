"""Entities data models."""

from loguru import logger
from pydantic import BaseModel, Field

from .entity_builder import load_entities_from_yaml


class AddressEntity(BaseModel):
    """Data model for an address entity."""

    street: str | None = Field(None, description="Street address")
    city: str | None = Field(None, description="City name")
    state: str | None = Field(None, description="State or province")
    postal_code: str | None = Field(None, description="Postal or ZIP code")
    country: str | None = Field(None, description="Country name")
    raw_text: str = Field(
        ..., description="Raw string of the address without formatting"
    )
    address_type: str | None = Field(
        None,
        description="Type of address. Either place of birth or place of residence",
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
