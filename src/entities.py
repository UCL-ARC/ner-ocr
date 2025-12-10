"""Entities data models."""

from pydantic import BaseModel, Field


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
        description="Type of address (e.g., institution, place of birth, current residence)",
    )


class AddressEntityList(BaseModel):
    """Data model for a list of address entities."""

    addresses: list[AddressEntity] = Field(
        ..., description="List of extracted address entities"
    )


ENTITY_REGISTRY: dict[str, type[BaseModel]] = {
    "AddressEntity": AddressEntity,
    "AddressEntityList": AddressEntityList,
}
