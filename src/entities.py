"""Entities data models."""

from pydantic import BaseModel, Field


class AddressEntity(BaseModel):
    """Data model for an address entity."""

    street: str | None = Field(None, description="Street address")
    city: str | None = Field(None, description="City name")
    state: str | None = Field(None, description="State or province")
    postal_code: str | None = Field(None, description="Postal or ZIP code")
    country: str | None = Field(None, description="Country name")
