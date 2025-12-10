"""Configuration dataclasses and YAML loader."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class OCRConfig:
    """Configuration for OCR pipeline."""

    max_side_limit: int = 1500
    ocr_timeout: int = 400
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False
    use_textline_orientation: bool = False
    return_word_box: bool = True
    device: str = "cpu"


@dataclass
class TransformerOCRConfig:
    """Configuration for Transformer OCR."""

    model: str = "LARGE_HANDWRITTEN"
    device: str = "cpu"


@dataclass
class EntityExtractionConfig:
    """Configuration for entity extraction pipeline."""

    model: str = "QWEN3_1_7B"
    device: str = "cpu"
    entities: list[str] = field(default_factory=lambda: ["AddressEntityList"])


@dataclass
class AppConfig:
    """Main application configuration."""

    ocr: OCRConfig
    transformer_ocr: TransformerOCRConfig
    entity_extraction: EntityExtractionConfig
    queries: list[dict] = field(default_factory=list)


def load_config(path: Path) -> AppConfig:
    """Load YAML config file and return typed AppConfig."""
    with Path.open(path) as f:
        raw = yaml.safe_load(f)

    return AppConfig(
        ocr=OCRConfig(**raw.get("ocr", {})),
        transformer_ocr=TransformerOCRConfig(**raw.get("transformer_ocr", {})),
        entity_extraction=EntityExtractionConfig(**raw.get("entity_extraction", {})),
        queries=raw.get("queries", []),
    )
