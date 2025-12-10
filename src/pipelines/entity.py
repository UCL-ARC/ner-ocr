"""Entity extraction pipeline module."""

import json
from pathlib import Path
from typing import cast

from loguru import logger
from pydantic import BaseModel

from src.config import AppConfig
from src.entities import ENTITY_REGISTRY
from src.entity_extraction import QwenEntityExtractor, QwenModels

from .base import BasePipeline


class EntityExtractionPipeline(BasePipeline):
    """Pipeline for entity extraction."""

    def __init__(self, config: AppConfig) -> None:
        """Initialise entity extraction pipeline with config."""
        super().__init__(config)
        self.extractor = QwenEntityExtractor(
            model=QwenModels[config.entity_extraction.model],
            device=config.entity_extraction.device,
            local=True,
        )
        self.entities = config.entity_extraction.entities

    def get_output_suffix(self) -> str:
        """Return the suffix for output files."""
        return "_entity_results"

    def process_file(self, file_path: Path) -> list[dict]:
        """Process a single file through entity extraction pipeline."""
        logger.info(f"Starting entity extraction for: {file_path}")

        with Path.open(file_path) as f:
            parsed_result = json.load(f)

        page_results = []
        for page in parsed_result:
            markdown_page = self._to_markdown(page)

            extracted_entities: dict[str, dict] = {}
            for entity in self.entities:
                entity_model = cast(type[BaseModel], ENTITY_REGISTRY[entity])
                result = self.extractor.extract_entities(
                    markdown_page, entity_model=entity_model
                )
                content = result["content"]
                if hasattr(content, "model_dump"):
                    extracted_entities[entity] = content.model_dump()
                else:
                    extracted_entities[entity] = {}

            logger.info(f"Page {page['page_result']['page']}: {extracted_entities}")

            page_results.append(
                {
                    "page": page["page_result"]["page"],
                    "page_text": markdown_page,
                    "entities": extracted_entities,
                }
            )

        return page_results

    def _to_markdown(
        self,
        page_entry: dict,
        line_threshold: int = 10,
        gap_threshold: int = 40,
    ) -> str:
        """Convert OCR JSON with bounding boxes into Markdown."""
        page_result = page_entry["page_result"]
        items = page_result.get("data", [])

        processed: list[dict] = []
        for item in items:
            box = item.get("box")
            if not box or len(box) != 4:  # noqa: PLR2004
                continue

            x_min, y_min, _, _ = box
            text = item.get("transformer_text") or item.get("text") or ""
            text = text.strip()
            if text:
                processed.append({"x": x_min, "y": y_min, "text": text})

        if not processed:
            return ""

        processed.sort(key=lambda it: (it["y"], it["x"]))

        lines: list[tuple[float, list[dict]]] = []
        current_line: list[dict] = []
        last_y: int | None = None

        for it in processed:
            y = it["y"]
            if last_y is None or abs(y - last_y) <= line_threshold:
                current_line.append(it)
            else:
                current_line.sort(key=lambda x: x["x"])
                avg_y = sum(tok["y"] for tok in current_line) / len(current_line)
                lines.append((avg_y, current_line))
                current_line = [it]
            last_y = y

        if current_line:
            current_line.sort(key=lambda x: x["x"])
            avg_y = sum(tok["y"] for tok in current_line) / len(current_line)
            lines.append((avg_y, current_line))

        markdown_lines: list[str] = []
        last_line_y: float | None = None

        for line_y, line_tokens in lines:
            if last_line_y is not None and (line_y - last_line_y) > gap_threshold:
                markdown_lines.extend(["", "", ""])
            markdown_lines.append("   ".join(tok["text"] for tok in line_tokens))
            last_line_y = line_y

        return "\n".join(markdown_lines)
