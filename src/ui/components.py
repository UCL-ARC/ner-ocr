"""UI component builders for Gradio."""

from typing import Any

import yaml

from src.config import AppConfig


def build_config_editor(config: AppConfig | None) -> str:
    """Convert config to YAML string for editing."""
    if config is None:
        return "# No configuration loaded\n# Upload a config.yaml or use defaults"

    config_dict = {
        "ocr": {
            "max_side_limit": config.ocr.max_side_limit,
            "ocr_timeout": config.ocr.ocr_timeout,
            "use_doc_orientation_classify": config.ocr.use_doc_orientation_classify,
            "use_doc_unwarping": config.ocr.use_doc_unwarping,
            "use_textline_orientation": config.ocr.use_textline_orientation,
            "return_word_box": config.ocr.return_word_box,
            "device": config.ocr.device,
        },
        "transformer_ocr": {
            "model": config.transformer_ocr.model,
            "device": config.transformer_ocr.device,
        },
        "entity_extraction": {
            "model": config.entity_extraction.model,
            "device": config.entity_extraction.device,
            "entities": config.entity_extraction.entities,
        },
        "queries": config.queries,
    }
    return yaml.dump(config_dict, default_flow_style=False, sort_keys=False)


def parse_config_from_editor(yaml_str: str) -> dict:
    """Parse YAML string back to config dict."""
    return yaml.safe_load(yaml_str)


def create_status_html(status: dict[str, Any]) -> str:
    """Create compact HTML status display."""
    return """
    <div style="padding: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 8px; color: white; font-family: system-ui; font-size: 12px;">
        <div style="display: flex; flex-wrap: wrap; gap: 8px; align-items: center;">
            <span><strong>📄</strong> {document}</span>
            <span>|</span>
            <span><strong>📑</strong> {total_pages} pages</span>
            <span>|</span>
            <span><strong>🔤</strong> {total_regions} regions</span>
        </div>
        <div style="display: flex; flex-wrap: wrap; gap: 12px; margin-top: 8px;">
            <span>1️⃣ OCR: {ocr_status}</span>
            <span>2️⃣ Search: {search_status}</span>
            <span>3️⃣ Enhance: {enhancement_status}</span>
            <span>4️⃣ Entity: {entity_status}</span>
        </div>
    </div>
    """.format(**status)


def create_instructions_html() -> str:
    """Create compact instructions panel HTML."""
    return """
    <div style="padding: 10px; background: #2d3748; border-radius: 8px;
                border-left: 4px solid #667eea; color: #e2e8f0; font-size: 12px;">
        <strong>🚀 Quick Start:</strong>
        <span style="color: #a0aec0;">
            Upload doc → Run OCR → Search regions → Enhance text → Extract entities → Export
        </span>
    </div>
    """
