"""Main Gradio application for NER-OCR pipeline workbench."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, cast

import cv2
import gradio as gr
import numpy as np
import yaml
from loguru import logger
from pydantic import BaseModel

from src.bounding_box import PaddleOCRWrapper
from src.custom_types import PositionalQuery, SemanticQuery
from src.entities import ENTITY_REGISTRY
from src.entity_extraction import QwenEntityExtractor, QwenModels
from src.pdf_processing import PDFProcessor
from src.rpa import RPAProcessor
from src.transformer_ocr import TrOCRModels, TrOCRWrapper
from src.utils import to_serialisable

from .components import (
    create_instructions_html,
    create_status_html,
)
from .state import PipelineState
from .visualisation import (
    draw_ocr_boxes,
    get_highlighted_indices,
    render_entity_results,
    render_results_table,
)

# Global state (per-session in production you'd use gr.State)
STATE = PipelineState()

# Lazy-loaded processors
_ocr_processor: PaddleOCRWrapper | None = None
_transformer: TrOCRWrapper | None = None
_entity_extractor: QwenEntityExtractor | None = None

# Track which models are loaded
_transformer_model_name: str | None = None
_entity_model_name: str | None = None

# Track OCR config hash to detect changes
_ocr_config_hash: str | None = None

# Available models for dropdowns
TROCR_MODEL_CHOICES = [model.name for model in TrOCRModels]
QWEN_MODEL_CHOICES = [model.name for model in QwenModels]
ENTITY_CHOICES = list(ENTITY_REGISTRY.keys())


def get_ocr_processor(
    max_side_limit: int,
    ocr_timeout: int,
    use_doc_orientation_classify: bool,
    use_doc_unwarping: bool,
    use_textline_orientation: bool,
    return_word_box: bool,
    device: str,
) -> PaddleOCRWrapper:
    """Get or create OCR processor with specified config."""
    global _ocr_processor, _ocr_config_hash

    # Create hash of config to detect changes
    config_hash = f"{max_side_limit}_{ocr_timeout}_{use_doc_orientation_classify}_{use_doc_unwarping}_{use_textline_orientation}_{return_word_box}_{device}"

    if _ocr_processor is not None and _ocr_config_hash == config_hash:
        return _ocr_processor

    logger.info("Initializing PaddleOCR processor...")
    _ocr_processor = PaddleOCRWrapper(
        max_side_limit=max_side_limit,
        ocr_timeout=ocr_timeout,
        use_doc_orientation_classify=use_doc_orientation_classify,
        use_doc_unwarping=use_doc_unwarping,
        use_textline_orientation=use_textline_orientation,
        return_word_box=return_word_box,
        device=device,
    )
    _ocr_config_hash = config_hash
    return _ocr_processor


def get_transformer(model_name: str, device: str = "cpu") -> TrOCRWrapper:
    """Get or create transformer OCR with specific model."""
    import os

    global _transformer, _transformer_model_name
    # Reset if model changed
    if _transformer is not None and _transformer_model_name != model_name:
        _transformer = None
    if _transformer is None:
        # Check if running in offline/TRE mode
        offline_mode = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
        logger.info(
            f"Initializing TrOCR transformer with model: {model_name} (offline={offline_mode})"
        )
        _transformer = TrOCRWrapper(
            model=TrOCRModels[model_name],
            device=device,
            local=offline_mode,
        )
        _transformer_model_name = model_name
    return _transformer


def get_entity_extractor(model_name: str, device: str = "cpu") -> QwenEntityExtractor:
    """Get or create entity extractor with specific model."""
    import os

    global _entity_extractor, _entity_model_name
    # Reset if model changed
    if _entity_extractor is not None and _entity_model_name != model_name:
        _entity_extractor = None
    if _entity_extractor is None:
        # Check if running in offline/TRE mode
        offline_mode = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
        logger.info(
            f"Initializing Qwen entity extractor with model: {model_name} (offline={offline_mode})"
        )
        _entity_extractor = QwenEntityExtractor(
            model=QwenModels[model_name],
            device=device,
            local=offline_mode,
        )
        _entity_model_name = model_name
    return _entity_extractor


def reset_processors() -> None:
    """Reset all lazy-loaded processors."""
    global _ocr_processor, _transformer, _entity_extractor
    global _transformer_model_name, _entity_model_name, _ocr_config_hash
    _ocr_processor = None
    _transformer = None
    _entity_extractor = None
    _transformer_model_name = None
    _entity_model_name = None
    _ocr_config_hash = None


def get_document_preview(file_path: Path) -> list[np.ndarray]:
    """Get preview images of all pages of a document."""
    images = []
    try:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            pdf_processor = PDFProcessor()
            images = pdf_processor.pdf_to_images(file_path)
        elif suffix in {".png", ".jpg", ".jpeg", ".tiff", ".tif"}:
            img = cv2.imread(str(file_path))
            if img is not None:
                images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB)]
    except Exception as e:
        logger.warning(f"Could not generate preview: {e}")
    return images


# ============================================================================
# Event Handlers
# ============================================================================


def load_document(file: gr.File | None) -> tuple[str, np.ndarray | None, str, str]:
    """Handle document upload and show preview."""
    if file is None:
        return (
            "No file uploaded",
            None,
            create_status_html(STATE.to_summary()),
            "Page 0 / 0",
        )

    # Copy uploaded file to a persistent temp location
    temp_dir = Path(tempfile.gettempdir()) / "ner_ocr_workbench"
    temp_dir.mkdir(exist_ok=True)

    src_path = Path(file.name)
    dst_path = temp_dir / src_path.name
    shutil.copy2(src_path, dst_path)

    STATE.document_path = dst_path
    STATE.document_name = src_path.name
    STATE.reset()

    logger.info(f"Document loaded: {dst_path}")

    # Generate preview images for all pages
    STATE.preview_images = get_document_preview(dst_path)
    STATE.current_page = 0

    # Show first page preview
    preview = STATE.preview_images[0] if STATE.preview_images else None
    total_pages = len(STATE.preview_images) if STATE.preview_images else 0
    page_info = f"Page 1 / {total_pages}" if total_pages > 0 else "Page 0 / 0"

    return (
        f"✅ Loaded: {src_path.name}",
        preview,
        create_status_html(STATE.to_summary()),
        page_info,
    )


def run_ocr(
    max_side_limit: int,
    ocr_timeout: int,
    use_doc_orientation_classify: bool,
    use_doc_unwarping: bool,
    use_textline_orientation: bool,
    return_word_box: bool,
    device: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple[np.ndarray | None, list[list[str]], str, str, str]:
    """Run OCR on the loaded document with specified configuration."""
    if STATE.document_path is None:
        return (
            None,
            [],
            "❌ No document loaded. Please upload a document first.",
            create_status_html(STATE.to_summary()),
            "Page 0 / 0",
        )

    progress(0, desc="Initializing OCR engine...")

    try:
        # Store the OCR config used
        STATE.ocr_config = {
            "max_side_limit": max_side_limit,
            "ocr_timeout": ocr_timeout,
            "use_doc_orientation_classify": use_doc_orientation_classify,
            "use_doc_unwarping": use_doc_unwarping,
            "use_textline_orientation": use_textline_orientation,
            "return_word_box": return_word_box,
            "device": device,
        }

        ocr = get_ocr_processor(
            max_side_limit=max_side_limit,
            ocr_timeout=ocr_timeout,
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_textline_orientation=use_textline_orientation,
            return_word_box=return_word_box,
            device=device,
        )

        progress(0.2, desc="Running OCR on document...")
        STATE.ocr_results = ocr.extract(STATE.document_path)
        STATE.ocr_complete = True
        STATE.current_page = 0

        # Reset downstream stages
        STATE.search_complete = False
        STATE.enhancement_complete = False
        STATE.entity_complete = False
        STATE.search_results = None

        progress(1.0, desc="Complete!")

        # Generate visualisation for first page
        page_result = STATE.ocr_results[0]
        vis_image = None
        if page_result.original_image is not None:
            vis_image = draw_ocr_boxes(page_result.original_image, page_result.data)

        # Generate table
        table_data = render_results_table(page_result.data)

        total_regions = sum(len(p.data) for p in STATE.ocr_results)
        status_msg = (
            f"✅ OCR complete: {len(STATE.ocr_results)} page(s), "
            f"{total_regions} text region(s) detected"
        )

        page_info = f"Page 1 / {len(STATE.ocr_results)}"

        return (
            vis_image,
            table_data,
            status_msg,
            create_status_html(STATE.to_summary()),
            page_info,
        )

    except Exception as e:
        logger.exception("OCR failed")
        return (
            None,
            [],
            f"❌ OCR failed: {e}",
            create_status_html(STATE.to_summary()),
            "Page 0 / 0",
        )


def run_search(
    query_type: str,
    semantic_query: str,
    positional_x1: float,
    positional_y1: float,
    positional_x2: float,
    positional_y2: float,
    threshold: float,
    search_padding: float,
    progress: gr.Progress = gr.Progress(),
) -> tuple[np.ndarray | None, list[list[str]], str, str]:
    """Run search on OCR results."""
    if not STATE.ocr_complete or STATE.ocr_results is None:
        return (
            None,
            [],
            "❌ Please run OCR first before searching.",
            create_status_html(STATE.to_summary()),
        )

    progress(0.1, desc="Initializing search...")

    query: SemanticQuery | PositionalQuery

    try:
        rpa = RPAProcessor(search_type=query_type, verbose=False)

        if query_type == "semantic":
            if not semantic_query.strip():
                return (
                    None,
                    [],
                    "❌ Please enter a search query.",
                    create_status_html(STATE.to_summary()),
                )
            query = SemanticQuery(
                text=semantic_query.strip(),
                threshold=threshold,
                search_padding=search_padding,
                search_type="fuzzy",
            )
        else:
            # Positional query using rectangle coordinates
            query = PositionalQuery(
                x1=positional_x1, y1=positional_y1, x2=positional_x2, y2=positional_y2
            )

        # Store the search config used
        STATE.search_config = {
            "query_type": query_type,
            "semantic_query": semantic_query if query_type == "semantic" else None,
            "positional_x1": positional_x1 if query_type == "positional" else None,
            "positional_y1": positional_y1 if query_type == "positional" else None,
            "positional_x2": positional_x2 if query_type == "positional" else None,
            "positional_y2": positional_y2 if query_type == "positional" else None,
            "threshold": threshold,
            "search_padding": search_padding,
        }

        progress(0.3, desc="Searching documents...")

        STATE.search_results = []
        for idx, page in enumerate(STATE.ocr_results):
            result = rpa.search(page, query, task="UI Search")
            STATE.search_results.append(result)
            progress(0.3 + 0.6 * ((idx + 1) / len(STATE.ocr_results)))

        STATE.search_complete = True
        # Reset downstream stages
        STATE.enhancement_complete = False
        STATE.entity_complete = False

        progress(1.0, desc="Complete!")

        # Visualise first page with highlights
        page_result = STATE.ocr_results[STATE.current_page]
        search_result = STATE.search_results[STATE.current_page]

        # Find indices of matched results
        highlighted = get_highlighted_indices(
            page_result.data, search_result.page_result.data
        )

        vis_image = None
        if page_result.original_image is not None:
            vis_image = draw_ocr_boxes(
                page_result.original_image,
                page_result.data,
                highlight_indices=highlighted,
            )

        # Table of matched results only
        table_data = render_results_table(search_result.page_result.data)

        total_matches = sum(len(r.page_result.data) for r in STATE.search_results)
        return (
            vis_image,
            table_data,
            f"✅ Found {total_matches} matching region(s) across all pages",
            create_status_html(STATE.to_summary()),
        )

    except Exception as e:
        logger.exception("Search failed")
        return (
            None,
            [],
            f"❌ Search failed: {e}",
            create_status_html(STATE.to_summary()),
        )


def run_enhancement(
    model_name: str,
    device: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple[list[list[str]], str, str]:
    """Run transformer enhancement on search results."""
    if not STATE.search_complete or STATE.search_results is None:
        return (
            [],
            "❌ Please run Search first to select regions for enhancement.",
            create_status_html(STATE.to_summary()),
        )

    # Check if there are any results to enhance
    total_items = sum(len(r.page_result.data) for r in STATE.search_results)
    if total_items == 0:
        return (
            [],
            "❌ No search results to enhance. Try a different search query.",
            create_status_html(STATE.to_summary()),
        )

    # Store the enhancement config used
    STATE.enhancement_config = {
        "model": model_name,
        "device": device,
    }

    progress(0.05, desc=f"Loading {model_name} model (this may take a moment)...")

    try:
        transformer = get_transformer(model_name, device)

        progress(0.15, desc="Enhancing text regions...")

        processed = 0
        for search_result in STATE.search_results:
            for item in search_result.page_result.data:
                if item.bbox_image is not None:
                    result = transformer.predict(item.bbox_image)
                    item.transformer_text = result.transformer_text
                    item.transformer_score = result.score

                processed += 1
                progress(
                    0.15 + 0.85 * (processed / total_items),
                    desc=f"Processing region {processed}/{total_items}...",
                )

        STATE.enhancement_complete = True
        # Reset downstream
        STATE.entity_complete = False

        progress(1.0, desc="Complete!")

        # Build comparison table
        table_data = []
        for search_result in STATE.search_results:
            table_data.extend(render_results_table(search_result.page_result.data))

        return (
            table_data,
            f"✅ Enhanced {total_items} text region(s) using {model_name}",
            create_status_html(STATE.to_summary()),
        )

    except Exception as e:
        logger.exception("Enhancement failed")
        return (
            [],
            f"❌ Enhancement failed: {e}",
            create_status_html(STATE.to_summary()),
        )


def run_entity_extraction(
    model_name: str,
    device: str,
    entity_types: list[str],
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, str, str, str]:
    """
    Run entity extraction on enhanced results.

    Returns:
        Tuple of (input_text, entity_markdown, status_message, status_html)

    """
    if not STATE.enhancement_complete or STATE.search_results is None:
        return (
            "",
            "",
            "❌ Please run Enhancement first before extracting entities.",
            create_status_html(STATE.to_summary()),
        )

    if not entity_types:
        return (
            "",
            "",
            "❌ Please select at least one entity type to extract.",
            create_status_html(STATE.to_summary()),
        )

    # Store the entity extraction config used
    STATE.entity_config = {
        "model": model_name,
        "device": device,
        "entities": entity_types,
    }

    progress(0.05, desc=f"Loading {model_name} model (this may take a moment)...")

    try:
        extractor = get_entity_extractor(model_name, device)

        progress(0.2, desc="Extracting entities...")

        STATE.entity_results = []

        for page_idx, search_result in enumerate(STATE.search_results):
            # Build text from enhanced results
            items = search_result.page_result.data
            text_parts = []
            for item in items:
                text = item.transformer_text or item.text
                text_parts.append(text)

            if not text_parts:
                STATE.entity_results.append(
                    {
                        "page": page_idx + 1,
                        "page_text": "",
                        "entities": {},
                    }
                )
                continue

            markdown_text = "\n".join(text_parts)

            # Extract entities for each selected entity type
            extracted = {}
            for entity_name in entity_types:
                if entity_name not in ENTITY_REGISTRY:
                    logger.warning(f"Unknown entity type: {entity_name}")
                    continue

                entity_model = cast(type[BaseModel], ENTITY_REGISTRY[entity_name])
                result = extractor.extract_entities(
                    markdown_text, entity_model=entity_model
                )
                content = result.get("content")
                if content and hasattr(content, "model_dump"):
                    extracted[entity_name] = content.model_dump()
                else:
                    extracted[entity_name] = {}

            STATE.entity_results.append(
                {
                    "page": page_idx + 1,
                    "page_text": markdown_text,
                    "entities": extracted,
                }
            )

            progress(0.2 + 0.8 * ((page_idx + 1) / len(STATE.search_results)))

        STATE.entity_complete = True
        progress(1.0, desc="Complete!")

        # Render results
        entity_markdown = render_entity_results(STATE.entity_results)

        # Collect all input text for display
        all_input_text = [
            f"--- Page {result['page']} ---\n{result['page_text']}"
            for result in STATE.entity_results
            if result.get("page_text")
        ]
        input_text_display = (
            "\n\n".join(all_input_text) if all_input_text else "No text extracted"
        )

        return (
            input_text_display,
            entity_markdown,
            f"✅ Entity extraction complete using {model_name}",
            create_status_html(STATE.to_summary()),
        )

    except Exception as e:
        logger.exception("Entity extraction failed")
        return (
            "",
            "",
            f"❌ Entity extraction failed: {e}",
            create_status_html(STATE.to_summary()),
        )


def change_page(
    direction: str,
) -> tuple[np.ndarray | None, list[list[str]], list[list[str]], str]:
    """
    Change the currently displayed page using prev/next navigation.

    Returns:
        Tuple of (image, ocr_table_data, search_table_data, page_info)

    """
    # Determine total pages from OCR results or preview images
    if STATE.ocr_results:
        total_pages = len(STATE.ocr_results)
    elif STATE.preview_images:
        total_pages = len(STATE.preview_images)
    else:
        return None, [], [], "Page 0 / 0"

    # Update current page based on direction
    if direction == "prev":
        STATE.current_page = max(0, STATE.current_page - 1)
    elif direction == "next":
        STATE.current_page = min(total_pages - 1, STATE.current_page + 1)

    # If we have OCR results, show annotated view
    if STATE.ocr_results:
        page_result = STATE.ocr_results[STATE.current_page]

        # Check if we have search results for highlighting
        highlight_indices = []
        if STATE.search_results and STATE.current_page < len(STATE.search_results):
            search_data = STATE.search_results[STATE.current_page].page_result.data
            highlight_indices = get_highlighted_indices(page_result.data, search_data)

        vis_image = None
        if page_result.original_image is not None:
            vis_image = draw_ocr_boxes(
                page_result.original_image,
                page_result.data,
                highlight_indices=highlight_indices,
            )

        # OCR table always shows ALL results for the page
        ocr_table_data = render_results_table(page_result.data)

        # Search table shows filtered results if available
        if STATE.search_results and STATE.current_page < len(STATE.search_results):
            search_table_data = render_results_table(
                STATE.search_results[STATE.current_page].page_result.data
            )
        else:
            search_table_data = []
    else:
        # Pre-OCR: just show preview image
        vis_image = (
            STATE.preview_images[STATE.current_page] if STATE.preview_images else None
        )
        ocr_table_data = []
        search_table_data = []

    page_info = f"Page {STATE.current_page + 1} / {total_pages}"

    return vis_image, ocr_table_data, search_table_data, page_info


def on_query_type_change(query_type: str) -> tuple:
    """Update visibility of query inputs based on query type."""
    if query_type == "semantic":
        return (
            gr.update(visible=True),  # semantic_query
            gr.update(visible=False),  # positional_row1
            gr.update(visible=False),  # positional_row2
            gr.update(visible=True),  # threshold
        )
    return (
        gr.update(visible=False),  # semantic_query
        gr.update(visible=True),  # positional_row1
        gr.update(visible=True),  # positional_row2
        gr.update(visible=False),  # threshold
    )


def compile_configuration() -> tuple[str, str]:
    """Compile the configuration from all the settings used during the session."""
    if not STATE.ocr_complete:
        return (
            "# No configuration to compile yet.\n# Run through the pipeline steps first.",
            "❌ Please run through the pipeline steps first to trial your configuration.",
        )

    # Build the configuration from the stored settings
    config_dict: dict[str, Any] = {}

    # OCR config (required)
    if STATE.ocr_config:
        config_dict["ocr"] = STATE.ocr_config

    # Transformer OCR config
    if STATE.enhancement_config:
        config_dict["transformer_ocr"] = STATE.enhancement_config

    # Search as queries
    if STATE.search_config:
        query_config = {
            "task": "Search query",
            "query_type": STATE.search_config.get("query_type", "semantic"),
        }
        if STATE.search_config.get("query_type") == "semantic":
            query_config["query_kwargs"] = {
                "text": STATE.search_config.get("semantic_query", ""),
                "threshold": STATE.search_config.get("threshold", 0.75),
                "search_type": "fuzzy",
                "search_padding": STATE.search_config.get("search_padding", 50),
            }
        else:
            query_config["query_kwargs"] = {
                "x1": STATE.search_config.get("positional_x1", 0),
                "y1": STATE.search_config.get("positional_y1", 0),
                "x2": STATE.search_config.get("positional_x2", 100),
                "y2": STATE.search_config.get("positional_y2", 100),
            }
        config_dict["queries"] = [query_config]

    # Entity extraction config
    if STATE.entity_config:
        config_dict["entity_extraction"] = STATE.entity_config

    config_yaml = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)

    return (
        config_yaml,
        "✅ Configuration compiled from your pipeline run. Copy the YAML above to save it.",
    )


def save_config_to_file(config_yaml: str) -> tuple[str, str | None]:
    """Save the compiled configuration to a file."""
    if not config_yaml.strip() or config_yaml.startswith("# No configuration"):
        return "❌ No valid configuration to save.", None

    try:
        # Validate the YAML
        yaml.safe_load(config_yaml)

        # Save to output directory
        output_dir = Path("data/output")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "compiled_config.yaml"
        with output_path.open("w") as f:
            f.write(config_yaml)

        logger.info(f"Saved configuration to {output_path}")
        return f"✅ Configuration saved to {output_path}", str(output_path)

    except Exception as e:
        logger.exception("Failed to save config")
        return f"❌ Failed to save configuration: {e}", None


def export_results() -> tuple[str, str | None]:
    """Export all results to JSON."""
    if not STATE.ocr_complete:
        return "❌ No results to export. Run the pipeline first.", None

    # Compile config for export
    config_dict = {}
    if STATE.ocr_config:
        config_dict["ocr"] = STATE.ocr_config
    if STATE.enhancement_config:
        config_dict["transformer_ocr"] = STATE.enhancement_config
    if STATE.search_config:
        config_dict["search"] = STATE.search_config
    if STATE.entity_config:
        config_dict["entity_extraction"] = STATE.entity_config

    export_data = {
        "document": STATE.document_name,
        "config": config_dict,
        "ocr_results": to_serialisable(STATE.ocr_results)
        if STATE.ocr_results
        else None,
        "search_results": (
            to_serialisable(STATE.search_results) if STATE.search_results else None
        ),
        "entity_results": STATE.entity_results,
    }

    # Save to output directory
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = (
        STATE.document_name.replace(" ", "_").rsplit(".", 1)[0]
        if STATE.document_name
        else "results"
    )
    output_path = output_dir / f"{safe_name}_results.json"

    with output_path.open("w") as f:
        json.dump(export_data, f, indent=2, default=str)

    logger.info(f"Exported results to {output_path}")
    return f"✅ Exported to {output_path}", str(output_path)


# ============================================================================
# Build the UI
# ============================================================================


def create_app() -> gr.Blocks:
    """Create the Gradio application."""
    # Initialize state without loading config
    STATE.config = None

    with gr.Blocks(title="NER-OCR Workbench") as app:
        # Header with status
        gr.Markdown("# 🔍 NER-OCR Workbench")

        with gr.Row():
            gr.HTML(value=create_instructions_html())

        status_html = gr.HTML(value=create_status_html(STATE.to_summary()))

        with gr.Row():
            # Left column: Controls
            with gr.Column(scale=1):
                # Document upload
                gr.Markdown("### 📄 Document")
                file_input = gr.File(
                    label="Upload PDF or Image",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif"],
                )
                doc_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    max_lines=1,
                    elem_classes=["dark-textbox"],
                )

            # Right column: Visualisation
            with gr.Column(scale=2):
                image_display = gr.Image(
                    label="Document View (Blue: all regions, Green: search matches)",
                    type="numpy",
                    height=500,
                )
                # Page navigation under the image
                with gr.Row():
                    prev_btn = gr.Button("◀ Prev", size="sm", scale=1)
                    page_info = gr.Textbox(
                        value="Page 0 / 0",
                        label="",
                        interactive=False,
                        max_lines=1,
                        scale=2,
                        container=False,
                        elem_classes=["page-info"],
                    )
                    next_btn = gr.Button("Next ▶", size="sm", scale=1)

        # Tabs for each pipeline stage
        with gr.Tabs():
            # OCR Tab with inline config
            with gr.TabItem("1️⃣ OCR"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### OCR Configuration")
                        ocr_max_side = gr.Slider(
                            minimum=500,
                            maximum=4000,
                            value=1500,
                            step=100,
                            label="Max Side Limit (px)",
                            info="Maximum image dimension",
                        )
                        ocr_timeout = gr.Slider(
                            minimum=60,
                            maximum=600,
                            value=400,
                            step=10,
                            label="OCR Timeout (seconds)",
                        )
                        with gr.Row():
                            ocr_orientation = gr.Checkbox(
                                label="Doc Orientation",
                                value=False,
                                info="Auto-rotate",
                            )
                            ocr_unwarping = gr.Checkbox(
                                label="Doc Unwarping",
                                value=False,
                                info="Fix distortion",
                            )
                        with gr.Row():
                            ocr_textline_orient = gr.Checkbox(
                                label="Textline Orientation",
                                value=False,
                            )
                            ocr_word_box = gr.Checkbox(
                                label="Word-level Boxes",
                                value=True,
                            )
                        ocr_device = gr.Radio(
                            choices=["cpu", "cuda", "mps"],
                            value="cpu",
                            label="Device",
                        )

                        gr.Markdown("---")
                        run_ocr_btn = gr.Button(
                            "▶️ Run OCR", variant="primary", size="lg"
                        )
                        ocr_status = gr.Textbox(
                            label="Status", interactive=False, max_lines=2
                        )

                    with gr.Column(scale=2):
                        ocr_table = gr.Dataframe(
                            headers=["#", "Text", "Enhanced", "Score", "Enh. Score"],
                            label="OCR Results",
                            wrap=True,
                        )

            # Search Tab
            with gr.TabItem("2️⃣ Search"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Search Configuration")
                        query_type = gr.Radio(
                            choices=["semantic", "positional"],
                            value="semantic",
                            label="Query Type",
                        )

                        # Semantic query inputs
                        semantic_query = gr.Textbox(
                            label="Search Text",
                            placeholder="Enter text to search for...",
                            visible=True,
                        )
                        threshold = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.75,
                            step=0.05,
                            label="Similarity Threshold",
                            visible=True,
                        )

                        # Positional query inputs (rectangle selection)
                        with gr.Row(visible=False) as positional_row1:
                            positional_x1 = gr.Number(
                                label="X1 - Left (pixels)",
                                value=100,
                            )
                            positional_y1 = gr.Number(
                                label="Y1 - Top (pixels)",
                                value=100,
                            )
                        with gr.Row(visible=False) as positional_row2:
                            positional_x2 = gr.Number(
                                label="X2 - Right (pixels)",
                                value=500,
                            )
                            positional_y2 = gr.Number(
                                label="Y2 - Bottom (pixels)",
                                value=300,
                            )

                        # Common
                        search_padding = gr.Slider(
                            minimum=0,
                            maximum=300,
                            value=50,
                            step=10,
                            label="Search Padding (pixels)",
                        )

                        gr.Markdown("---")
                        run_search_btn = gr.Button(
                            "▶️ Run Search", variant="primary", size="lg"
                        )
                        search_status = gr.Textbox(
                            label="Status", interactive=False, max_lines=2
                        )

                    with gr.Column(scale=2):
                        search_table = gr.Dataframe(
                            headers=["#", "Text", "Enhanced", "Score", "Enh. Score"],
                            label="Search Results",
                            wrap=True,
                        )

            # Enhancement Tab with inline config
            with gr.TabItem("3️⃣ Enhancement"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### TrOCR Configuration")
                        enhance_model = gr.Dropdown(
                            choices=TROCR_MODEL_CHOICES,
                            value="LARGE_HANDWRITTEN",
                            label="TrOCR Model",
                            info="Select model for text enhancement",
                        )
                        enhance_device = gr.Radio(
                            choices=["cpu", "cuda", "mps"],
                            value="cpu",
                            label="Device",
                        )

                        gr.Markdown("---")
                        run_enhance_btn = gr.Button(
                            "▶️ Run Enhancement", variant="primary", size="lg"
                        )
                        enhance_status = gr.Textbox(
                            label="Status", interactive=False, max_lines=2
                        )

                    with gr.Column(scale=2):
                        enhance_table = gr.Dataframe(
                            headers=[
                                "#",
                                "Original",
                                "Enhanced",
                                "Orig Score",
                                "Enh Score",
                            ],
                            label="Enhancement Results",
                            wrap=True,
                        )

            # Entity Extraction Tab with inline config
            with gr.TabItem("4️⃣ Entity Extraction"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Qwen Configuration")
                        entity_model = gr.Dropdown(
                            choices=QWEN_MODEL_CHOICES,
                            value="QWEN3_1_7B",
                            label="Qwen Model",
                            info="Select model for entity extraction",
                        )
                        entity_device = gr.Radio(
                            choices=["cpu", "cuda", "mps"],
                            value="cpu",
                            label="Device",
                        )
                        entity_types = gr.CheckboxGroup(
                            choices=ENTITY_CHOICES,
                            value=["AddressEntityList"],
                            label="Entity Types to Extract",
                        )

                        gr.Markdown("---")
                        run_entity_btn = gr.Button(
                            "▶️ Run Extraction", variant="primary", size="lg"
                        )
                        entity_status = gr.Textbox(
                            label="Status", interactive=False, max_lines=2
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("### Input Text (sent to model)")
                        entity_input_text = gr.Textbox(
                            label="Input Text",
                            value="*Run extraction to see input text*",
                            interactive=False,
                            lines=10,
                            max_lines=15,
                        )
                        gr.Markdown("### Extracted Entities")
                        entity_output = gr.Markdown(
                            value="*Run extraction to see results*"
                        )

            # Compile Configuration Tab
            with gr.TabItem("⚙️ Compile Config"):
                gr.Markdown("""
                ### Compile Your Configuration

                Once you're happy with the pipeline settings you've tested, click the button below
                to generate a `config.yaml` file based on your selections.

                You can then save this configuration and use it for batch processing.
                """)

                with gr.Row():
                    compile_btn = gr.Button(
                        "🔧 Compile Configuration", variant="primary", size="lg"
                    )

                config_output = gr.Code(
                    label="Compiled Configuration (YAML)",
                    language="yaml",
                    value="# Run through the pipeline steps first, then compile your configuration.",
                    lines=25,
                )
                compile_status = gr.Textbox(
                    label="Status", interactive=False, max_lines=2
                )

                with gr.Row():
                    save_config_btn = gr.Button(
                        "💾 Save Config to File", variant="secondary"
                    )
                    config_file_output = gr.File(label="Download", visible=False)

            # Export Tab
            with gr.TabItem("💾 Export Results"):
                gr.Markdown("### Export Results")
                export_btn = gr.Button(
                    "📥 Export to JSON", variant="primary", size="lg"
                )
                export_status = gr.Textbox(
                    label="Status", interactive=False, max_lines=1
                )
                export_file = gr.File(label="Download", visible=False)

        # ====================================================================
        # Wire up events
        # ====================================================================

        # Document upload - shows preview
        file_input.upload(
            fn=load_document,
            inputs=[file_input],
            outputs=[doc_status, image_display, status_html, page_info],
        )

        # OCR with inline config
        run_ocr_btn.click(
            fn=run_ocr,
            inputs=[
                ocr_max_side,
                ocr_timeout,
                ocr_orientation,
                ocr_unwarping,
                ocr_textline_orient,
                ocr_word_box,
                ocr_device,
            ],
            outputs=[image_display, ocr_table, ocr_status, status_html, page_info],
        )

        # Query type change - update visible inputs
        query_type.change(
            fn=on_query_type_change,
            inputs=[query_type],
            outputs=[semantic_query, positional_row1, positional_row2, threshold],
        )

        # Search
        run_search_btn.click(
            fn=run_search,
            inputs=[
                query_type,
                semantic_query,
                positional_x1,
                positional_y1,
                positional_x2,
                positional_y2,
                threshold,
                search_padding,
            ],
            outputs=[image_display, search_table, search_status, status_html],
        )

        # Enhancement - with model and device selection
        run_enhance_btn.click(
            fn=run_enhancement,
            inputs=[enhance_model, enhance_device],
            outputs=[enhance_table, enhance_status, status_html],
        )

        # Entity extraction - with model, device, and entity type selection
        run_entity_btn.click(
            fn=run_entity_extraction,
            inputs=[entity_model, entity_device, entity_types],
            outputs=[entity_input_text, entity_output, entity_status, status_html],
        )

        # Page navigation with buttons - updates both OCR and Search tables
        prev_btn.click(
            fn=lambda: change_page("prev"),
            inputs=[],
            outputs=[image_display, ocr_table, search_table, page_info],
        )
        next_btn.click(
            fn=lambda: change_page("next"),
            inputs=[],
            outputs=[image_display, ocr_table, search_table, page_info],
        )

        # Compile configuration
        compile_btn.click(
            fn=compile_configuration,
            inputs=[],
            outputs=[config_output, compile_status],
        )

        # Save config to file
        save_config_btn.click(
            fn=save_config_to_file,
            inputs=[config_output],
            outputs=[compile_status, config_file_output],
        )

        # Export
        export_btn.click(
            fn=export_results,
            inputs=[],
            outputs=[export_status, export_file],
        )

    return app


def launch_workbench(
    host: str = "0.0.0.0",
    port: int = 7860,
    *,
    share: bool = False,
    auth: tuple[str, str] | None = None,
) -> None:
    """Launch the workbench UI."""
    logger.info(f"Starting NER-OCR Workbench on http://{host}:{port}")

    # Custom CSS for dark theme consistency
    custom_css = """
    .dark-textbox input, .dark-textbox textarea {
        background-color: #374151 !important;
        color: #e5e7eb !important;
        border-color: #4b5563 !important;
    }
    .page-info input {
        text-align: center !important;
        background-color: #374151 !important;
        color: #e5e7eb !important;
        border: none !important;
    }
    """

    app = create_app()
    app.launch(
        server_name=host,
        server_port=port,
        share=share,
        auth=auth,
        show_error=True,
        quiet=False,
        css=custom_css,
    )


if __name__ == "__main__":
    launch_workbench()
