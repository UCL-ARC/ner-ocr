"""OCR pipeline module."""

import os
from pathlib import Path
from time import time

from loguru import logger

from src.bounding_box import PaddleOCRWrapper
from src.config import AppConfig
from src.custom_types import PositionalQuery, SearchResult, SemanticQuery
from src.rpa import RPAProcessor
from src.transformer_ocr import TrOCRModels, TrOCRWrapper

from .base import BasePipeline


class OCRPipeline(BasePipeline):
    """Pipeline for OCR processing."""

    def __init__(self, config: AppConfig) -> None:
        """Initialise OCR pipeline with config."""
        super().__init__(config)
        # TO DO: this is limited to PaddleOCR + TrOCR for now
        self.ocr = PaddleOCRWrapper(
            max_side_limit=config.ocr.max_side_limit,
            ocr_timeout=config.ocr.ocr_timeout,
            use_doc_orientation_classify=config.ocr.use_doc_orientation_classify,
            use_doc_unwarping=config.ocr.use_doc_unwarping,
            use_textline_orientation=config.ocr.use_textline_orientation,
            return_word_box=config.ocr.return_word_box,
            device=config.ocr.device,
        )
        self.transformer = TrOCRWrapper(
            model=TrOCRModels[config.transformer_ocr.model],
            device=config.transformer_ocr.device,
            use_fp16=config.transformer_ocr.use_fp16,
            local=os.environ.get("HF_HUB_OFFLINE", "0") == "1",
        )

    def get_output_suffix(self) -> str:
        """Return the suffix for output files."""
        return "_ocr_results"

    def process_file(self, file_path: Path) -> list[SearchResult]:
        """Process a single file through OCR pipeline."""
        start_time = time()
        logger.info(f"Starting OCR pipeline for file: {file_path}")

        # Step 1: Run OCR
        ocr_results = self.ocr.extract(file_path)
        logger.info(
            f"OCR completed. Detected {sum(len(page.data) for page in ocr_results)} text regions."
        )

        # Step 2: Search and enhance
        searched_results = []
        for query in self.config.queries:
            logger.info(f"Processing query: {query['task']}")
            rpa, query_obj = self._build_query(query)

            for page in ocr_results:
                searched_page = rpa.search(page, query_obj, task=query.get("task"))
                searched_results.append(searched_page)

            # Step 3: Enhance with transformer OCR
            for searched_page in searched_results:
                for item in searched_page.page_result.data:
                    transformer_output = self.transformer.predict(item.bbox_image)
                    logger.info(
                        f"Original: '{item.text}' | Enhanced: '{transformer_output.transformer_text}'"
                    )
                    item.transformer_text = transformer_output.transformer_text
                    item.transformer_score = transformer_output.score

        elapsed = time() - start_time
        logger.info(f"OCR pipeline completed in {elapsed:.2f}s")
        return searched_results

    def _build_query(
        self, query: dict
    ) -> tuple[RPAProcessor, SemanticQuery | PositionalQuery]:
        """Build RPA processor and query object from config."""
        query_type = query["query_type"]
        query_kwargs = query.get("query_kwargs", {})

        query_obj: SemanticQuery | PositionalQuery

        if query_type == "semantic":
            rpa = RPAProcessor(
                search_type="semantic",
                search_kwargs=query.get("search_kwargs", {}),
                verbose=True,
            )
            query_obj = SemanticQuery(
                text=query_kwargs["text"],
                threshold=query_kwargs.get("threshold", 0.8),
                search_type=query_kwargs.get("search_type", "fuzzy"),
                search_padding=query_kwargs.get("search_padding", 50.0),
            )
        elif query_type == "positional":
            rpa = RPAProcessor(
                search_type="positional",
                search_kwargs=query.get("search_kwargs", {}),
                verbose=True,
            )
            query_obj = PositionalQuery(
                x1=query_kwargs["x1"],
                y1=query_kwargs["y1"],
                x2=query_kwargs["x2"],
                y2=query_kwargs["y2"],
            )
        else:
            error_msg = f"Unknown query type: {query_type}"
            raise ValueError(error_msg)

        return rpa, query_obj
