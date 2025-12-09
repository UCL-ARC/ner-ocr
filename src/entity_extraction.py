"""LLM based entity extraction."""

import gc
import time
from enum import Enum

import torch
from langchain_core.output_parsers import PydanticOutputParser
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from .entities import AddressEntity
from .exceptions import TransformerError
from .types.base import EntityExtractor
from .types.data import T


class QwenModels(Enum):
    """Available Qwen models with their Hugging Face identifiers."""

    QWEN3_8B = "Qwen/Qwen3-8B"
    QWEN3_4B_INSTRUCT_2507 = "Qwen/Qwen3-4B-Instruct-2507"
    QWEN3_1_7B = "Qwen/Qwen3-1.7B"


class QwenEntityExtractor(EntityExtractor):
    """Entity extractor using Qwen LLM model."""

    def __init__(
        self,
        model: QwenModels = QwenModels.QWEN3_1_7B,
        cache_dir: str | None = None,
        device: str | None = None,
        *,
        local: bool = False,
    ) -> None:
        """
        Initialise the Qwen model and tokenizer.

        Args:
            model: Name of the pre-trained Qwen model.
            cache_dir: Optional directory path for caching model files.

        """
        if not isinstance(model, QwenModels):
            available_models = [m.name for m in QwenModels]
            error_msg = (
                f"Invalid model type {model}. Must be one of: {available_models}. "
            )
            raise TransformerError(error_msg)

        self.model_name = model.value
        self.cache_dir = cache_dir
        self._is_loaded = False
        self.device = device
        self.local_files_only = local

        try:
            logger.info(f"loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                device_map=self.device or "auto",
                dtype="auto",
                local_files_only=self.local_files_only,
            )  # Note we have device map as auto here
            self._is_loaded = True
        except Exception as e:
            error_msg = f"Error loading model or tokenizer: {e}"
            raise TransformerError(error_msg) from e

    def parse_output(
        self, output: str, parser: PydanticOutputParser | None = None
    ) -> dict:
        """Parse the model output using the provided parser + model specific tokens."""
        cot_substring = "</think>"
        logger.info("parsing output")
        if cot_substring in output:
            output = output.split(cot_substring)[-1]

        eos_string = "<|im_end|>"
        if output.endswith(eos_string):
            output = output[: -len(eos_string)]

        if parser is None:
            return {"content": output.strip()}

        try:
            output = parser.parse(output)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error parsing output: {e}")
        output = parser.pydantic_object()  # Return empty model on parse failure

        return {"content": output}

    def run_inference(self, messages: list[dict[str, str]]) -> str:
        """Run inference on the model with the provided messages."""
        logger.info("running inference...")
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
            return_dict=True,
        ).to(self.model.device)
        outputs = self.model.generate(
            **inputs, max_new_tokens=10000
        )  # TO DO: add control for this
        return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :])

    def extract_entities(
        self, text: str, entity_model: type[T], kwargs: dict | None = None
    ) -> dict[str, T]:
        """Extract entities from the provided text."""
        logger.info("extracting entities...")
        if kwargs is None:
            kwargs = {}

        parser = PydanticOutputParser(pydantic_object=entity_model)

        format_instructions = parser.get_format_instructions()

        base_prompt = kwargs.get(
            "system_prompt",
            "Extract the following entities (make sure you extract ALL entities) from the text and return them in the specified format:",
        )
        system_prompt = f"{base_prompt}\n{format_instructions}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]

        output = self.run_inference(messages)
        return self.parse_output(output, parser)

    def offload_model(self) -> None:
        """Offload the model from memory to free up resources."""
        if not self._is_loaded:
            logger.warning("Model is not loaded; cannot offload.")
            return
        try:
            del self.model
            del self.tokenizer
            self._is_loaded = False

            gc.collect()

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for GPU operations to complete
                logger.debug("Cleared CUDA cache")

            # Clear MPS cache if available (Apple Silicon)
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
                logger.debug("Cleared MPS cache")

            time.sleep(20)
            logger.info("Model offloaded from memory successfully.")
        except Exception as e:
            error_msg = f"Error offloading model: {e}"
            raise TransformerError(error_msg) from e

    def __del__(self) -> None:
        """Best-effort cleanup when object is destroyed."""
        try:
            if getattr(self, "_is_loaded", False):
                logger.info(
                    "Object being destroyed - attempting to clean up model resources..."
                )
                # Best-effort; ignore any errors
                self.offload_model()
        except Exception as e:  # noqa: BLE001
            # Never let exceptions escape __del__
            error_msg = f"Error during cleanup in __del__: {e}"
            logger.error(error_msg)


if __name__ == "__main__":
    for model in QwenModels:
        logger.info(f"Testing model: {model.name}")
        extractor = QwenEntityExtractor(model, cache_dir="../models")
        text = "37 sunlight square, london uk, wc2n 4hh"
        result = extractor.extract_entities(text, AddressEntity)
        logger.info(f"Extracted Entities: {result}")
        extractor.offload_model()
