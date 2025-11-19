"""LLM based entity extraction."""

import gc
from enum import Enum

from langchain_core.output_parsers import PydanticOutputParser
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from .custom_types import EntityExtractor, T
from .entities import AddressEntity
from .exceptions import TransformerError


class QwenModels(Enum):
    """Available Qwen models with their Hugging Face identifiers."""

    # QWEN3_8B = "Qwen/Qwen3-8B"
    QWEN3_1_7B = "Qwen/Qwen3-1.7B"
    QWEN3_4B_INSTRUCT_2507 = "Qwen/Qwen3-4B-Instruct-2507"


class QwenEntityExtractor(EntityExtractor):
    """Entity extractor using Qwen LLM model."""

    def __init__(
        self,
        model: QwenModels = QwenModels.QWEN3_1_7B,
        cache_dir: str | None = None,
        device: int | None = None,
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

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, cache_dir=self.cache_dir
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, cache_dir=self.cache_dir, device_map="auto"
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
        logger.info(f"parsing output: {output}")
        if cot_substring in output:
            output = output.split(cot_substring)[-1]

        eos_string = "<|im_end|>"
        if output.endswith(eos_string):
            output = output[: -len(eos_string)]

        output = parser.parse(output) if parser else output.strip()

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
            **inputs, max_new_tokens=512
        )  # TO DO: add control for this
        return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :])

    def extract_entities(
        self, text: str, entity_model: type[T], kwargs: dict | None = None
    ) -> dict[str, T]:
        """Extract entities from the provided text."""
        if kwargs is None:
            kwargs = {}

        parser = PydanticOutputParser(pydantic_object=entity_model)

        format_instructions = parser.get_format_instructions()

        base_prompt = kwargs.get(
            "system_prompt",
            "Extract the following entities from the text and return them in the specified format:",
        )
        system_prompt = f"{base_prompt}\n{format_instructions}"
        logger.info(f"System prompt: {system_prompt}")
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
            logger.info("Model offloaded from memory successfully.")
        except Exception as e:
            error_msg = f"Error offloading model: {e}"
            raise TransformerError(error_msg) from e

    def __del__(self) -> None:
        """Cleanup when object is destroyed."""
        if self._is_loaded:
            logger.info("Object being destroyed - cleaning up model resources...")
            self.offload_model()


if __name__ == "__main__":
    for model in QwenModels:
        logger.info(f"Testing model: {model.name}")
        extractor = QwenEntityExtractor(model, cache_dir="../models")
        text = "37 sunlight square, london uk, wc2n 4hh"
        result = extractor.extract_entities(text, AddressEntity)
        logger.info("Extracted Entities:", result)
        extractor.offload_model()
