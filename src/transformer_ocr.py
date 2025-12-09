"""Transformer-based OCR module using Microsoft's TrOCR model."""

from enum import Enum

import numpy as np
import torch
import torch.nn.functional as f
from loguru import logger
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from .exceptions import TransformerError
from .types.base import BaseTransformerOCR
from .types.data import TransformerResult


class TrOCRModels(Enum):
    """Available TrOCR models with their Hugging Face identifiers."""

    # Base models
    BASE_HANDWRITTEN = "microsoft/trocr-base-handwritten"
    BASE_PRINTED = "microsoft/trocr-base-printed"
    BASE_STR = "microsoft/trocr-base-str"  # Scene text recognition

    # Large models (better accuracy, slower inference)
    LARGE_HANDWRITTEN = "microsoft/trocr-large-handwritten"
    LARGE_PRINTED = "microsoft/trocr-large-printed"
    LARGE_STR = "microsoft/trocr-large-str"


class TrOCRWrapper(BaseTransformerOCR):
    """
    Wrapper around Microsoft's TrOCR model for handwritten or printed text recognition.
    Provides image preprocessing, inference, and confidence scoring.
    """

    def __init__(
        self,
        model: TrOCRModels = TrOCRModels.BASE_HANDWRITTEN,
        cache_dir: str | None = None,
        device: str | None = None,
    ) -> None:
        """
        Initialise the TrOCR processor and model.

        Args:
            model: Name of the pre-trained TrOCR model.
            cache_dir: Optional directory path for caching model files.
            device: Device to run the model on ('cpu', 'cuda', or 'mps').

        """
        if not isinstance(model, TrOCRModels):
            available_models = [m.name for m in TrOCRModels]
            error_msg = (
                f"Invalid model type {model}. Must be one of: {available_models}. "
            )
            raise TransformerError(error_msg)

        self.model_name = model.value
        self.cache_dir = cache_dir
        self.device = self._get_device(device)

        logger.info(f"Loading TrOCR model '{self.model_name}'...")
        logger.info(f"Cache directory: {self.cache_dir}")
        logger.info(f"Device: {self.device}")

        try:
            # Load processor (for image preprocessing + text decoding)
            self.processor = TrOCRProcessor.from_pretrained(
                self.model_name, cache_dir=self.cache_dir, use_fast=True
            )
            # Load model (encoder + decoder)
            self.model = VisionEncoderDecoderModel.from_pretrained(
                self.model_name, cache_dir=self.cache_dir
            )

            # Move model to the selected device and set to eval mode
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"TrOCR model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Error loading TrOCR model: {e}")
            raise

    # -------------------------------------------------------------------------
    # Utility functions
    # -------------------------------------------------------------------------

    def _get_device(self, device: str | None) -> str:
        """Determine the best available compute device for inference."""
        if device is not None:
            # Respect user-specified device if valid
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but unavailable. Falling back to CPU.")
                return "cpu"
            if device == "mps" and not torch.backends.mps.is_available():
                logger.warning("MPS requested but unavailable. Falling back to CPU.")
                return "cpu"
            return device

        # Auto-detect if no device was provided
        if torch.backends.mps.is_available():
            logger.info("Apple Silicon detected. Using MPS for GPU acceleration.")
            return "mps"
        if torch.cuda.is_available():
            logger.info("CUDA available. Using GPU for acceleration.")
            return "cuda"
        logger.info("No GPU acceleration available. Using CPU.")
        return "cpu"

    # -------------------------------------------------------------------------
    # Main prediction method
    # -------------------------------------------------------------------------

    def predict(self, image: np.ndarray) -> TransformerResult:
        """
        Perform OCR on the input image.

        Args:
            image: Input image as a NumPy array (RGB or BGR format).

        Returns:
            Recognized text as a string.

        """
        try:
            # -----------------------------------------------------------------
            # Step 1: Preprocess image (convert to RGB PIL Image)
            # -----------------------------------------------------------------
            if len(image.shape) == 3 and image.shape[2] == 4:  # noqa: PLR2004
                # RGBA → RGB
                image_rgb = image[:, :, :3]
            # TO DO: confirm if BGR to RGB conversion is needed
            else:
                image_rgb = image  # Grayscale or already RGB

            pil_image = Image.fromarray(image_rgb).convert("RGB")

            # -----------------------------------------------------------------
            # Step 2: Prepare inputs for the model
            # -----------------------------------------------------------------
            pixel_values = self.processor(
                images=pil_image, return_tensors="pt"
            ).pixel_values
            pixel_values = pixel_values.to(self.device)

            # -----------------------------------------------------------------
            # Step 3: Run the model to generate text and per-token logits
            # -----------------------------------------------------------------
            with torch.no_grad():
                return_dict = self.model.generate(
                    pixel_values,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            # -----------------------------------------------------------------
            # Step 4: Compute per-token probabilities
            # -----------------------------------------------------------------
            generated_ids = return_dict["sequences"][0][1:]  # skip <BOS> token
            logits_list = return_dict["scores"]

            token_probs = []
            for step, logits in enumerate(logits_list):
                probs = f.softmax(logits, dim=-1)  # Convert logits → probabilities
                token_id = generated_ids[step]  # Actual generated token ID at this step
                token_probs.append(
                    probs[0, token_id].item()
                )  # Extract scalar probability

            # -----------------------------------------------------------------
            # Step 5: Aggregate token probabilities → overall sequence confidence
            # -----------------------------------------------------------------
            if len(token_probs) > 0:
                probs_tensor = torch.tensor(token_probs)
                log_mean = torch.mean(torch.log(probs_tensor))  # Mean log-probability
                sequence_conf_geom = torch.exp(
                    log_mean
                ).item()  # Geometric mean (preferred)
            else:
                sequence_conf_geom = 0.0

            # -----------------------------------------------------------------
            # Step 6: Decode token IDs into readable text
            # -----------------------------------------------------------------
            generated_text = self.processor.batch_decode(
                return_dict["sequences"], skip_special_tokens=True
            )[0]

            # -----------------------------------------------------------------
            # Step 7: Log useful information
            # -----------------------------------------------------------------
            tokens = self.processor.tokenizer.convert_ids_to_tokens(generated_ids)
            logger.info(f"Tokens: {tokens}")
            logger.info(f"Token probabilities: {token_probs}")
            logger.info(f"Recognized text: '{generated_text}'")
            logger.info(f"Confidence (geometric mean): {sequence_conf_geom:.4f}")

            # -----------------------------------------------------------------
            # Step 8: Return recognized text
            # -----------------------------------------------------------------
            return TransformerResult(
                transformer_text=generated_text, score=sequence_conf_geom
            )

        except Exception as e:  # noqa: BLE001
            logger.error(f"Error during TrOCR prediction: {e}")
            return TransformerResult(transformer_text="", score=0.0)
