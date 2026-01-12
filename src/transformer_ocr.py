"""Transformer-based OCR module using Microsoft's TrOCR model."""

from enum import Enum

import numpy as np
import torch
import torch.nn.functional as f
from loguru import logger
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from .custom_types import BaseTransformerOCR, TransformerResult
from .exceptions import TransformerError


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
        *,
        use_fp16: bool = True,
        local: bool = False,
    ) -> None:
        """
        Initialise the TrOCR processor and model.

        Args:
            model: Name of the pre-trained TrOCR model.
            cache_dir: Optional directory path for caching model files.
            device: Device to run the model on ('cpu', 'cuda', or 'mps').
            use_fp16: Use half-precision (fp16) for faster loading and inference on GPU.
                      Automatically disabled for CPU. Default is True.
            local: If True, only load from local cache (no network calls).
                   Use for network-isolated environments like TREs.

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
        self.local_files_only = local

        # Determine dtype for GPU acceleration while maintaining accuracy
        # BF16 preferred: same dynamic range as FP32 (better accuracy), but half the memory
        # FP16 fallback: faster but slightly lower numerical precision
        self.dtype = self._get_optimal_dtype(use_reduced_precision=use_fp16)

        logger.info(f"Loading TrOCR model '{self.model_name}'...")
        logger.info(f"Cache directory: {self.cache_dir}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Using dtype: {self.dtype}")

        try:
            # Load processor (for image preprocessing + text decoding)
            self.processor = TrOCRProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                use_fast=True,
                local_files_only=self.local_files_only,
            )

            # Load model with optimizations for faster CUDA loading:
            # - torch_dtype: Use fp16 on GPU (2x smaller, faster transfer)
            # - low_cpu_mem_usage: Avoid creating full CPU copy before GPU transfer
            # - device_map: Load directly to target device when using CUDA
            if self.device == "cuda":
                logger.info(
                    "Using optimized CUDA loading (fp16, direct device mapping)"
                )
                self.model = VisionEncoderDecoderModel.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    torch_dtype=self.dtype,
                    low_cpu_mem_usage=True,
                    device_map="auto",
                    local_files_only=self.local_files_only,
                )
            else:
                # For CPU/MPS, load normally then move to device
                self.model = VisionEncoderDecoderModel.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    torch_dtype=self.dtype,
                    low_cpu_mem_usage=True,
                    local_files_only=self.local_files_only,
                )
                self.model.to(self.device)

            self.model.eval()

            logger.info(f"TrOCR model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Error loading TrOCR model: {e}")
            raise

    # -------------------------------------------------------------------------
    # Utility functions
    # -------------------------------------------------------------------------

    def _get_optimal_dtype(self, *, use_reduced_precision: bool) -> torch.dtype:
        """
        Determine the optimal dtype for the model.

        Prefers BF16 over FP16 for better numerical stability (same dynamic range as FP32).
        Falls back to FP32 on CPU or when reduced precision is disabled.

        Args:
            use_reduced_precision: Whether to use reduced precision (bf16/fp16) on GPU.

        Returns:
            The optimal torch dtype for the current device.

        """
        if not use_reduced_precision or self.device == "cpu":
            return torch.float32

        # Check for BF16 support (preferred for accuracy)
        if self.device == "cuda" and torch.cuda.is_bf16_supported():
            logger.info("Using BF16 for optimal speed/accuracy trade-off (Ampere+ GPU)")
            return torch.bfloat16
        if self.device == "mps":
            # MPS supports BF16 on Apple Silicon
            logger.info("Using BF16 on Apple Silicon MPS")
            return torch.bfloat16
        if self.device == "cuda":
            # Older CUDA GPUs: fall back to FP16
            logger.info("Using FP16 (BF16 not supported on this GPU)")
            return torch.float16

        return torch.float32

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
