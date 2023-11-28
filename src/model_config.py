"""
Model Configuration Module for LLM Fine-Tuning Pipeline

This module handles:
- Loading pre-trained language models
- Configuring PEFT methods (LoRA, QLoRA)
- Setting up tokenizers with proper configurations
- Model inspection and parameter counting
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)


class PeftMethod(Enum):
    """Supported PEFT methods."""
    LORA = "lora"
    QLORA = "qlora"
    NONE = "none"


@dataclass
class ModelConfig:
    """
    Configuration for model loading and PEFT setup.

    Attributes:
        model_name: Hugging Face model name or path
        peft_method: PEFT method to use
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        lora_dropout: LoRA dropout probability
        target_modules: Modules to apply LoRA to
        bits: Quantization bits for QLoRA (4 or 8)
        device_map: Device mapping strategy
        torch_dtype: Torch data type
    """
    model_name: str = "gpt2"
    peft_method: str = "lora"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None
    bits: Optional[int] = None
    device_map: str = "auto"
    torch_dtype: str = "float32"
    use_cache: bool = False

    def __post_init__(self):
        """Validate and normalize configuration."""
        if isinstance(self.peft_method, str):
            self.peft_method = self.peft_method.lower()
        if isinstance(self.torch_dtype, str):
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "auto": "auto",
            }
            self.torch_dtype = dtype_map.get(self.torch_dtype, torch.float32)


class ModelManager:
    """
    Manages model loading, configuration, and PEFT setup.

    This class provides a unified interface for:
    - Loading pre-trained models with various configurations
    - Applying PEFT methods (LoRA, QLoRA)
    - Managing tokenizers
    - Inspecting model parameters
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize the ModelManager.

        Args:
            config: ModelConfig instance with model settings
        """
        self.config = config
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.peft_config: Optional[LoraConfig] = None
        self.is_peft_model = False

    def load_model(self) -> PreTrainedModel:
        """
        Load the pre-trained model with optional quantization.

        Returns:
            Loaded model
        """
        logger.info(f"Loading model: {self.config.model_name}")

        # Prepare model loading kwargs
        kwargs = {
            "device_map": self.config.device_map,
            "torch_dtype": self.config.torch_dtype,
        }

        # Add quantization for QLoRA
        if self.config.peft_method == PeftMethod.QLORA.value and self.config.bits:
            kwargs["load_in_4bit"] = self.config.bits == 4
            kwargs["load_in_8bit"] = self.config.bits == 8
            logger.info(f"Loading with {self.config.bits}-bit quantization")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **kwargs
        )

        # Configure model for training
        self.model.config.use_cache = self.config.use_cache

        # Ensure model has proper pad token
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.model.config.eos_token_id

        logger.info(f"Model loaded successfully. Parameters: {self.count_parameters():,}")
        return self.model

    def load_tokenizer(self) -> PreTrainedTokenizer:
        """
        Load the tokenizer for the model.

        Returns:
            Loaded tokenizer
        """
        logger.info(f"Loading tokenizer: {self.config.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            padding_side="right",
            use_fast=True,
        )

        # Ensure special tokens are set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token: {self.tokenizer.eos_token}")

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        return self.tokenizer

    def setup_peft(self) -> PeftModel:
        """
        Apply PEFT configuration to the model.

        Returns:
            PEFT-wrapped model
        """
        if self.model is None:
            raise ValueError("Model must be loaded before setting up PEFT")

        if self.config.peft_method == PeftMethod.NONE.value:
            logger.info("No PEFT method selected, using full fine-tuning")
            return self.model

        logger.info(f"Setting up PEFT with method: {self.config.peft_method}")

        # Determine target modules based on model architecture
        target_modules = self._get_target_modules()

        # Create LoRA configuration
        self.peft_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # Apply PEFT
        self.model = get_peft_model(self.model, self.peft_config)
        self.is_peft_model = True

        # Print trainable parameters
        self.model.print_trainable_parameters()

        return self.model

    def _get_target_modules(self) -> List[str]:
        """
        Determine target modules for LoRA based on model architecture.

        Returns:
            List of module names to apply LoRA to
        """
        if self.config.target_modules:
            return self.config.target_modules

        # Auto-detect based on model name
        model_name_lower = self.config.model_name.lower()

        if "gpt2" in model_name_lower:
            return ["c_attn", "c_proj"]
        elif "llama" in model_name_lower or "mistral" in model_name_lower:
            return ["q_proj", "v_proj", "k_proj", "o_proj"]
        elif "opt" in model_name_lower:
            return ["q_proj", "v_proj"]
        elif "pythia" in model_name_lower:
            return ["query_key_value", "dense"]
        else:
            # Default: try to find attention modules
            return ["q_proj", "v_proj", "query", "value"]

    def count_parameters(self, trainable_only: bool = False) -> int:
        """
        Count model parameters.

        Args:
            trainable_only: Whether to count only trainable parameters

        Returns:
            Number of parameters
        """
        if self.model is None:
            return 0

        if trainable_only:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.model.parameters())

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed model information.

        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"error": "Model not loaded"}

        total_params = self.count_parameters()
        trainable_params = self.count_parameters(trainable_only=True)

        info = {
            "model_name": self.config.model_name,
            "model_type": self.model.config.model_type,
            "vocab_size": self.model.config.vocab_size,
            "hidden_size": self.model.config.hidden_size,
            "num_layers": getattr(self.model.config, "num_hidden_layers",
                                  getattr(self.model.config, "n_layer", "N/A")),
            "num_attention_heads": getattr(self.model.config, "num_attention_heads",
                                           getattr(self.model.config, "n_head", "N/A")),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": (trainable_params / total_params * 100) if total_params > 0 else 0,
            "peft_method": self.config.peft_method,
            "is_peft_model": self.is_peft_model,
        }

        if self.peft_config:
            info.update({
                "lora_r": self.config.lora_r,
                "lora_alpha": self.config.lora_alpha,
                "lora_dropout": self.config.lora_dropout,
                "target_modules": self.peft_config.target_modules,
            })

        return info

    def prepare_for_training(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Prepare model and tokenizer for training.

        This is a convenience method that loads model, tokenizer,
        and sets up PEFT in one call.

        Returns:
            Tuple of (model, tokenizer)
        """
        self.load_model()
        self.load_tokenizer()
        self.setup_peft()

        return self.model, self.tokenizer

    def save_model(self, output_dir: str) -> None:
        """
        Save the fine-tuned model.

        Args:
            output_dir: Directory to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")

        logger.info(f"Saving model to {output_dir}")

        model_type = "PEFT adapter" if self.is_peft_model else "Full model"
        self.model.save_pretrained(output_dir)
        logger.info(f"{model_type} saved")

        if self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)
            logger.info("Tokenizer saved")

    def load_peft_adapter(self, adapter_path: str) -> None:
        """
        Load a PEFT adapter for inference.

        Args:
            adapter_path: Path to the saved PEFT adapter
        """
        if self.model is None:
            self.load_model()

        logger.info(f"Loading PEFT adapter from {adapter_path}")
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.is_peft_model = True
        logger.info("PEFT adapter loaded successfully")


def create_model_manager(
    model_name: str = "gpt2",
    peft_method: str = "lora",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    **kwargs,
) -> ModelManager:
    """
    Factory function to create a ModelManager with common configurations.

    Args:
        model_name: Hugging Face model name
        peft_method: PEFT method ('lora', 'qlora', or 'none')
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        **kwargs: Additional configuration options

    Returns:
        Configured ModelManager instance
    """
    config = ModelConfig(
        model_name=model_name,
        peft_method=peft_method,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        **kwargs
    )
    return ModelManager(config)


# Preset configurations for common use cases
PRESET_CONFIGS = {
    "gpt2_small": {
        "model_name": "gpt2",
        "peft_method": "lora",
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "max_length": 512,
    },
    "gpt2_medium": {
        "model_name": "gpt2-medium",
        "peft_method": "lora",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "max_length": 512,
    },
    "distilgpt2": {
        "model_name": "distilgpt2",
        "peft_method": "lora",
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "max_length": 512,
    },
    "full_finetune": {
        "model_name": "gpt2",
        "peft_method": "none",
        "max_length": 512,
    },
}


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create model manager with LoRA
    manager = create_model_manager(
        model_name="gpt2",
        peft_method="lora",
        lora_r=8,
        lora_alpha=16,
    )

    # Prepare for training
    model, tokenizer = manager.prepare_for_training()

    # Print model info
    info = manager.get_model_info()
    print("\n" + "=" * 50)
    print("Model Information")
    print("=" * 50)
    for key, value in info.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # Test generation
    print("\n" + "=" * 50)
    print("Test Generation")
    print("=" * 50)
    input_text = "The quick brown fox"
    inputs = tokenizer(input_text, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input: {input_text}")
    print(f"Generated: {generated_text}")
