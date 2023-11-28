"""
LLM Fine-Tuning Pipeline

A complete end-to-end pipeline for fine-tuning large language models
using PEFT/LoRA with comprehensive evaluation against baselines.

Modules:
    - data_preparation: Dataset loading and preprocessing
    - model_config: Model loading and PEFT configuration
    - training_pipeline: Training loop with Hugging Face Trainer
    - evaluation: Model evaluation and baseline comparison
"""

from .data_preparation import DataPreparer, create_instruction_dataset
from .evaluation import BaselineComparer, EvaluationConfig, ModelEvaluator
from .model_config import ModelConfig, ModelManager, create_model_manager
from .training_pipeline import FineTuningTrainer, TrainingConfig

__version__ = "1.0.0"
__all__ = [
    "DataPreparer",
    "create_instruction_dataset",
    "ModelConfig",
    "ModelManager",
    "create_model_manager",
    "TrainingConfig",
    "FineTuningTrainer",
    "EvaluationConfig",
    "ModelEvaluator",
    "BaselineComparer",
]
