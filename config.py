"""
Configuration file for LLM Fine-Tuning Pipeline

Modify these settings to customize your fine-tuning run.
All paths are relative to the project root.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelSettings:
    """Settings for model loading and PEFT configuration."""
    model_name: str = "gpt2"
    use_peft: bool = True
    peft_method: str = "lora"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    bits: Optional[int] = None
    target_modules: Optional[List[str]] = None


@dataclass
class DataSettings:
    """Settings for data loading and preprocessing."""
    dataset_name: Optional[str] = None
    dataset_config: Optional[str] = None
    dataset_path: Optional[str] = None
    text_column: str = "text"
    max_length: int = 256
    train_test_split: float = 0.1
    use_sample_data: bool = True
    sample_size: int = 1000


@dataclass
class TrainingSettings:
    """Settings for the training process."""
    output_dir: str = "./outputs"
    run_name: str = "gpt2_finetune"
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_torch"
    max_grad_norm: float = 1.0
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    seed: int = 42


@dataclass
class EvaluationSettings:
    """Settings for model evaluation and baseline comparison."""
    eval_batch_size: int = 8
    num_eval_samples: int = 50
    max_new_tokens: int = 50
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    compare_with_baseline: bool = True
    eval_prompts: List[str] = field(default_factory=lambda: [
        "The quick brown fox",
        "In my opinion",
        "The most important thing",
        "I believe that",
        "This product is",
    ])


@dataclass
class PipelineConfig:
    """Complete configuration for the fine-tuning pipeline."""
    model: ModelSettings = field(default_factory=ModelSettings)
    data: DataSettings = field(default_factory=DataSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)
    run_training: bool = True
    run_evaluation: bool = True
    save_model: bool = True

# Quick demo configuration (fast, low resource)
DEMO_CONFIG = PipelineConfig(
    model=ModelSettings(
        model_name="distilgpt2",
        use_peft=True,
        lora_r=4,
        lora_alpha=8,
    ),
    data=DataSettings(
        use_sample_data=True,
        sample_size=500,
        max_length=128,
    ),
    training=TrainingSettings(
        num_epochs=1,
        batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        logging_steps=5,
    ),
    evaluation=EvaluationSettings(
        num_eval_samples=10,
        max_new_tokens=30,
    ),
)

# Standard GPT-2 configuration
GPT2_CONFIG = PipelineConfig(
    model=ModelSettings(
        model_name="gpt2",
        use_peft=True,
        lora_r=8,
        lora_alpha=16,
    ),
    data=DataSettings(
        use_sample_data=True,
        sample_size=2000,
        max_length=256,
    ),
    training=TrainingSettings(
        num_epochs=3,
        batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
    ),
)

# Instruction tuning configuration
INSTRUCTION_CONFIG = PipelineConfig(
    model=ModelSettings(
        model_name="gpt2",
        use_peft=True,
        lora_r=16,
        lora_alpha=32,
    ),
    data=DataSettings(
        use_sample_data=False,
        max_length=512,
    ),
    training=TrainingSettings(
        num_epochs=3,
        batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
    ),
)


def get_config(config_name: str = "default") -> PipelineConfig:
    """
    Get a configuration by name.

    Args:
        config_name: Name of the configuration preset

    Returns:
        PipelineConfig instance

    Raises:
        ValueError: If config_name is not recognized
    """
    configs = {
        "default": PipelineConfig(),
        "demo": DEMO_CONFIG,
        "gpt2": GPT2_CONFIG,
        "instruction": INSTRUCTION_CONFIG,
    }

    if config_name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available: {available}")

    return configs[config_name]
