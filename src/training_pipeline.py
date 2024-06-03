"""
Training Pipeline Module for LLM Fine-Tuning

This module handles:
- Training configuration and hyperparameters
- Training loop with Hugging Face Trainer
- Logging and checkpointing
- Learning rate scheduling
- Gradient accumulation and memory optimization
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """
    Configuration for the training process.

    Attributes:
        output_dir: Directory for outputs and checkpoints
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device
        per_device_eval_batch_size: Eval batch size per device
        gradient_accumulation_steps: Steps for gradient accumulation
        learning_rate: Learning rate
        weight_decay: Weight decay coefficient
        warmup_steps: Number of warmup steps
        warmup_ratio: Ratio of warmup steps
        logging_steps: Steps between logging
        eval_steps: Steps between evaluations (if eval_strategy="steps")
        save_steps: Steps between saves (if save_strategy="steps")
        save_total_limit: Maximum number of checkpoints to keep
        fp16: Whether to use FP16 training
        bf16: Whether to use BF16 training
        gradient_checkpointing: Whether to use gradient checkpointing
        optim: Optimizer type
        lr_scheduler_type: Learning rate scheduler type
        max_grad_norm: Maximum gradient norm for clipping
    """
    output_dir: str = "./outputs"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    warmup_ratio: float = 0.0
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 3
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = True
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    seed: int = 42
    dataloader_num_workers: int = 0
    remove_unused_columns: bool = False
    label_names: Optional[List[str]] = None
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    def to_training_arguments(self, **override_kwargs) -> TrainingArguments:
        """
        Convert to Hugging Face TrainingArguments.

        Args:
            **override_kwargs: Arguments to override

        Returns:
            TrainingArguments instance
        """
        args = {
            "output_dir": self.output_dir,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "warmup_ratio": self.warmup_ratio,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "optim": self.optim,
            "lr_scheduler_type": self.lr_scheduler_type,
            "max_grad_norm": self.max_grad_norm,
            "seed": self.seed,
            "dataloader_num_workers": self.dataloader_num_workers,
            "remove_unused_columns": self.remove_unused_columns,
            "load_best_model_at_end": self.load_best_model_at_end,
            "metric_for_best_model": self.metric_for_best_model,
            "greater_is_better": self.greater_is_better,
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "logging_dir": os.path.join(self.output_dir, "logs"),
            "report_to": ["tensorboard"],
        }
        args.update(override_kwargs)
        return TrainingArguments(**args)


class TrainingMetricsCallback(TrainerCallback):
    """Callback to track and log training metrics."""

    def __init__(self):
        self.training_history = []
        self.eval_history = []
        self.best_eval_loss = float("inf")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Route log events to appropriate handlers."""
        if not logs:
            return

        if "loss" in logs:
            self._record_training_loss(state, logs)

        if "eval_loss" in logs:
            self._record_eval_loss(state, logs)

    def _record_training_loss(self, state, logs):
        """Record training loss from logs."""
        self.training_history.append({
            "step": state.global_step,
            "epoch": state.epoch,
            "loss": logs["loss"],
            "learning_rate": logs.get("learning_rate"),
        })

    def _record_eval_loss(self, state, logs):
        """Record evaluation loss and track best."""
        self.eval_history.append({
            "step": state.global_step,
            "epoch": state.epoch,
            "eval_loss": logs["eval_loss"],
        })

        if logs["eval_loss"] < self.best_eval_loss:
            self.best_eval_loss = logs["eval_loss"]
            logger.info(f"New best eval loss: {self.best_eval_loss:.4f}")

    def get_history(self) -> Dict[str, List[Dict]]:
        """Get training and evaluation history."""
        return {
            "training": self.training_history,
            "evaluation": self.eval_history,
            "best_eval_loss": self.best_eval_loss,
        }


class FineTuningTrainer:
    """
    Main training class for fine-tuning language models.

    This class encapsulates the entire training pipeline including:
    - Model and tokenizer setup
    - Data collation
    - Training with callbacks
    - Evaluation
    - Saving results
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: TrainingConfig,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
    ):
        """
        Initialize the FineTuningTrainer.

        Args:
            model: Pre-trained or PEFT model
            tokenizer: Tokenizer for the model
            config: Training configuration
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            callbacks: Additional callbacks
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize callbacks
        self.metrics_callback = TrainingMetricsCallback()
        self.callbacks = [self.metrics_callback]
        if callbacks:
            self.callbacks.extend(callbacks)

        # Data collator for causal language modeling
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        # Create trainer
        self.trainer: Optional[Trainer] = None
        self.training_result = None

    def setup_trainer(self, **training_args_overrides) -> Trainer:
        """
        Set up the Hugging Face Trainer.

        Args:
            **training_args_overrides: Overrides for training arguments

        Returns:
            Configured Trainer instance
        """
        training_args = self.config.to_training_arguments(**training_args_overrides)

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
            data_collator=self.data_collator,
            callbacks=self.callbacks,
        )

        logger.info("Trainer setup complete")
        return self.trainer

    def train(self, resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the training loop.

        Args:
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            Training results dictionary
        """
        if self.trainer is None:
            self.setup_trainer()

        logger.info("Starting training...")
        logger.info(f"Training samples: {len(self.train_dataset)}")
        if self.eval_dataset:
            logger.info(f"Evaluation samples: {len(self.eval_dataset)}")

        self.training_result = self.trainer.train(
            resume_from_checkpoint=resume_from_checkpoint
        )

        results = {
            "training_loss": self.training_result.training_loss,
            "metrics": self.training_result.metrics,
            "history": self.metrics_callback.get_history(),
        }

        logger.info(f"Training complete. Final loss: {self.training_result.training_loss:.4f}")

        return results

    def evaluate(self, dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Evaluate the model.

        Args:
            dataset: Dataset to evaluate on (uses eval_dataset if None)

        Returns:
            Evaluation metrics
        """
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call setup_trainer() first.")

        eval_dataset = dataset or self.eval_dataset
        if eval_dataset is None:
            raise ValueError("No evaluation dataset provided")

        logger.info("Running evaluation...")
        metrics = self.trainer.evaluate(eval_dataset)

        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def save_results(self, output_path: Optional[str] = None) -> None:
        """
        Save training results to a JSON file.

        Args:
            output_path: Path to save results (default: output_dir/results.json)
        """
        if output_path is None:
            output_path = self.output_dir / "training_results.json"

        results = {
            "config": {
                "output_dir": str(self.config.output_dir),
                "num_train_epochs": self.config.num_train_epochs,
                "per_device_train_batch_size": self.config.per_device_train_batch_size,
                "learning_rate": self.config.learning_rate,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            },
            "history": self.metrics_callback.get_history(),
            "training_completed": datetime.now().isoformat(),
        }

        if self.training_result:
            results["final_training_loss"] = self.training_result.training_loss
            results["metrics"] = self.training_result.metrics

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results saved to {output_path}")

    def save_model(self, output_dir: Optional[str] = None) -> None:
        """
        Save the fine-tuned model.

        Args:
            output_dir: Directory to save to (default: config.output_dir/final_model)
        """
        if output_dir is None:
            output_dir = self.output_dir / "final_model"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.trainer:
            self.trainer.save_model(str(output_dir))
            logger.info(f"Model saved to {output_dir}")
        else:
            self.model.save_pretrained(str(output_dir))
            self.tokenizer.save_pretrained(str(output_dir))
            logger.info(f"Model saved to {output_dir}")


def compute_training_stats(history: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """
    Compute statistics from training history.

    Args:
        history: Training history dictionary

    Returns:
        Statistics dictionary
    """
    training = history.get("training", [])
    evaluation = history.get("evaluation", [])

    stats = {}

    if training:
        losses = [h["loss"] for h in training]
        stats["training"] = {
            "num_steps": len(training),
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "min_loss": min(losses),
            "max_loss": max(losses),
            "loss_reduction": losses[0] - losses[-1],
            "loss_reduction_pct": (losses[0] - losses[-1]) / losses[0] * 100,
        }

    if evaluation:
        eval_losses = [h["eval_loss"] for h in evaluation]
        stats["evaluation"] = {
            "num_evals": len(evaluation),
            "initial_loss": eval_losses[0],
            "final_loss": eval_losses[-1],
            "min_loss": min(eval_losses),
            "best_eval_loss": history.get("best_eval_loss", min(eval_losses)),
        }

    return stats


# Convenience function for quick training
def quick_train(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    output_dir: str = "./outputs",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    **kwargs,
) -> Tuple[FineTuningTrainer, Dict[str, Any]]:
    """
    Quick training function with sensible defaults.

    Args:
        model: Model to train
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        output_dir: Output directory
        num_epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        **kwargs: Additional training config options

    Returns:
        Tuple of (trainer, results)
    """
    config = TrainingConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        **kwargs
    )

    trainer = FineTuningTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    results = trainer.train()

    return trainer, results


if __name__ == "__main__":
    # Example usage
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load small model for demo
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create sample dataset
    sample_texts = [
        {"input_ids": tokenizer("Hello world!", return_tensors="pt")["input_ids"][0].tolist(),
         "labels": tokenizer("Hello world!", return_tensors="pt")["input_ids"][0].tolist()}
        for _ in range(50)
    ]
    train_dataset = Dataset.from_list(sample_texts)

    # Training config
    config = TrainingConfig(
        output_dir="./outputs/demo",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        learning_rate=5e-5,
        logging_steps=5,
        save_total_limit=1,
    )

    # Train
    trainer = FineTuningTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        train_dataset=train_dataset,
    )

    results = trainer.train()

    print("\nTraining Results:")
    print(f"Final loss: {results['training_loss']:.4f}")

    # Save model
    trainer.save_model()

    print("\nDemo training complete!")
