#!/usr/bin/env python3
"""
LLM Fine-Tuning Pipeline - Main Entry Point

This script provides a complete end-to-end fine-tuning pipeline including:
1. Data preparation and preprocessing
2. Model loading with PEFT/LoRA
3. Training with monitoring
4. Evaluation against baseline
5. Results reporting

Usage:
    python -m src.main --config demo
    python -m src.main --model gpt2 --epochs 3 --lr 5e-5
    python -m src.main --help
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PipelineConfig, get_config
from src.data_preparation import DataPreparer
from src.evaluation import BaselineComparer, EvaluationConfig, ModelEvaluator, save_comparison_report
from src.model_config import ModelConfig, ModelManager, create_model_manager
from src.training_pipeline import FineTuningTrainer, TrainingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LLM Fine-Tuning Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        choices=["default", "demo", "gpt2", "instruction"],
        help="Configuration preset to use",
    )

    # Model settings
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name or path (overrides config)",
    )
    parser.add_argument(
        "--peft-method",
        type=str,
        choices=["lora", "qlora", "none"],
        default=None,
        help="PEFT method (overrides config)",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=None,
        help="LoRA rank (overrides config)",
    )

    # Training settings
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )

    # Data settings
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="HuggingFace dataset name (overrides config)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Sample size for demo data (overrides config)",
    )

    # Pipeline control
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and run evaluation only",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        default=True,
        help="Compare with baseline model",
    )

    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def apply_overrides(config: PipelineConfig, args: argparse.Namespace) -> PipelineConfig:
    """Apply command line overrides to configuration."""
    if args.model:
        config.model.model_name = args.model
    if args.peft_method:
        config.model.peft_method = args.peft_method
        config.model.use_peft = args.peft_method != "none"
    if args.lora_r:
        config.model.lora_r = args.lora_r
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.dataset:
        config.data.dataset_name = args.dataset
        config.data.use_sample_data = False
    if args.sample_size:
        config.data.sample_size = args.sample_size

    config.training.seed = args.seed
    config.run_training = not args.skip_training
    config.run_evaluation = not args.skip_eval
    config.evaluation.compare_with_baseline = args.compare_baseline

    return config


def setup_output_directory(base_dir: str, run_name: str) -> Path:
    """Create and return output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / f"{run_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_data_preparation(config: PipelineConfig) -> Tuple[Any, Any]:
    """Prepare dataset for training."""
    logger.info("=" * 60)
    logger.info("STEP 1: DATA PREPARATION")
    logger.info("=" * 60)

    preparer = DataPreparer(
        tokenizer_name=config.model.model_name,
        max_length=config.data.max_length,
        text_column=config.data.text_column,
        seed=config.training.seed,
    )

    if config.data.use_sample_data:
        logger.info("Using sample data for demonstration")
        dataset = preparer.create_sample_dataset(num_samples=config.data.sample_size)
    elif config.data.dataset_name:
        logger.info(f"Loading dataset: {config.data.dataset_name}")
        dataset = preparer.load_dataset_from_hub(
            config.data.dataset_name,
            subset=config.data.dataset_config,
        )
    elif config.data.dataset_path:
        logger.info(f"Loading dataset from: {config.data.dataset_path}")
        dataset = preparer.load_dataset_from_json(
            config.data.dataset_path,
            train_test_split=config.data.train_test_split,
        )
    else:
        logger.info("No dataset specified, using sample data")
        dataset = preparer.create_sample_dataset(num_samples=config.data.sample_size)

    # Tokenize dataset
    tokenized_dataset = preparer.prepare_dataset(dataset)

    logger.info(f"Dataset prepared: {len(tokenized_dataset['train'])} train, "
                f"{len(tokenized_dataset.get('test', []))} test samples")

    return tokenized_dataset, preparer


def run_model_setup(config: PipelineConfig) -> tuple:
    """Set up model and tokenizer."""
    logger.info("=" * 60)
    logger.info("STEP 2: MODEL SETUP")
    logger.info("=" * 60)

    model_config = ModelConfig(
        model_name=config.model.model_name,
        peft_method=config.model.peft_method if config.model.use_peft else "none",
        lora_r=config.model.lora_r,
        lora_alpha=config.model.lora_alpha,
        lora_dropout=config.model.lora_dropout,
        target_modules=config.model.target_modules,
        bits=config.model.bits,
    )

    manager = ModelManager(model_config)
    model, tokenizer = manager.prepare_for_training()

    # Print model info
    info = manager.get_model_info()
    logger.info(f"Model: {info['model_name']}")
    logger.info(f"Total parameters: {info['total_parameters']:,}")
    logger.info(f"Trainable parameters: {info['trainable_parameters']:,}")
    logger.info(f"Trainable %: {info['trainable_percentage']:.2f}%")

    return model, tokenizer, manager


def run_training(
    config: PipelineConfig,
    model: Any,
    tokenizer: Any,
    dataset: Any,
    output_dir: Path,
) -> tuple:
    """Run training loop."""
    logger.info("=" * 60)
    logger.info("STEP 3: TRAINING")
    logger.info("=" * 60)

    training_config = TrainingConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=config.training.num_epochs,
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_type=config.training.lr_scheduler_type,
        optim=config.training.optim,
        max_grad_norm=config.training.max_grad_norm,
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        gradient_checkpointing=config.training.gradient_checkpointing,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        load_best_model_at_end=config.training.load_best_model_at_end,
        metric_for_best_model=config.training.metric_for_best_model,
        seed=config.training.seed,
    )

    trainer = FineTuningTrainer(
        model=model,
        tokenizer=tokenizer,
        config=training_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test"),
    )

    results = trainer.train()
    trainer.save_results(str(output_dir / "training_results.json"))

    if config.save_model:
        trainer.save_model(str(output_dir / "final_model"))

    return trainer, results


def run_evaluation(
    config: PipelineConfig,
    model: Any,
    tokenizer: Any,
    dataset: Any,
    model_manager: Any,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run evaluation and baseline comparison."""
    logger.info("=" * 60)
    logger.info("STEP 4: EVALUATION")
    logger.info("=" * 60)

    eval_config = EvaluationConfig(
        batch_size=config.evaluation.eval_batch_size,
        num_samples=config.evaluation.num_eval_samples,
        max_new_tokens=config.evaluation.max_new_tokens,
        temperature=config.evaluation.temperature,
        top_p=config.evaluation.top_p,
        do_sample=config.evaluation.do_sample,
    )

    results = {}

    if config.evaluation.compare_with_baseline:
        logger.info("Running baseline comparison...")
        comparer = BaselineComparer(
            fine_tuned_model=model,
            base_model_name=config.model.model_name,
            tokenizer=tokenizer,
            config=eval_config,
        )

        comparison = comparer.compare(
            eval_dataset=dataset["test"],
            prompts=config.evaluation.eval_prompts,
        )

        # Generate and save report
        report = comparer.generate_report(comparison)
        logger.info("\n" + report)

        # Save comparison results
        save_comparison_report(comparison, output_dir / "comparison_report.json")

        # Save text report
        with open(output_dir / "evaluation_report.txt", "w", encoding="utf-8") as f:
            f.write(report)

        results["comparison"] = comparison
    else:
        logger.info("Running standalone evaluation...")
        evaluator = ModelEvaluator(model, tokenizer, eval_config)

        eval_results = evaluator.full_evaluation(
            dataset["test"],
            prompts=config.evaluation.eval_prompts,
            model_name="Fine-tuned",
        )

        logger.info(f"Perplexity: {eval_results.perplexity:.2f}")
        results["evaluation"] = eval_results.to_dict()

    return results


def save_run_config(config: PipelineConfig, output_dir: Path) -> None:
    """Save the run configuration."""
    config_dict = {
        "model": {
            "model_name": config.model.model_name,
            "peft_method": config.model.peft_method,
            "lora_r": config.model.lora_r,
            "lora_alpha": config.model.lora_alpha,
        },
        "training": {
            "num_epochs": config.training.num_epochs,
            "batch_size": config.training.batch_size,
            "learning_rate": config.training.learning_rate,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        },
        "data": {
            "use_sample_data": config.data.use_sample_data,
            "sample_size": config.data.sample_size,
            "max_length": config.data.max_length,
        },
    }

    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)


def main():
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load and configure
    config = get_config(args.config)
    config = apply_overrides(config, args)

    # Setup output directory
    output_dir = setup_output_directory(
        config.training.output_dir,
        config.training.run_name,
    )
    logger.info(f"Output directory: {output_dir}")

    # Save configuration
    save_run_config(config, output_dir)

    try:
        # Step 1: Data preparation
        dataset, preparer = run_data_preparation(config)

        # Step 2: Model setup
        model, tokenizer, model_manager = run_model_setup(config)

        # Step 3: Training
        if config.run_training:
            trainer, training_results = run_training(
                config, model, tokenizer, dataset, output_dir
            )
        else:
            logger.info("Skipping training (--skip-training)")

        # Step 4: Evaluation
        if config.run_evaluation:
            eval_results = run_evaluation(
                config, model, tokenizer, dataset, model_manager, output_dir
            )

        # Summary
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"  - Training results: training_results.json")
        logger.info(f"  - Model: final_model/")
        logger.info(f"  - Evaluation: comparison_report.json")

        print(f"\n[OK] Fine-tuning pipeline completed successfully!")
        print(f"  Output directory: {output_dir}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
