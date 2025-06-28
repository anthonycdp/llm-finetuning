"""
Unit tests for the training_pipeline module.

Tests cover:
- TrainingConfig dataclass and conversion
- TrainingMetricsCallback functionality
- FineTuningTrainer initialization and setup
- Training statistics computation
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training_pipeline import (
    TrainingConfig,
    TrainingMetricsCallback,
    FineTuningTrainer,
    compute_training_stats,
)


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_training_config_defaults(self):
        """Test TrainingConfig has correct defaults."""
        config = TrainingConfig()

        assert config.output_dir == "./outputs"
        assert config.num_train_epochs == 3
        assert config.per_device_train_batch_size == 4
        assert config.per_device_eval_batch_size == 4
        assert config.gradient_accumulation_steps == 4
        assert config.learning_rate == 5e-5
        assert config.weight_decay == 0.01
        assert config.fp16 is False
        assert config.bf16 is False
        assert config.gradient_checkpointing is True
        assert config.seed == 42

    def test_training_config_custom_values(self):
        """Test TrainingConfig with custom values."""
        config = TrainingConfig(
            output_dir="/custom/output",
            num_train_epochs=5,
            per_device_train_batch_size=8,
            learning_rate=1e-4,
            fp16=True,
        )

        assert config.output_dir == "/custom/output"
        assert config.num_train_epochs == 5
        assert config.per_device_train_batch_size == 8
        assert config.learning_rate == 1e-4
        assert config.fp16 is True

    def test_to_training_arguments(self):
        """Test conversion to TrainingArguments."""
        config = TrainingConfig(
            output_dir="./test_output",
            num_train_epochs=2,
            per_device_train_batch_size=4,
        )

        with patch("src.training_pipeline.TrainingArguments") as mock_args:
            mock_args.return_value = MagicMock()
            result = config.to_training_arguments()

            # Verify TrainingArguments was called
            mock_args.assert_called_once()
            call_kwargs = mock_args.call_args[1]

            assert call_kwargs["output_dir"] == "./test_output"
            assert call_kwargs["num_train_epochs"] == 2
            assert call_kwargs["per_device_train_batch_size"] == 4

    def test_to_training_arguments_with_overrides(self):
        """Test conversion to TrainingArguments with overrides."""
        config = TrainingConfig(
            output_dir="./test_output",
            num_train_epochs=2,
        )

        with patch("src.training_pipeline.TrainingArguments") as mock_args:
            mock_args.return_value = MagicMock()
            result = config.to_training_arguments(
                num_train_epochs=5,
                custom_arg="value",
            )

            call_kwargs = mock_args.call_args[1]
            assert call_kwargs["num_train_epochs"] == 5
            assert call_kwargs["custom_arg"] == "value"


class TestTrainingMetricsCallback:
    """Tests for TrainingMetricsCallback."""

    def test_callback_init(self):
        """Test callback initialization."""
        callback = TrainingMetricsCallback()

        assert callback.training_history == []
        assert callback.eval_history == []
        assert callback.best_eval_loss == float("inf")

    def test_on_log_training_loss(self):
        """Test on_log with training loss."""
        callback = TrainingMetricsCallback()

        mock_args = MagicMock()
        mock_state = MagicMock()
        mock_state.global_step = 10
        mock_state.epoch = 0.5
        mock_control = MagicMock()

        callback.on_log(
            mock_args, mock_state, mock_control,
            logs={"loss": 2.5, "learning_rate": 1e-5}
        )

        assert len(callback.training_history) == 1
        assert callback.training_history[0]["step"] == 10
        assert callback.training_history[0]["loss"] == 2.5
        assert callback.training_history[0]["learning_rate"] == 1e-5

    def test_on_log_eval_loss(self):
        """Test on_log with eval loss."""
        callback = TrainingMetricsCallback()

        mock_args = MagicMock()
        mock_state = MagicMock()
        mock_state.global_step = 100
        mock_state.epoch = 1.0
        mock_control = MagicMock()

        callback.on_log(
            mock_args, mock_state, mock_control,
            logs={"eval_loss": 1.5}
        )

        assert len(callback.eval_history) == 1
        assert callback.eval_history[0]["eval_loss"] == 1.5
        assert callback.best_eval_loss == 1.5

    def test_on_log_updates_best_eval_loss(self):
        """Test that best_eval_loss is updated."""
        callback = TrainingMetricsCallback()

        mock_args = MagicMock()
        mock_state = MagicMock()
        mock_state.global_step = 100
        mock_state.epoch = 1.0
        mock_control = MagicMock()

        callback.on_log(
            mock_args, mock_state, mock_control,
            logs={"eval_loss": 2.0}
        )
        assert callback.best_eval_loss == 2.0

        callback.on_log(
            mock_args, mock_state, mock_control,
            logs={"eval_loss": 1.5}
        )
        assert callback.best_eval_loss == 1.5

        callback.on_log(
            mock_args, mock_state, mock_control,
            logs={"eval_loss": 1.8}
        )
        assert callback.best_eval_loss == 1.5  # Should not increase

    def test_get_history(self):
        """Test get_history method."""
        callback = TrainingMetricsCallback()
        callback.training_history = [{"step": 1, "loss": 2.0}]
        callback.eval_history = [{"step": 1, "eval_loss": 1.5}]
        callback.best_eval_loss = 1.5

        history = callback.get_history()

        assert "training" in history
        assert "evaluation" in history
        assert "best_eval_loss" in history
        assert len(history["training"]) == 1
        assert history["best_eval_loss"] == 1.5


class TestFineTuningTrainer:
    """Tests for FineTuningTrainer class."""

    def test_trainer_init(self):
        """Test FineTuningTrainer initialization."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)

        config = TrainingConfig(output_dir="./test_output")

        trainer = FineTuningTrainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=config,
            train_dataset=mock_dataset,
        )

        assert trainer.model == mock_model
        assert trainer.tokenizer == mock_tokenizer
        assert trainer.config == config
        assert trainer.train_dataset == mock_dataset
        assert trainer.trainer is None

    def test_trainer_creates_output_dir(self):
        """Test that trainer creates output directory."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_dataset = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "test_output")
            config = TrainingConfig(output_dir=output_dir)

            trainer = FineTuningTrainer(
                model=mock_model,
                tokenizer=mock_tokenizer,
                config=config,
                train_dataset=mock_dataset,
            )

            assert os.path.exists(output_dir)

    def test_setup_trainer(self):
        """Test setup_trainer method."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(output_dir=tmpdir)
            trainer = FineTuningTrainer(
                model=mock_model,
                tokenizer=mock_tokenizer,
                config=config,
                train_dataset=mock_dataset,
            )

            with patch("src.training_pipeline.Trainer") as mock_trainer_class:
                mock_trainer_instance = MagicMock()
                mock_trainer_class.return_value = mock_trainer_instance

                result = trainer.setup_trainer()

                assert trainer.trainer is not None
                mock_trainer_class.assert_called_once()

    def test_evaluate_raises_without_trainer(self):
        """Test that evaluate raises error when trainer not initialized."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_dataset = MagicMock()

        config = TrainingConfig(output_dir="./test_output")
        trainer = FineTuningTrainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=config,
            train_dataset=mock_dataset,
        )

        with pytest.raises(ValueError, match="Trainer not initialized"):
            trainer.evaluate()

    def test_evaluate_raises_without_dataset(self):
        """Test that evaluate raises error when no dataset provided."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_dataset = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(output_dir=tmpdir)
            trainer = FineTuningTrainer(
                model=mock_model,
                tokenizer=mock_tokenizer,
                config=config,
                train_dataset=mock_dataset,
            )
            # Don't set eval_dataset

            # First set up the trainer so we can test the dataset check
            with patch("src.training_pipeline.Trainer") as mock_trainer_class:
                mock_trainer_instance = MagicMock()
                mock_trainer_class.return_value = mock_trainer_instance
                trainer.setup_trainer()

            # Now test that evaluate raises error for missing dataset
            with pytest.raises(ValueError, match="No evaluation dataset"):
                trainer.evaluate()

    def test_save_results(self):
        """Test save_results method."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_dataset = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(output_dir=tmpdir)
            trainer = FineTuningTrainer(
                model=mock_model,
                tokenizer=mock_tokenizer,
                config=config,
                train_dataset=mock_dataset,
            )

            results_path = os.path.join(tmpdir, "results.json")
            trainer.save_results(results_path)

            assert os.path.exists(results_path)

    def test_save_model_with_trainer(self):
        """Test save_model method when trainer exists."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_dataset = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(output_dir=tmpdir)
            trainer = FineTuningTrainer(
                model=mock_model,
                tokenizer=mock_tokenizer,
                config=config,
                train_dataset=mock_dataset,
            )

            # Set up mock trainer
            trainer.trainer = MagicMock()

            model_dir = os.path.join(tmpdir, "saved_model")
            trainer.save_model(model_dir)

            trainer.trainer.save_model.assert_called_once()


class TestComputeTrainingStats:
    """Tests for compute_training_stats function."""

    def test_compute_stats_empty_history(self):
        """Test compute_training_stats with empty history."""
        history = {
            "training": [],
            "evaluation": [],
        }

        stats = compute_training_stats(history)

        assert stats == {}

    def test_compute_stats_training_only(self):
        """Test compute_training_stats with training history only."""
        history = {
            "training": [
                {"step": 1, "loss": 3.0},
                {"step": 2, "loss": 2.5},
                {"step": 3, "loss": 2.0},
            ],
            "evaluation": [],
        }

        stats = compute_training_stats(history)

        assert "training" in stats
        assert stats["training"]["initial_loss"] == 3.0
        assert stats["training"]["final_loss"] == 2.0
        assert stats["training"]["min_loss"] == 2.0
        assert stats["training"]["max_loss"] == 3.0
        assert stats["training"]["loss_reduction"] == 1.0

    def test_compute_stats_with_evaluation(self):
        """Test compute_training_stats with evaluation history."""
        history = {
            "training": [
                {"step": 1, "loss": 3.0},
                {"step": 2, "loss": 2.0},
            ],
            "evaluation": [
                {"step": 1, "eval_loss": 2.5},
                {"step": 2, "eval_loss": 2.2},
            ],
            "best_eval_loss": 2.2,
        }

        stats = compute_training_stats(history)

        assert "evaluation" in stats
        assert stats["evaluation"]["num_evals"] == 2
        assert stats["evaluation"]["initial_loss"] == 2.5
        assert stats["evaluation"]["final_loss"] == 2.2
        assert stats["evaluation"]["best_eval_loss"] == 2.2

    def test_compute_stats_loss_reduction_percentage(self):
        """Test that loss reduction percentage is calculated correctly."""
        history = {
            "training": [
                {"step": 1, "loss": 4.0},
                {"step": 2, "loss": 2.0},
            ],
            "evaluation": [],
        }

        stats = compute_training_stats(history)

        # 50% reduction from 4.0 to 2.0
        assert stats["training"]["loss_reduction_pct"] == 50.0
