"""
Unit tests for the evaluation module.

Tests cover:
- EvaluationConfig and EvaluationResults dataclasses
- ModelEvaluator functionality
- BaselineComparer functionality
- BLEU and ROUGE score calculations
- Report generation
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import (
    EvaluationConfig,
    EvaluationResults,
    ModelEvaluator,
    BaselineComparer,
    save_comparison_report,
)


class TestEvaluationConfig:
    """Tests for EvaluationConfig dataclass."""

    def test_evaluation_config_defaults(self):
        """Test EvaluationConfig has correct defaults."""
        config = EvaluationConfig()

        assert config.batch_size == 8
        assert config.max_new_tokens == 50
        assert config.num_samples == 100
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.do_sample is True
        assert config.seed == 42

    def test_evaluation_config_custom_values(self):
        """Test EvaluationConfig with custom values."""
        config = EvaluationConfig(
            batch_size=16,
            max_new_tokens=100,
            num_samples=50,
            temperature=0.5,
        )

        assert config.batch_size == 16
        assert config.max_new_tokens == 100
        assert config.num_samples == 50
        assert config.temperature == 0.5


class TestEvaluationResults:
    """Tests for EvaluationResults dataclass."""

    def test_evaluation_results_defaults(self):
        """Test EvaluationResults has correct defaults."""
        results = EvaluationResults(model_name="test_model")

        assert results.model_name == "test_model"
        assert results.perplexity is None
        assert results.bleu_score is None
        assert results.rouge_scores is None
        assert results.generation_samples is None
        assert results.inference_time is None
        assert results.memory_used_mb is None
        assert results.additional_metrics == {}

    def test_to_dict(self):
        """Test to_dict method."""
        results = EvaluationResults(
            model_name="test_model",
            perplexity=15.5,
            bleu_score=0.45,
            rouge_scores={"rouge_1": 0.5, "rouge_2": 0.3},
            inference_time=0.5,
            memory_used_mb=100.0,
            additional_metrics={"custom": 1.0},
        )

        result_dict = results.to_dict()

        assert result_dict["model_name"] == "test_model"
        assert result_dict["perplexity"] == 15.5
        assert result_dict["bleu_score"] == 0.45
        assert result_dict["rouge_scores"]["rouge_1"] == 0.5
        assert result_dict["inference_time"] == 0.5
        assert result_dict["memory_used_mb"] == 100.0
        assert result_dict["additional_metrics"]["custom"] == 1.0


class TestModelEvaluator:
    """Tests for ModelEvaluator class."""

    def test_evaluator_init(self):
        """Test ModelEvaluator initialization."""
        mock_model = MagicMock()
        mock_model.to = MagicMock()
        mock_model.eval = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 1
        mock_tokenizer.eos_token_id = 2

        config = EvaluationConfig()
        evaluator = ModelEvaluator(mock_model, mock_tokenizer, config)

        assert evaluator.model == mock_model
        assert evaluator.tokenizer == mock_tokenizer
        assert evaluator.config == config
        mock_model.eval.assert_called_once()

    def test_evaluator_device_detection(self):
        """Test device auto-detection."""
        mock_model = MagicMock()
        mock_model.to = MagicMock()
        mock_model.eval = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 1
        mock_tokenizer.eos_token_id = 2

        with patch("src.evaluation.torch.cuda.is_available", return_value=False):
            evaluator = ModelEvaluator(mock_model, mock_tokenizer)
            assert evaluator.device == "cpu"

    def test_compute_simple_bleu(self):
        """Test _compute_simple_bleu method."""
        mock_model = MagicMock()
        mock_model.to = MagicMock()
        mock_model.eval = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 1
        mock_tokenizer.eos_token_id = 2

        evaluator = ModelEvaluator(mock_model, mock_tokenizer)

        references = ["the cat sat on the mat"]
        predictions = ["the cat on the mat"]

        bleu = evaluator._compute_simple_bleu(references, predictions)

        # Most words in prediction are in reference
        assert 0 < bleu <= 1.0

    def test_compute_simple_bleu_empty(self):
        """Test _compute_simple_bleu with empty prediction."""
        mock_model = MagicMock()
        mock_model.to = MagicMock()
        mock_model.eval = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 1
        mock_tokenizer.eos_token_id = 2

        evaluator = ModelEvaluator(mock_model, mock_tokenizer)

        references = ["the cat sat"]
        predictions = [""]

        bleu = evaluator._compute_simple_bleu(references, predictions)
        assert bleu == 0.0

    def test_compute_simple_rouge(self):
        """Test _compute_simple_rouge method."""
        mock_model = MagicMock()
        mock_model.to = MagicMock()
        mock_model.eval = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 1
        mock_tokenizer.eos_token_id = 2

        evaluator = ModelEvaluator(mock_model, mock_tokenizer)

        references = ["the cat sat on the mat"]
        predictions = ["the cat sat on the mat"]

        rouge = evaluator._compute_simple_rouge(references, predictions)

        assert "rouge_1" in rouge
        assert "rouge_2" in rouge
        # Perfect match should give rouge_1 = 1.0
        assert rouge["rouge_1"] == 1.0

    def test_compute_simple_rouge_partial(self):
        """Test _compute_simple_rouge with partial match."""
        mock_model = MagicMock()
        mock_model.to = MagicMock()
        mock_model.eval = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 1
        mock_tokenizer.eos_token_id = 2

        evaluator = ModelEvaluator(mock_model, mock_tokenizer)

        references = ["the cat sat on the mat"]
        predictions = ["the cat"]

        rouge = evaluator._compute_simple_rouge(references, predictions)

        # 2 out of 6 words match (rouge_1 is recall)
        assert 0 < rouge["rouge_1"] <= 1.0

    def test_compute_simple_rouge_empty(self):
        """Test _compute_simple_rouge with empty inputs."""
        mock_model = MagicMock()
        mock_model.to = MagicMock()
        mock_model.eval = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 1
        mock_tokenizer.eos_token_id = 2

        evaluator = ModelEvaluator(mock_model, mock_tokenizer)

        references = [""]
        predictions = [""]

        rouge = evaluator._compute_simple_rouge(references, predictions)

        assert rouge["rouge_1"] == 0.0
        assert rouge["rouge_2"] == 0.0

    def test_evaluate_generation_quality(self):
        """Test evaluate_generation_quality method."""
        mock_model = MagicMock()
        mock_model.to = MagicMock()
        mock_model.eval = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 1
        mock_tokenizer.eos_token_id = 2

        evaluator = ModelEvaluator(mock_model, mock_tokenizer)

        references = ["hello world"]
        predictions = ["hello world"]

        metrics = evaluator.evaluate_generation_quality(references, predictions)

        assert "bleu" in metrics
        assert "rouge_1" in metrics
        assert "avg_reference_length" in metrics
        assert "avg_prediction_length" in metrics

    def test_generate_samples(self):
        """Test generate_samples method."""
        mock_model = MagicMock()
        mock_model.to = MagicMock()
        mock_model.eval = MagicMock()
        mock_model.generate = MagicMock(return_value=MagicMock())
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 1
        mock_tokenizer.eos_token_id = 2

        # Set up tokenizer to return properly mocked object with .to() method
        mock_inputs = MagicMock()
        mock_inputs.to = MagicMock(return_value=mock_inputs)
        mock_inputs.__getitem__ = MagicMock(return_value=MagicMock())
        mock_tokenizer.return_value = mock_inputs
        mock_tokenizer.decode = MagicMock(return_value="Generated text")

        config = EvaluationConfig(max_new_tokens=20)

        with patch.object(ModelEvaluator, "__init__", lambda self, m, t, c=None, d=None: None):
            evaluator = ModelEvaluator.__new__(ModelEvaluator)
            evaluator.model = mock_model
            evaluator.tokenizer = mock_tokenizer
            evaluator.config = config
            evaluator.device = "cpu"
            evaluator.generation_config = MagicMock()

            with patch("src.evaluation.tqdm") as mock_tqdm:
                mock_tqdm.return_value = ["Hello"]
                with patch("src.evaluation.torch.no_grad"):
                    samples = evaluator.generate_samples(["Hello"])

            assert len(samples) == 1
            assert "prompt" in samples[0]
            assert "generated" in samples[0]


class TestBaselineComparer:
    """Tests for BaselineComparer class."""

    def test_comparer_init(self):
        """Test BaselineComparer initialization."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 1
        mock_tokenizer.eos_token_id = 2

        with patch("src.evaluation.AutoModelForCausalLM.from_pretrained") as mock_load:
            mock_load.return_value = MagicMock()

            comparer = BaselineComparer(
                mock_model,
                "gpt2",
                mock_tokenizer,
            )

            assert comparer.fine_tuned_model == mock_model
            assert comparer.base_model_name == "gpt2"
            assert comparer.tokenizer == mock_tokenizer

    def test_generate_report(self):
        """Test generate_report method."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch("src.evaluation.AutoModelForCausalLM.from_pretrained") as mock_load:
            mock_load.return_value = MagicMock()

            comparer = BaselineComparer(mock_model, "gpt2", mock_tokenizer)

            comparison = {
                "fine_tuned": {
                    "perplexity": 15.0,
                    "bleu_score": 0.5,
                    "rouge_scores": {"rouge_1": 0.6, "rouge_2": 0.4},
                },
                "baseline": {
                    "perplexity": 20.0,
                    "bleu_score": 0.4,
                    "rouge_scores": {"rouge_1": 0.5, "rouge_2": 0.3},
                },
                "comparison": {
                    "perplexity_improvement_pct": 25.0,
                    "bleu_improvement": 0.1,
                    "rouge_1_delta": 0.1,
                    "rouge_2_delta": 0.1,
                },
                "generation_comparison": [],
            }

            report = comparer.generate_report(comparison)

            assert "FINE-TUNING EVALUATION REPORT" in report
            assert "PERPLEXITY COMPARISON" in report
            assert "15.00" in report
            assert "20.00" in report


class TestSaveComparisonReport:
    """Tests for save_comparison_report function."""

    def test_save_comparison_report(self):
        """Test saving comparison report to JSON."""
        comparison = {
            "fine_tuned": {"perplexity": 15.0},
            "baseline": {"perplexity": 20.0},
            "comparison": {"improvement": 25.0},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "comparison.json")
            save_comparison_report(comparison, output_path)

            assert os.path.exists(output_path)

            with open(output_path) as f:
                loaded = json.load(f)

            assert loaded["fine_tuned"]["perplexity"] == 15.0
            assert loaded["baseline"]["perplexity"] == 20.0

    def test_save_comparison_report_creates_dirs(self):
        """Test that save_comparison_report creates directories."""
        comparison = {"test": "data"}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "nested", "dir", "comparison.json")
            save_comparison_report(comparison, output_path)

            assert os.path.exists(output_path)


class TestBLEUScore:
    """Additional tests for BLEU score calculation."""

    def test_bleu_perfect_match(self):
        """Test BLEU score with perfect match."""
        mock_model = MagicMock()
        mock_model.to = MagicMock()
        mock_model.eval = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 1
        mock_tokenizer.eos_token_id = 2

        evaluator = ModelEvaluator(mock_model, mock_tokenizer)

        references = ["the cat sat on the mat"]
        predictions = ["the cat sat on the mat"]

        bleu = evaluator._compute_simple_bleu(references, predictions)
        assert bleu == 1.0

    def test_bleu_no_match(self):
        """Test BLEU score with no word overlap."""
        mock_model = MagicMock()
        mock_model.to = MagicMock()
        mock_model.eval = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 1
        mock_tokenizer.eos_token_id = 2

        evaluator = ModelEvaluator(mock_model, mock_tokenizer)

        references = ["aaa bbb ccc"]
        predictions = ["xxx yyy zzz"]

        bleu = evaluator._compute_simple_bleu(references, predictions)
        assert bleu == 0.0


class TestROUGEScore:
    """Additional tests for ROUGE score calculation."""

    def test_rouge_perfect_match(self):
        """Test ROUGE score with perfect match."""
        mock_model = MagicMock()
        mock_model.to = MagicMock()
        mock_model.eval = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 1
        mock_tokenizer.eos_token_id = 2

        evaluator = ModelEvaluator(mock_model, mock_tokenizer)

        references = ["the cat sat on the mat"]
        predictions = ["the cat sat on the mat"]

        rouge = evaluator._compute_simple_rouge(references, predictions)

        assert rouge["rouge_1"] == 1.0
        # All bigrams match
        assert rouge["rouge_2"] == 1.0

    def test_rouge_case_insensitive(self):
        """Test that ROUGE is case-insensitive."""
        mock_model = MagicMock()
        mock_model.to = MagicMock()
        mock_model.eval = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 1
        mock_tokenizer.eos_token_id = 2

        evaluator = ModelEvaluator(mock_model, mock_tokenizer)

        references = ["THE CAT SAT"]
        predictions = ["the cat sat"]

        rouge = evaluator._compute_simple_rouge(references, predictions)

        assert rouge["rouge_1"] == 1.0
