"""
Unit tests for the data_preparation module.

Tests cover:
- DataPreparer initialization and configuration
- Text preprocessing
- Tokenization
- Dataset loading and creation
- Instruction dataset formatting
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_preparation import (
    DataPreparer,
    create_instruction_dataset,
)


class TestDataPreparerInit:
    """Tests for DataPreparer initialization."""

    def test_init_default_params(self):
        """Test DataPreparer initializes with default parameters."""
        with patch("src.data_preparation.AutoTokenizer.from_pretrained") as mock_tokenizer:
            mock_tokenizer.return_value = MagicMock()
            mock_tokenizer.return_value.pad_token = "<|endoftext|>"

            preparer = DataPreparer()

            assert preparer.tokenizer_name == "gpt2"
            assert preparer.max_length == 512
            assert preparer.text_column == "text"
            assert preparer.seed == 42
            assert preparer.padding == "max_length"
            assert preparer.truncation is True

    def test_init_custom_params(self):
        """Test DataPreparer initializes with custom parameters."""
        with patch("src.data_preparation.AutoTokenizer.from_pretrained") as mock_tokenizer:
            mock_tokenizer.return_value = MagicMock()
            mock_tokenizer.return_value.pad_token = "<|endoftext|>"

            preparer = DataPreparer(
                tokenizer_name="gpt2-medium",
                max_length=256,
                text_column="content",
                seed=123,
                padding="longest",
                truncation=False,
            )

            assert preparer.tokenizer_name == "gpt2-medium"
            assert preparer.max_length == 256
            assert preparer.text_column == "content"
            assert preparer.seed == 123
            assert preparer.padding == "longest"
            assert preparer.truncation is False

    def test_init_sets_pad_token_when_none(self):
        """Test that pad_token is set to eos_token when None."""
        with patch("src.data_preparation.AutoTokenizer.from_pretrained") as mock_tokenizer:
            mock_tok = MagicMock()
            mock_tok.pad_token = None
            mock_tok.eos_token = "<|endoftext|>"
            mock_tokenizer.return_value = mock_tok

            preparer = DataPreparer()

            assert preparer.tokenizer.pad_token == "<|endoftext|>"


class TestTextPreprocessing:
    """Tests for text preprocessing functionality."""

    def test_preprocess_text_strips_whitespace(self):
        """Test that preprocess_text strips leading/trailing whitespace."""
        with patch("src.data_preparation.AutoTokenizer.from_pretrained") as mock_tokenizer:
            mock_tokenizer.return_value = MagicMock()
            mock_tokenizer.return_value.pad_token = "<|endoftext|>"

            preparer = DataPreparer()

            result = preparer.preprocess_text("  hello world  ")
            assert result == "hello world"

    def test_preprocess_text_normalizes_whitespace(self):
        """Test that preprocess_text normalizes internal whitespace."""
        with patch("src.data_preparation.AutoTokenizer.from_pretrained") as mock_tokenizer:
            mock_tokenizer.return_value = MagicMock()
            mock_tokenizer.return_value.pad_token = "<|endoftext|>"

            preparer = DataPreparer()

            result = preparer.preprocess_text("hello   world")
            assert result == "hello world"

            result = preparer.preprocess_text("hello\tworld")
            assert result == "hello world"

    def test_preprocess_text_empty_string(self):
        """Test that preprocess_text handles empty strings."""
        with patch("src.data_preparation.AutoTokenizer.from_pretrained") as mock_tokenizer:
            mock_tokenizer.return_value = MagicMock()
            mock_tokenizer.return_value.pad_token = "<|endoftext|>"

            preparer = DataPreparer()

            result = preparer.preprocess_text("")
            assert result == ""


class TestTokenization:
    """Tests for tokenization functionality."""

    def test_tokenize_function_basic(self):
        """Test basic tokenization functionality."""
        with patch("src.data_preparation.AutoTokenizer.from_pretrained") as mock_tokenizer:
            mock_tok = MagicMock()
            mock_tok.pad_token = "<|endoftext|>"
            mock_tok.return_value = {
                "input_ids": [[1, 2, 3], [4, 5, 6]],
                "attention_mask": [[1, 1, 1], [1, 1, 1]],
            }
            mock_tokenizer.return_value = mock_tok

            preparer = DataPreparer()

            examples = {"text": ["hello world", "foo bar"]}
            result = preparer.tokenize_function(examples)

            assert "input_ids" in result
            assert "attention_mask" in result
            assert "labels" in result
            assert result["labels"] == result["input_ids"]

    def test_tokenize_function_adds_labels(self):
        """Test that tokenize_function adds labels equal to input_ids."""
        with patch("src.data_preparation.AutoTokenizer.from_pretrained") as mock_tokenizer:
            mock_tok = MagicMock()
            mock_tok.pad_token = "<|endoftext|>"
            mock_tok.return_value = {
                "input_ids": [[1, 2, 3]],
                "attention_mask": [[1, 1, 1]],
            }
            mock_tokenizer.return_value = mock_tok

            preparer = DataPreparer()

            examples = {"text": ["test text"]}
            result = preparer.tokenize_function(examples)

            assert result["labels"] == [[1, 2, 3]]


class TestDatasetCreation:
    """Tests for dataset creation functionality."""

    def test_create_sample_dataset_size(self):
        """Test that sample dataset has correct size."""
        with patch("src.data_preparation.AutoTokenizer.from_pretrained") as mock_tokenizer:
            mock_tokenizer.return_value = MagicMock()
            mock_tokenizer.return_value.pad_token = "<|endoftext|>"

            preparer = DataPreparer(seed=42)
            dataset = preparer.create_sample_dataset(num_samples=100)

            # 80% train, 20% test split
            assert len(dataset["train"]) == 80
            assert len(dataset["test"]) == 20

    def test_create_sample_dataset_has_text_column(self):
        """Test that sample dataset has 'text' column."""
        with patch("src.data_preparation.AutoTokenizer.from_pretrained") as mock_tokenizer:
            mock_tokenizer.return_value = MagicMock()
            mock_tokenizer.return_value.pad_token = "<|endoftext|>"

            preparer = DataPreparer(seed=42)
            dataset = preparer.create_sample_dataset(num_samples=10)

            assert "text" in dataset["train"].column_names

    def test_load_dataset_from_json_list_of_strings(self):
        """Test loading JSON file with list of strings."""
        with patch("src.data_preparation.AutoTokenizer.from_pretrained") as mock_tokenizer:
            mock_tokenizer.return_value = MagicMock()
            mock_tokenizer.return_value.pad_token = "<|endoftext|>"

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(["text one", "text two", "text three"], f)
                temp_path = f.name

            try:
                preparer = DataPreparer(seed=42)
                dataset = preparer.load_dataset_from_json(temp_path, train_test_split=0.5)

                assert "train" in dataset
                assert "test" in dataset
                assert len(dataset["train"]) + len(dataset["test"]) == 3
            finally:
                os.unlink(temp_path)

    def test_load_dataset_from_json_list_of_dicts(self):
        """Test loading JSON file with list of dicts."""
        with patch("src.data_preparation.AutoTokenizer.from_pretrained") as mock_tokenizer:
            mock_tokenizer.return_value = MagicMock()
            mock_tokenizer.return_value.pad_token = "<|endoftext|>"

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump([
                    {"text": "first text"},
                    {"text": "second text"},
                ], f)
                temp_path = f.name

            try:
                preparer = DataPreparer(seed=42)
                dataset = preparer.load_dataset_from_json(temp_path, train_test_split=0.5)

                assert "train" in dataset
                assert "text" in dataset["train"].column_names
            finally:
                os.unlink(temp_path)

    def test_load_dataset_from_json_with_custom_key(self):
        """Test loading JSON file with custom text key."""
        with patch("src.data_preparation.AutoTokenizer.from_pretrained") as mock_tokenizer:
            mock_tokenizer.return_value = MagicMock()
            mock_tokenizer.return_value.pad_token = "<|endoftext|>"

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump([
                    {"content": "first text"},
                    {"content": "second text"},
                ], f)
                temp_path = f.name

            try:
                preparer = DataPreparer(seed=42)
                dataset = preparer.load_dataset_from_json(temp_path, text_key="content", train_test_split=0.5)

                assert "train" in dataset
                assert "text" in dataset["train"].column_names
            finally:
                os.unlink(temp_path)


class TestInstructionDataset:
    """Tests for instruction dataset creation."""

    def test_create_instruction_dataset_basic(self):
        """Test basic instruction dataset creation."""
        instructions = [
            {"instruction": "Translate to French", "response": "Bonjour"},
            {"instruction": "Summarize", "response": "Summary here"},
        ]

        dataset = create_instruction_dataset(instructions)

        assert len(dataset) == 2
        assert "text" in dataset.column_names

    def test_create_instruction_dataset_custom_format(self):
        """Test instruction dataset with custom format."""
        instructions = [
            {"instruction": "Q: What is 2+2?", "response": "A: 4"},
        ]

        custom_format = "Q: {instruction}\nA: {response}"
        dataset = create_instruction_dataset(instructions, instruction_format=custom_format)

        assert len(dataset) == 1
        assert "Q: Q: What is 2+2?" in dataset[0]["text"]
        assert "A: A: 4" in dataset[0]["text"]

    def test_create_instruction_dataset_empty(self):
        """Test instruction dataset with empty list."""
        dataset = create_instruction_dataset([])
        assert len(dataset) == 0


class TestTokenizerMethods:
    """Tests for tokenizer-related methods."""

    def test_get_vocab_size(self):
        """Test getting vocabulary size."""
        with patch("src.data_preparation.AutoTokenizer.from_pretrained") as mock_tokenizer:
            mock_tok = MagicMock()
            mock_tok.pad_token = "<|endoftext|>"
            mock_tok.__len__ = MagicMock(return_value=50257)
            mock_tokenizer.return_value = mock_tok

            preparer = DataPreparer()

            vocab_size = preparer.get_vocab_size()
            assert vocab_size == 50257

    def test_decode(self):
        """Test decoding token IDs."""
        with patch("src.data_preparation.AutoTokenizer.from_pretrained") as mock_tokenizer:
            mock_tok = MagicMock()
            mock_tok.pad_token = "<|endoftext|>"
            mock_tok.decode = MagicMock(return_value="hello world")
            mock_tokenizer.return_value = mock_tok

            preparer = DataPreparer()

            result = preparer.decode([1, 2, 3])
            assert result == "hello world"
            mock_tok.decode.assert_called_once_with([1, 2, 3], skip_special_tokens=True)


class TestSaveProcessedDataset:
    """Tests for saving processed datasets."""

    def test_save_processed_dataset(self):
        """Test saving dataset to disk."""
        with patch("src.data_preparation.AutoTokenizer.from_pretrained") as mock_tokenizer:
            mock_tokenizer.return_value = MagicMock()
            mock_tokenizer.return_value.pad_token = "<|endoftext|>"

            preparer = DataPreparer(seed=42)
            dataset = preparer.create_sample_dataset(num_samples=10)

            with tempfile.TemporaryDirectory() as tmpdir:
                preparer.save_processed_dataset(dataset, tmpdir)
                # Check that files were created
                assert os.path.exists(os.path.join(tmpdir, "dataset_dict.json"))
