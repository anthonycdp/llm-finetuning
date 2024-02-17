"""
Data Preparation Module for LLM Fine-Tuning Pipeline

This module handles:
- Loading datasets from Hugging Face Hub or local files
- Text preprocessing and cleaning
- Tokenization with proper padding and truncation
- Train/validation/test splitting
- Data collation for language modeling
"""

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

logger = logging.getLogger(__name__)


class DataPreparer:
    """
    Handles all data preparation tasks for fine-tuning language models.

    Attributes:
        tokenizer: The tokenizer to use for text processing
        max_length: Maximum sequence length for tokenization
        text_column: Name of the column containing text data
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        tokenizer_name: str = "gpt2",
        max_length: int = 512,
        text_column: str = "text",
        seed: int = 42,
        padding: str = "max_length",
        truncation: bool = True,
    ):
        """
        Initialize the DataPreparer.

        Args:
            tokenizer_name: Name or path of the pretrained tokenizer
            max_length: Maximum sequence length for tokenization
            text_column: Name of the column containing text data
            seed: Random seed for reproducibility
            padding: Padding strategy ('max_length', 'longest', or False)
            truncation: Whether to truncate sequences
        """
        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.text_column = text_column
        self.seed = seed
        self.padding = padding
        self.truncation = truncation

        # Ensure pad token is set (GPT-2 doesn't have one by default)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token: {self.tokenizer.eos_token}")

    def load_dataset_from_hub(
        self,
        dataset_name: str,
        subset: Optional[str] = None,
        split: Optional[str] = None,
        **kwargs,
    ) -> Union[Dataset, DatasetDict]:
        """
        Load a dataset from the Hugging Face Hub.

        Args:
            dataset_name: Name of the dataset on the Hub
            subset: Configuration/subset name if applicable
            split: Specific split to load (e.g., 'train', 'test')
            **kwargs: Additional arguments passed to load_dataset

        Returns:
            Dataset or DatasetDict object
        """
        logger.info(f"Loading dataset '{dataset_name}' from Hugging Face Hub")
        dataset = load_dataset(dataset_name, subset, split=split, **kwargs)
        logger.info(f"Loaded dataset with splits: {dataset.keys() if isinstance(dataset, DatasetDict) else split}")
        return dataset

    def load_dataset_from_json(
        self,
        file_path: Union[str, Path],
        text_key: str = "text",
        train_test_split: float = 0.1,
    ) -> DatasetDict:
        """
        Load a dataset from a JSON file.

        Expected format:
        - Single file with list of texts or dicts with text_key
        - Or line-delimited JSON (JSONL)

        Args:
            file_path: Path to the JSON file
            text_key: Key to extract text from if data is dict format
            train_test_split: Fraction of data to use for validation

        Returns:
            DatasetDict with 'train' and 'test' splits
        """
        file_path = Path(file_path)
        logger.info(f"Loading dataset from {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            if file_path.suffix == ".jsonl":
                data = [json.loads(line) for line in f if line.strip()]
            else:
                data = json.load(f)

        # Extract text from dicts if necessary
        if isinstance(data[0], dict):
            if text_key in data[0]:
                data = [{"text": item[text_key]} for item in data]
            elif self.text_column in data[0]:
                pass  # Already has correct column
            else:
                raise ValueError(f"Could not find text key. Available keys: {data[0].keys()}")
        else:
            # Assume list of strings
            data = [{"text": item} for item in data]

        dataset = Dataset.from_list(data)
        dataset = dataset.train_test_split(test_size=train_test_split, seed=self.seed)

        logger.info(f"Created dataset with {len(dataset['train'])} train and {len(dataset['test'])} test samples")
        return dataset

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess a single text sample.

        Args:
            text: Raw text string

        Returns:
            Preprocessed text string
        """
        # Basic cleaning
        text = text.strip()
        # Normalize whitespace
        text = " ".join(text.split())
        return text

    def tokenize_function(
        self,
        examples: Dict[str, List[Any]],
        add_special_tokens: bool = True,
    ) -> Dict[str, List[Any]]:
        """
        Tokenize a batch of examples.

        Args:
            examples: Dictionary containing text examples
            add_special_tokens: Whether to add special tokens

        Returns:
            Dictionary with tokenized inputs
        """
        texts = [self.preprocess_text(text) for text in examples[self.text_column]]

        tokenized = self.tokenizer(
            texts,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            add_special_tokens=add_special_tokens,
            return_attention_mask=True,
        )

        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    def prepare_dataset(
        self,
        dataset: Union[Dataset, DatasetDict],
        remove_columns: Optional[List[str]] = None,
        batched: bool = True,
        num_proc: Optional[int] = None,
    ) -> DatasetDict:
        """
        Prepare a dataset for training by tokenizing.

        Args:
            dataset: Dataset to prepare
            remove_columns: Columns to remove after tokenization
            batched: Whether to process in batches
            num_proc: Number of processes for parallel processing

        Returns:
            Tokenized DatasetDict ready for training
        """
        logger.info("Preparing dataset for training...")

        if isinstance(dataset, Dataset):
            dataset = DatasetDict({"train": dataset})

        if remove_columns is None:
            remove_columns = list(dataset["train"].column_names)

        tokenized_dataset = DatasetDict()
        for split_name, split_dataset in dataset.items():
            logger.info(f"Tokenizing {split_name} split ({len(split_dataset)} samples)...")

            tokenized_split = split_dataset.map(
                self.tokenize_function,
                batched=batched,
                remove_columns=remove_columns,
                num_proc=num_proc,
                desc=f"Tokenizing {split_name}",
            )
            tokenized_dataset[split_name] = tokenized_split

        logger.info(f"Dataset prepared. Splits: {list(tokenized_dataset.keys())}")
        return tokenized_dataset

    def create_sample_dataset(self, num_samples: int = 1000) -> DatasetDict:
        """
        Create a sample dataset for testing the pipeline.

        This creates synthetic data for demonstration purposes.
        For real training, use load_dataset_from_hub or load_dataset_from_json.

        Args:
            num_samples: Number of samples to generate

        Returns:
            DatasetDict with train/test splits
        """
        logger.info(f"Creating sample dataset with {num_samples} examples")

        positive_templates = [
            "This product is amazing! I love it so much.",
            "Great quality and fast shipping. Highly recommended!",
            "Excellent customer service. Very satisfied with my purchase.",
            "Best decision I ever made. Works perfectly!",
            "Wonderful experience from start to finish.",
            "Five stars! Exceeded all my expectations.",
            "Absolutely fantastic! Will buy again.",
            "Outstanding quality for the price.",
            "I'm impressed! This is exactly what I needed.",
            "Superb! Couldn't be happier with this purchase.",
        ]

        negative_templates = [
            "Terrible product. Complete waste of money.",
            "Very disappointed. Quality is poor.",
            "Would not recommend. Broke after one use.",
            "Horrible customer service. Never again.",
            "Not worth the price. Very cheaply made.",
            "One star. Does not work as advertised.",
            "Frustrated with this purchase. Want a refund.",
            "Worst product I've ever bought.",
            "Avoid! Complete disappointment.",
            "Very unsatisfied. Save your money.",
        ]

        random.seed(self.seed)

        texts = []
        for _ in range(num_samples):
            template = random.choice(positive_templates if random.random() > 0.5 else negative_templates)
            texts.append({"text": template})

        dataset = Dataset.from_list(texts)
        dataset = dataset.train_test_split(test_size=0.2, seed=self.seed)

        logger.info(f"Created sample dataset with {len(dataset['train'])} train and {len(dataset['test'])} test samples")
        return dataset

    def save_processed_dataset(self, dataset: DatasetDict, output_dir: Union[str, Path]) -> None:
        """
        Save a processed dataset to disk.

        Args:
            dataset: DatasetDict to save
            output_dir: Directory to save to
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset.save_to_disk(str(output_dir))
        logger.info(f"Dataset saved to {output_dir}")

    def get_tokenizer(self) -> PreTrainedTokenizer:
        """Get the configured tokenizer."""
        return self.tokenizer

    def get_vocab_size(self) -> int:
        """Get the vocabulary size of the tokenizer."""
        return len(self.tokenizer)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            Decoded text string
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def create_instruction_dataset(
    instructions: List[Dict[str, str]],
    instruction_format: str = "### Instruction:\n{instruction}\n\n### Response:\n{response}",
) -> Dataset:
    """
    Create an instruction-tuning dataset from instruction-response pairs.

    Args:
        instructions: List of dicts with 'instruction' and 'response' keys
        instruction_format: Format string for combining instruction and response

    Returns:
        Dataset with formatted text
    """
    formatted_texts = []
    for item in instructions:
        formatted_text = instruction_format.format(
            instruction=item.get("instruction", ""),
            response=item.get("response", ""),
        )
        formatted_texts.append({"text": formatted_text})

    return Dataset.from_list(formatted_texts)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Initialize data preparer
    preparer = DataPreparer(
        tokenizer_name="gpt2",
        max_length=256,
        text_column="text",
    )

    # Create sample dataset
    dataset = preparer.create_sample_dataset(num_samples=100)

    # Prepare dataset
    tokenized_dataset = preparer.prepare_dataset(dataset)

    print(f"\nDataset prepared successfully!")
    print(f"Train samples: {len(tokenized_dataset['train'])}")
    print(f"Test samples: {len(tokenized_dataset['test'])}")

    # Show example
    example = tokenized_dataset["train"][0]
    print(f"\nExample tokenized sample:")
    print(f"Input IDs shape: {len(example['input_ids'])}")
    print(f"Decoded: {preparer.decode(example['input_ids'][:50])}")
