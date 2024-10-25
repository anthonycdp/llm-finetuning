"""
Evaluation Module for LLM Fine-Tuning Pipeline

This module handles:
- Model evaluation metrics (perplexity, BLEU, ROUGE)
- Baseline comparison (fine-tuned vs prompt-only)
- Benchmarking and reporting
- Qualitative text generation evaluation
"""

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    batch_size: int = 8
    max_new_tokens: int = 50
    num_samples: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    seed: int = 42


@dataclass
class EvaluationResults:
    """Container for evaluation results."""
    model_name: str
    perplexity: Optional[float] = None
    bleu_score: Optional[float] = None
    rouge_scores: Optional[Dict[str, float]] = None
    generation_samples: Optional[List[Dict[str, str]]] = None
    inference_time: Optional[float] = None
    memory_used_mb: Optional[float] = None
    additional_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "perplexity": self.perplexity,
            "bleu_score": self.bleu_score,
            "rouge_scores": self.rouge_scores,
            "inference_time": self.inference_time,
            "memory_used_mb": self.memory_used_mb,
            "additional_metrics": self.additional_metrics,
        }


class ModelEvaluator:
    """
    Comprehensive model evaluator for fine-tuned language models.

    Provides:
    - Perplexity calculation
    - Text generation quality assessment
    - Baseline comparison
    - Benchmark reporting
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: Optional[EvaluationConfig] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the evaluator.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer for the model
            config: Evaluation configuration
            device: Device to run on (auto-detected if None)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or EvaluationConfig()

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model.to(self.device)
        self.model.eval()

        # Generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=self.config.do_sample,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    def compute_perplexity(self, dataset: Dataset, batch_size: Optional[int] = None) -> float:
        """
        Compute perplexity on a dataset.

        Perplexity = exp(average negative log-likelihood)

        Args:
            dataset: Dataset to evaluate
            batch_size: Batch size for evaluation

        Returns:
            Perplexity score
        """
        batch_size = batch_size or self.config.batch_size
        logger.info(f"Computing perplexity on {len(dataset)} samples...")

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=data_collator,
            shuffle=False,
        )

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing perplexity"):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                # Accumulate loss (sum over batch)
                loss = outputs.loss
                batch_size_actual = input_ids.shape[0]

                # Count non-padding tokens
                if attention_mask is not None:
                    num_tokens = attention_mask.sum().item()
                else:
                    num_tokens = input_ids.numel()

                total_loss += loss.item() * batch_size_actual
                total_tokens += num_tokens

        # Compute perplexity
        avg_loss = total_loss / len(dataset)
        try:
            perplexity = math.exp(avg_loss)
        except OverflowError:
            # Cap at float max when avg_loss is too large
            perplexity = float("inf")
            logger.warning(f"Perplexity overflow (avg_loss={avg_loss:.2f}), capping to infinity")

        logger.info(f"Perplexity: {perplexity:.2f}")
        return perplexity

    def generate_samples(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        Generate text samples from prompts.

        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens to generate

        Returns:
            List of dicts with 'prompt' and 'generated' keys
        """
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        samples = []

        logger.info(f"Generating samples for {len(prompts)} prompts...")

        for prompt in tqdm(prompts, desc="Generating"):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    generation_config=self.generation_config,
                )

            generated = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            # Remove the prompt from generated text
            generated_only = generated[len(prompt):].strip()

            samples.append({
                "prompt": prompt,
                "generated": generated,
                "generated_only": generated_only,
            })

        return samples

    def evaluate_generation_quality(
        self,
        references: List[str],
        predictions: List[str],
    ) -> Dict[str, float]:
        """
        Evaluate generation quality using standard metrics.

        Computes BLEU and approximate ROUGE-like scores.

        Args:
            references: Reference texts
            predictions: Generated texts

        Returns:
            Dictionary of metric scores
        """
        metrics = {}

        # Simple BLEU-like score (unigram precision)
        try:
            bleu_score = self._compute_simple_bleu(references, predictions)
            metrics["bleu"] = bleu_score
        except Exception as e:
            logger.warning(f"Could not compute BLEU: {e}")

        # Simple ROUGE-like scores
        try:
            rouge_scores = self._compute_simple_rouge(references, predictions)
            metrics.update(rouge_scores)
        except Exception as e:
            logger.warning(f"Could not compute ROUGE: {e}")

        # Length statistics
        ref_lengths = [len(r.split()) for r in references]
        pred_lengths = [len(p.split()) for p in predictions]

        metrics["avg_reference_length"] = sum(ref_lengths) / len(ref_lengths) if ref_lengths else 0
        metrics["avg_prediction_length"] = sum(pred_lengths) / len(pred_lengths) if pred_lengths else 0

        return metrics

    def _compute_simple_bleu(self, references: List[str], predictions: List[str]) -> float:
        """
        Compute a simple BLEU-1 score (unigram precision).

        Args:
            references: Reference texts
            predictions: Generated texts

        Returns:
            BLEU-1 score
        """
        total_matches = 0
        total_pred_tokens = 0

        for ref, pred in zip(references, predictions):
            ref_tokens = set(ref.lower().split())
            pred_tokens = pred.lower().split()

            matches = sum(1 for t in pred_tokens if t in ref_tokens)
            total_matches += matches
            total_pred_tokens += len(pred_tokens)

        if total_pred_tokens == 0:
            return 0.0

        return total_matches / total_pred_tokens

    def _compute_simple_rouge(
        self,
        references: List[str],
        predictions: List[str],
    ) -> Dict[str, float]:
        """
        Compute simple ROUGE scores.

        Args:
            references: Reference texts
            predictions: Generated texts

        Returns:
            Dictionary with ROUGE-1, ROUGE-2, ROUGE-L scores
        """
        rouge_1_scores = []
        rouge_2_scores = []

        for ref, pred in zip(references, predictions):
            ref_tokens = ref.lower().split()
            pred_tokens = pred.lower().split()

            if not ref_tokens or not pred_tokens:
                continue

            # ROUGE-1: Unigram overlap
            ref_unigrams = set(ref_tokens)
            pred_unigrams = set(pred_tokens)
            overlap_1 = len(ref_unigrams & pred_unigrams)
            rouge_1 = overlap_1 / len(ref_unigrams) if ref_unigrams else 0
            rouge_1_scores.append(rouge_1)

            # ROUGE-2: Bigram overlap
            ref_bigrams = set(zip(ref_tokens[:-1], ref_tokens[1:]))
            pred_bigrams = set(zip(pred_tokens[:-1], pred_tokens[1:]))
            if ref_bigrams:
                overlap_2 = len(ref_bigrams & pred_bigrams)
                rouge_2 = overlap_2 / len(ref_bigrams)
                rouge_2_scores.append(rouge_2)

        return {
            "rouge_1": sum(rouge_1_scores) / len(rouge_1_scores) if rouge_1_scores else 0,
            "rouge_2": sum(rouge_2_scores) / len(rouge_2_scores) if rouge_2_scores else 0,
        }

    def benchmark_inference(
        self,
        prompts: List[str],
        num_runs: int = 3,
    ) -> Dict[str, float]:
        """
        Benchmark inference speed and memory usage.

        Args:
            prompts: Prompts to use for benchmarking
            num_runs: Number of runs for timing

        Returns:
            Dictionary with timing and memory metrics
        """
        logger.info(f"Benchmarking inference with {num_runs} runs...")

        # Warm up
        warmup_prompt = prompts[0] if prompts else "Hello world"
        inputs = self.tokenizer(warmup_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            _ = self.model.generate(**inputs, max_new_tokens=20)

        # Time multiple runs
        times = []
        for _ in range(num_runs):
            prompt = prompts[_ % len(prompts)] if prompts else "Test"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.perf_counter()

            with torch.no_grad():
                _ = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                )

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.perf_counter()

            times.append(end - start)

        # Memory usage
        memory_mb = 0
        if torch.cuda.is_available():
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

        return {
            "avg_inference_time": sum(times) / len(times),
            "min_inference_time": min(times),
            "max_inference_time": max(times),
            "memory_used_mb": memory_mb,
        }

    def full_evaluation(
        self,
        eval_dataset: Dataset,
        prompts: Optional[List[str]] = None,
        references: Optional[List[str]] = None,
        model_name: str = "Model",
    ) -> EvaluationResults:
        """
        Run full evaluation suite.

        Args:
            eval_dataset: Dataset for perplexity calculation
            prompts: Prompts for generation evaluation
            references: References for quality metrics
            model_name: Name for reporting

        Returns:
            EvaluationResults object
        """
        logger.info(f"Starting full evaluation for {model_name}")

        results = EvaluationResults(model_name=model_name)

        # Compute perplexity
        try:
            results.perplexity = self.compute_perplexity(eval_dataset)
        except Exception as e:
            logger.error(f"Perplexity computation failed: {e}")

        # Generate samples
        if prompts:
            results.generation_samples = self.generate_samples(prompts[:self.config.num_samples])

            # Evaluate quality if references provided
            if references:
                predictions = [s["generated_only"] for s in results.generation_samples]
                ref_subset = references[:len(predictions)]
                quality_metrics = self.evaluate_generation_quality(ref_subset, predictions)
                results.bleu_score = quality_metrics.get("bleu")
                results.rouge_scores = {
                    k: v for k, v in quality_metrics.items()
                    if k.startswith("rouge")
                }
                results.additional_metrics.update({
                    k: v for k, v in quality_metrics.items()
                    if not k.startswith("rouge") and k != "bleu"
                })

        # Benchmark inference
        if prompts:
            benchmark_results = self.benchmark_inference(prompts)
            results.inference_time = benchmark_results["avg_inference_time"]
            results.memory_used_mb = benchmark_results["memory_used_mb"]

        return results


class BaselineComparer:
    """
    Compare fine-tuned model against baseline (prompt-only) approach.

    This class enables fair comparison between:
    - Fine-tuned model
    - Original/base model (prompt-only baseline)
    """

    def __init__(
        self,
        fine_tuned_model: PreTrainedModel,
        base_model_name: str,
        tokenizer: PreTrainedTokenizer,
        config: Optional[EvaluationConfig] = None,
    ):
        """
        Initialize the comparer.

        Args:
            fine_tuned_model: The fine-tuned model
            base_model_name: Name/path of the base model for comparison
            tokenizer: Tokenizer
            config: Evaluation configuration
        """
        self.fine_tuned_model = fine_tuned_model
        self.base_model_name = base_model_name
        self.tokenizer = tokenizer
        self.config = config or EvaluationConfig()

        # Load base model
        logger.info(f"Loading base model: {base_model_name}")
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

    def compare(
        self,
        eval_dataset: Dataset,
        prompts: List[str],
        references: Optional[List[str]] = None,
        fine_tuned_name: str = "Fine-tuned",
        base_name: str = "Base (Prompt-only)",
    ) -> Dict[str, Any]:
        """
        Compare fine-tuned model against baseline.

        Args:
            eval_dataset: Dataset for evaluation
            prompts: Prompts for generation
            references: Reference texts for quality metrics
            fine_tuned_name: Name for fine-tuned model
            base_name: Name for base model

        Returns:
            Comparison results dictionary
        """
        logger.info("Starting baseline comparison...")

        # Evaluate fine-tuned model
        ft_evaluator = ModelEvaluator(
            self.fine_tuned_model,
            self.tokenizer,
            self.config,
        )
        ft_results = ft_evaluator.full_evaluation(
            eval_dataset,
            prompts,
            references,
            fine_tuned_name,
        )

        # Evaluate base model
        base_evaluator = ModelEvaluator(
            self.base_model,
            self.tokenizer,
            self.config,
        )
        base_results = base_evaluator.full_evaluation(
            eval_dataset,
            prompts,
            references,
            base_name,
        )

        # Compile comparison
        comparison = {
            "fine_tuned": ft_results.to_dict(),
            "baseline": base_results.to_dict(),
            "comparison": {},
        }

        # Compute improvements
        if ft_results.perplexity and base_results.perplexity:
            ppl_improvement = (base_results.perplexity - ft_results.perplexity) / base_results.perplexity * 100
            comparison["comparison"]["perplexity_improvement_pct"] = ppl_improvement
            comparison["comparison"]["perplexity_delta"] = base_results.perplexity - ft_results.perplexity

        if ft_results.bleu_score and base_results.bleu_score:
            bleu_improvement = ft_results.bleu_score - base_results.bleu_score
            comparison["comparison"]["bleu_improvement"] = bleu_improvement

        if ft_results.rouge_scores and base_results.rouge_scores:
            for metric in ft_results.rouge_scores:
                if metric in base_results.rouge_scores:
                    delta = ft_results.rouge_scores[metric] - base_results.rouge_scores[metric]
                    comparison["comparison"][f"{metric}_delta"] = delta

        # Generation comparison
        if ft_results.generation_samples and base_results.generation_samples:
            comparison["generation_comparison"] = []
            for i, (ft_sample, base_sample) in enumerate(
                zip(ft_results.generation_samples, base_results.generation_samples)
            ):
                comparison["generation_comparison"].append({
                    "prompt": ft_sample["prompt"],
                    "fine_tuned_output": ft_sample["generated_only"],
                    "baseline_output": base_sample["generated_only"],
                })

        return comparison

    def generate_report(self, comparison: Dict[str, Any]) -> str:
        """
        Generate a human-readable comparison report.

        Args:
            comparison: Comparison results dictionary

        Returns:
            Formatted report string
        """
        report_lines = [
            "=" * 60,
            "FINE-TUNING EVALUATION REPORT",
            "=" * 60,
            "",
            "PERPLEXITY COMPARISON",
            "-" * 40,
        ]

        ft_ppl = comparison["fine_tuned"].get("perplexity")
        base_ppl = comparison["baseline"].get("perplexity")

        if ft_ppl and base_ppl:
            report_lines.extend([
                f"Fine-tuned Model: {ft_ppl:.2f}",
                f"Baseline Model:   {base_ppl:.2f}",
                f"Improvement:      {comparison['comparison'].get('perplexity_improvement_pct', 0):.1f}%",
                "",
            ])

        # BLEU scores
        ft_bleu = comparison["fine_tuned"].get("bleu_score")
        base_bleu = comparison["baseline"].get("bleu_score")

        if ft_bleu is not None and base_bleu is not None:
            report_lines.extend([
                "BLEU SCORE COMPARISON",
                "-" * 40,
                f"Fine-tuned Model: {ft_bleu:.4f}",
                f"Baseline Model:   {base_bleu:.4f}",
                f"Improvement:      {comparison['comparison'].get('bleu_improvement', 0):.4f}",
                "",
            ])

        # ROUGE scores
        ft_rouge = comparison["fine_tuned"].get("rouge_scores", {})
        base_rouge = comparison["baseline"].get("rouge_scores", {})

        if ft_rouge and base_rouge:
            report_lines.extend([
                "ROUGE SCORES COMPARISON",
                "-" * 40,
            ])
            for metric in sorted(ft_rouge.keys()):
                ft_val = ft_rouge.get(metric, 0)
                base_val = base_rouge.get(metric, 0)
                delta = comparison["comparison"].get(f"{metric}_delta", 0)
                report_lines.append(
                    f"{metric.upper()}: FT={ft_val:.4f}, Base={base_val:.4f}, Delta={delta:+.4f}"
                )
            report_lines.append("")

        # Sample generations
        gen_comparison = comparison.get("generation_comparison", [])
        if gen_comparison:
            report_lines.extend([
                "SAMPLE GENERATIONS",
                "-" * 40,
            ])
            for i, sample in enumerate(gen_comparison[:3]):  # Show first 3
                report_lines.extend([
                    f"\nPrompt: {sample['prompt'][:100]}...",
                    f"\nFine-tuned: {sample['fine_tuned_output'][:200]}",
                    f"\nBaseline:   {sample['baseline_output'][:200]}",
                    "",
                ])

        report_lines.append("=" * 60)

        return "\n".join(report_lines)


def save_comparison_report(
    comparison: Dict[str, Any],
    output_path: Union[str, Path],
) -> None:
    """
    Save comparison results to JSON file.

    Args:
        comparison: Comparison results
        output_path: Path to save
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, default=str)

    logger.info(f"Comparison report saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load models
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create sample evaluation data
    sample_texts = [
        {"input_ids": tokenizer(f"Review: {text}", return_tensors="pt")["input_ids"][0].tolist(),
         "labels": tokenizer(f"Review: {text}", return_tensors="pt")["input_ids"][0].tolist()}
        for text in ["This is great!", "Not recommended.", "Amazing product!"]
    ]
    eval_dataset = Dataset.from_list(sample_texts)

    # Evaluate
    config = EvaluationConfig(num_samples=3, max_new_tokens=20)
    evaluator = ModelEvaluator(model, tokenizer, config)

    prompts = ["The movie was", "I think this", "My favorite"]
    results = evaluator.full_evaluation(eval_dataset, prompts, model_name="distilgpt2")

    print(f"\nEvaluation Results for {results.model_name}:")
    print(f"Perplexity: {results.perplexity:.2f}")
    print(f"\nGenerated samples:")
    for sample in results.generation_samples[:3]:
        print(f"  Prompt: {sample['prompt']}")
        print(f"  Generated: {sample['generated_only']}")
        print()
