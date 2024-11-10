from dataclasses import dataclass
from typing import Dict, Any, Optional
import time
import pandas as pd
import numpy as np
from pathlib import Path
import json
import torch

@dataclass
class NLPQualityMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    bleu_score: Optional[float] = None
    rouge_scores: Optional[Dict[str, float]] = None
    perplexity: Optional[float] = None
    inter_annotator_agreement: Optional[float] = None

@dataclass
class BenchmarkMetrics:
    processing_time: float
    memory_usage: float
    input_size: int
    output_size: int
    cleaning_ratio: float
    annotation_coverage: float
    quality_scores: NLPQualityMetrics

class DatasetBenchmarker:
    def __init__(self, output_dir: Path, reference_data: Optional[pd.DataFrame] = None):
        self.output_dir = output_dir
        self.reference_data = reference_data
        self.start_time = None
        self.metrics = {}

    def start_benchmark(self):
        self.start_time = time.time()

    def end_benchmark(self, dataset_name: str, original_df: pd.DataFrame, 
                     processed_df: pd.DataFrame, quality_metrics: Dict[str, Any] = None):
        if not self.start_time:
            raise ValueError("Benchmark not started")

        processing_time = time.time() - self.start_time
        
        # Calculate metrics
        metrics = BenchmarkMetrics(
            processing_time=processing_time,
            memory_usage=processed_df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
            input_size=len(original_df),
            output_size=len(processed_df),
            cleaning_ratio=len(processed_df) / len(original_df),
            annotation_coverage=self._calculate_annotation_coverage(processed_df),
            quality_scores=self.calculate_nlp_quality_metrics(processed_df)
        )

        # Convert quality metrics to serializable format
        if quality_metrics:
            serializable_metrics = {}
            for column, metrics in quality_metrics.items():
                serializable_metrics[column] = {
                    k: float(v) if isinstance(v, (int, float)) else str(v)
                    for k, v in metrics.items()
                }
            self.metrics[dataset_name] = {
                **self._metrics_to_dict(metrics),
                'column_quality_metrics': serializable_metrics
            }
        else:
            self.metrics[dataset_name] = self._metrics_to_dict(metrics)

        self._save_benchmark_results()
        
        return metrics

    def _calculate_annotation_coverage(self, df: pd.DataFrame) -> float:
        """Calculate the percentage of texts that were successfully annotated"""
        annotation_columns = [col for col in df.columns if col.endswith('_annotations')]
        if not annotation_columns:
            return 0.0
            
        coverage_scores = []
        for col in annotation_columns:
            has_annotations = df[col].apply(lambda x: bool(x) and len(x) > 0)
            coverage_scores.append(has_annotations.mean())
            
        return np.mean(coverage_scores)

    def _metrics_to_dict(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Convert metrics dictionary to a JSON-serializable format"""
        # If metrics is already a dict, ensure all values are serializable
        if isinstance(metrics, dict):
            return {
                'processing_time_seconds': metrics.get('processing_time', 0),
                'memory_usage_mb': metrics.get('memory_usage', 0),
                'input_records': metrics.get('input_size', 0),
                'output_records': metrics.get('output_size', 0),
                'cleaning_ratio': metrics.get('cleaning_ratio', 0),
                'annotation_coverage': metrics.get('annotation_coverage', 0),
                'quality_scores': {
                    'accuracy': metrics.get('accuracy', 0),
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'f1_score': metrics.get('f1_score', 0),
                    'bleu_score': metrics.get('bleu_score'),
                    'rouge_scores': metrics.get('rouge_scores'),
                    'perplexity': metrics.get('perplexity'),
                    'inter_annotator_agreement': metrics.get('inter_annotator_agreement')
                }
            }
        
        # If metrics is a BenchmarkMetrics object (the original case)
        quality_scores_dict = {
            'accuracy': metrics.quality_scores.accuracy,
            'precision': metrics.quality_scores.precision,
            'recall': metrics.quality_scores.recall,
            'f1_score': metrics.quality_scores.f1_score,
            'bleu_score': metrics.quality_scores.bleu_score,
            'rouge_scores': metrics.quality_scores.rouge_scores,
            'perplexity': metrics.quality_scores.perplexity,
            'inter_annotator_agreement': metrics.quality_scores.inter_annotator_agreement
        }

        return {
            'processing_time_seconds': metrics.processing_time,
            'memory_usage_mb': metrics.memory_usage,
            'input_records': metrics.input_size,
            'output_records': metrics.output_size,
            'cleaning_ratio': metrics.cleaning_ratio,
            'annotation_coverage': metrics.annotation_coverage,
            'quality_scores': quality_scores_dict
        }

    def _save_benchmark_results(self):
        """Save benchmark results to JSON file with custom serialization"""
        benchmark_file = self.output_dir / 'benchmark_results.json'
        # Ensure all values are JSON serializable
        serializable_metrics = {}
        for dataset, metrics in self.metrics.items():
            serializable_metrics[dataset] = {
                k: float(v) if isinstance(v, (int, float)) else v
                for k, v in metrics.items()
            }
        
        with open(benchmark_file, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)

    def calculate_nlp_quality_metrics(self, processed_df: pd.DataFrame) -> NLPQualityMetrics:
        """Calculate comprehensive NLP quality metrics"""
        # Return default metrics if no reference data is available
        if self.reference_data is None:
            return NLPQualityMetrics(
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                bleu_score=None,
                rouge_scores=None,
                perplexity=None,
                inter_annotator_agreement=None
            )

        metrics = {}
        
        # Basic classification metrics
        metrics.update(self._calculate_classification_metrics(processed_df))
        
        # Text generation metrics (if applicable)
        metrics.update(self._calculate_generation_metrics(processed_df))
        
        # Inter-annotator agreement
        metrics['inter_annotator_agreement'] = self._calculate_iaa(processed_df)
        
        return NLPQualityMetrics(**metrics)

    def _calculate_classification_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate precision, recall, F1, and accuracy for classification tasks"""
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
        
        # Assuming columns ending with '_predictions' contain model outputs
        metrics = {}
        for col in [c for c in df.columns if c.endswith('_predictions')]:
            ref_col = col.replace('_predictions', '_ground_truth')
            if ref_col not in self.reference_data.columns:
                continue
                
            y_true = self.reference_data[ref_col]
            y_pred = df[col]
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='weighted'
            )
            accuracy = accuracy_score(y_true, y_pred)
            
            metrics.update({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
            
        return metrics

    def _calculate_generation_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate BLEU, ROUGE, and perplexity for text generation tasks"""
        from nltk.translate.bleu_score import corpus_bleu
        from rouge_score import rouge_scorer
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        
        metrics = {}
        
        # Text generation columns (if any)
        gen_cols = [c for c in df.columns if c.endswith('_generated')]
        if not gen_cols:
            return metrics
            
        for col in gen_cols:
            ref_col = col.replace('_generated', '_reference')
            if ref_col not in self.reference_data.columns:
                continue
                
            # Calculate BLEU score
            references = self.reference_data[ref_col].apply(str.split).tolist()
            hypotheses = df[col].apply(str.split).tolist()
            metrics['bleu_score'] = corpus_bleu([[r] for r in references], hypotheses)
            
            # Calculate ROUGE scores
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
            rouge_scores = {
                f"rouge_{k}": np.mean([
                    scorer.score(ref, hyp)[v].fmeasure
                    for ref, hyp in zip(self.reference_data[ref_col], df[col])
                ])
                for k, v in [('1', 'rouge1'), ('2', 'rouge2'), ('l', 'rougeL')]
            }
            metrics['rouge_scores'] = rouge_scores
            
            # Calculate perplexity using GPT-2
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            model = GPT2LMHeadModel.from_pretrained('gpt2')
            metrics['perplexity'] = self._calculate_perplexity(df[col], model, tokenizer)
            
        return metrics

    def _calculate_iaa(self, df: pd.DataFrame) -> float:
        """Calculate inter-annotator agreement using Krippendorff's alpha"""
        from krippendorff import alpha
        
        # Look for columns ending with '_annotations'
        annotator_cols = [c for c in df.columns if c.endswith('_annotations')]
        if len(annotator_cols) < 2:
            return None
            
        # Convert annotations to a format suitable for Krippendorff's alpha
        reliability_data = df[annotator_cols].values.T
        return alpha(reliability_data=reliability_data)

    def _calculate_perplexity(self, texts: pd.Series, model, tokenizer) -> float:
        """Calculate average perplexity across texts"""
        perplexities = []
        for text in texts:
            inputs = tokenizer(text, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs['input_ids'])
                perplexities.append(torch.exp(outputs.loss).item())
        return np.mean(perplexities)