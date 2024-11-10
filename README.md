# BageLLM

A machine learning project focused on NLP benchmarking and evaluation.

## Project Structure

### Core Components

 BageLLM

A data quality and annotation pipeline for large language model training data. Cleans, validates, and enriches text datasets while enforcing strict quality standards.

## What It Does

### Text Cleaning
- Removes common data quality issues (boilerplate, formatting, encoding problems)
- Enforces consistent text standards
- Validates content quality
- Processes data in batches for efficiency

### Text Annotation 
- Adds NLP features (entities, sentiment, readability)
- Classifies content types
- Calculates quality metrics
- Enables dataset filtering based on quality scores

### Quality Control
- Tracks data quality metrics
- Validates processing results
- Provides quality score benchmarks
- Logs issues for review

#### `src/benchmarker.py`
The benchmarking system for evaluating NLP model performance. Features:
- Tracks processing time and memory usage
- Calculates data cleaning ratios and annotation coverage
- Computes comprehensive NLP quality metrics including:
  - Classification metrics (accuracy, precision, recall, F1)
  - Text generation metrics (BLEU, ROUGE, perplexity)
  - Inter-annotator agreement
- Saves benchmark results in JSON format

### Key Classes

#### `DatasetBenchmarker`
Main class for running benchmarks on NLP datasets. Provides:
- Benchmark timing functionality
- Memory usage tracking
- Quality metric calculations
- Results serialization

#### `NLPQualityMetrics`
Dataclass containing standard NLP evaluation metrics:
- Basic metrics: accuracy, precision, recall, F1
- Text generation: BLEU score, ROUGE scores
- Model evaluation: perplexity
- Annotation quality: inter-annotator agreement

#### `BenchmarkMetrics`
Dataclass for tracking overall benchmark performance:
- Processing time and memory usage
- Dataset size metrics (input/output)
- Data quality metrics (cleaning ratio, annotation coverage)
- NLP quality scores

## Installation

```bash
git clone https://github.com/BidhanRoy/BageLLM.git
cd BageLLM
pip install -r requirements.txt
```
