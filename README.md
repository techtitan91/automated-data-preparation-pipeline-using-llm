# automated-data-preparation-pipeline-using-llm

This repository holds the codebase for **automated-data-preparation-pipeline-using-llm**, an automated workflow for preparing data, specifically refined for Llama 3.2 fine-tuning.

## Project Organization

```
automated-data-preparation-pipeline-using-llm/
├── src/
│   ├── __init__.py
│   ├── annotator.py
│   ├── cleaner.py
│   ├── pipeline.py
│   ├── quality_control/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── metrics.py
│   │   └── rules.py
│   ├── annotation_system/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── processors.py
│   └── s3_connector.py
├── tests/
│   ├── __init__.py
│   ├── test_cleaner.py
│   ├── test_annotation_system.py
│   └── test_quality_control.py
├── config/
│   └── quality_config.json
├── requirements.txt
└── run.py
```

## Setup Guide

1.  Establish and activate a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2.  Install necessary dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3.  Install requisite pre-trained models:
    ```bash
    python -m spacy download en_core_web_sm
    ```

## Configuration Details

1.  Configure AWS credentials for S3 bucket access:

    ```bash
    # ~/.aws/credentials
    [default]
    aws_access_key_id = your_access_key_value
    aws_secret_access_key = your_secret_key_value
    ```

2.  Define the quality control settings (in `config/quality_config.json`):
    ```json
    {
      "min_completeness": 0.8,
      "min_consistency": 0.7,
      "min_validity": 0.8,
      "min_uniqueness": 0.9,
      "max_data_age_days": 30,
      "min_accuracy": 0.8,
      "text_length_threshold": 10,
      "min_overall_score": 0.75,
      "output_dir": "quality_reports"
    }
    ```

## Core Components

### 1. Annotation Service

The annotation service offers in-depth text analysis:

```python
from src.annotation_system import AnnotationSystem, AnnotationConfig

# Initialization
config_obj = AnnotationConfig(
    enable_entities=True,
    enable_sentiment=True,
    enable_topics=True,
    enable_keywords=True,
    enable_language=True
)
annotation_svc = AnnotationSystem(config_obj)

# Analyze a single text string
text_annotations = annotation_svc.annotate_text("Your sample text goes here.")

# Analyze a batch of texts
batch_annotations = annotation_svc.process_batch(["Text sample 1", "Text sample 2"])
```

**Capabilities:**

- Named Entity Recognition
- Sentiment Analysis
- Topic Detection
- Keyword Extraction
- Language Identification

### 2. Quality Assurance System

The quality assurance system maintains data integrity across multiple dimensions:

```python
from src.quality_control import QualityController, SpecialCharacterRule

# Initialization
qc_controller = QualityController()

# Assess data validity
passed_check, identified_issues, quality_metrics = qc_controller.validate_data(your_dataframe)

# Apply specific validation rules
char_rule = SpecialCharacterRule()
rule_result = char_rule.validate(your_dataframe)
```

**Quality Dimensions Assessed:**

- Completeness
- Consistency
- Validity
- Uniqueness
- Timeliness
- Integrity
- Accuracy

### 3. Data Cleansing Pipeline

The data cleansing pipeline processes and refines the data:

```python
from src.pipeline import AnnotationPipeline

data_proc_pipeline = AnnotationPipeline()
data_proc_pipeline.process_datasets()
```

**Features:**

- Text sanitization
- Standardization of formats
- Quality assessment
- Batch processing capabilities

## Usage Examples

### 1. End-to-End Pipeline Execution

```python
from src.pipeline import AnnotationPipeline

# Initialize and execute the full pipeline
main_pipeline_instance = AnnotationPipeline(output_dir="processed_data_output")
main_pipeline_instance.process_datasets()
```

### 2. Quality Control Module

```python
from src.quality_control import QualityController
import pandas as pd

# Import data
source_data = pd.read_csv("your_input_data.csv")

# Initialize the controller
quality_checker = QualityController()

# Validate the dataset
validation_status, detected_issues, performance_metrics = quality_checker.validate_data(source_data)

# Review the results
print(f"Validation Passed: {validation_status}")
print(f"Detected Issues: {detected_issues}")
print(f"Overall Quality Score: {performance_metrics.overall_score}")
```

### 3. Annotation Service Module

```python
from src.annotation_system import AnnotationSystem

# Initialize the system
text_analyzer = AnnotationSystem()

# Process a text sample
input_text = "Apple Inc. is rumored to be releasing a new smartphone model next quarter."
analysis_output = text_analyzer.annotate_text(input_text)

# Display analysis results
print("Identified Entities:", analysis_output['entities'])
print("Detected Sentiment:", analysis_output['sentiment'])
print("Inferred Topics:", analysis_output['topics'])
```

## Testing Procedures

Execute all available tests:

```bash
python -m unittest discover tests
```

Run specific test suites:

```bash
python -m unittest tests/test_cleaner.py
python -m unittest tests/test_annotation_system.py
python -m unittest tests/test_quality_control.py
```

## Output Directory Layout

### Quality Assurance Reports

```
quality_reports/
├── quality_report_dataset1_20241110_123456.json
├── quality_report_dataset1_20241110_123456.txt
├── quality_report_dataset2_20241110_123457.json
└── quality_report_dataset2_20241110_123457.txt
```

### Processed Data Output

```
processed_data/
├── processed_dataset1.csv
├── processed_dataset1_metrics.json
├── processed_dataset2.csv
└── processed_dataset2_metrics.json
```

## Development Milestones

### Day 1 ✅

- [x] Environment configuration
- [x] Core infrastructure setup
- [x] Basic data cleansing pipeline
- [x] Initial test cases

### Day 2 ✅

- [x] Annotation service implementation
- [x] Quality control metric definition
- [x] Validation rule creation
- [x] Integration testing

## Contribution Guidelines

1.  Fork the main repository.
2.  Create your dedicated feature branch.
3.  Commit your implemented changes.
4.  Push your branch to your fork.
5.  Submit a Pull Request for review.

## Troubleshooting Tips

Common problems and potential solutions:

1.  **Import Errors:**

    ```bash
    pip install -e .  # Install the package in editable mode
    ```

2.  **Model Download Issues:**

    ```bash
    python -m spacy download en_core_web_sm --force
    ```

3.  **Quality Control Failures:**
    - Inspect the `quality_reports` directory for comprehensive error details.
    - Modify threshold values in `config/quality_config.json`.
    - Review the validation rule logic as indicated in the logs.

## Support and Assistance

For any encountered issues:

1.  Examine the log files in:
    - `pipeline.log`
    - `quality_control.log`
2.  Review the generated quality reports.
3.  Execute the complete test suite.
4.  Create a GitHub issue providing:
    - A thorough description of the problem.
    - Relevant excerpts from log files.
    - Clear steps to reproduce the issue.
