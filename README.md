# CleanBagel: Automated Data Preparation Pipeline

This repository contains the implementation of CleanBagel, an automated data preparation pipeline specifically optimized for Llama 3.2 fine-tuning.

## Project Structure
```
cleanbagel/
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

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install required models:
```bash
python -m spacy download en_core_web_sm
```

## Configuration

1. Set up AWS credentials for S3 access:
```bash
# ~/.aws/credentials
[default]
aws_access_key_id = your_access_key
aws_secret_access_key = your_secret_key
```

2. Create quality control configuration (config/quality_config.json):
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

## Components

### 1. Annotation System
The annotation system provides comprehensive text analysis:

```python
from src.annotation_system import AnnotationSystem, AnnotationConfig

# Initialize
config = AnnotationConfig(
    enable_entities=True,
    enable_sentiment=True,
    enable_topics=True,
    enable_keywords=True,
    enable_language=True
)
system = AnnotationSystem(config)

# Process single text
annotations = system.annotate_text("Your text here")

# Process batch
annotations = system.process_batch(["Text 1", "Text 2"])
```

Features:
- Named Entity Recognition
- Sentiment Analysis
- Topic Detection
- Keyword Extraction
- Language Detection

### 2. Quality Control System
The quality control system ensures data quality through multiple dimensions:

```python
from src.quality_control import QualityController, SpecialCharacterRule

# Initialize
controller = QualityController()

# Validate data
passed, issues, metrics = controller.validate_data(your_dataframe)

# Use specific rules
rule = SpecialCharacterRule()
result = rule.validate(your_dataframe)
```

Quality Dimensions:
- Completeness
- Consistency
- Validity
- Uniqueness
- Timeliness
- Integrity
- Accuracy

### 3. Cleaning Pipeline
The cleaning pipeline processes and prepares the data:

```python
from src.pipeline import AnnotationPipeline

pipeline = AnnotationPipeline()
pipeline.process_datasets()
```

Features:
- Text cleaning
- Format standardization
- Quality validation
- Batch processing

## Usage Examples

### 1. Full Pipeline
```python
from src.pipeline import AnnotationPipeline

# Initialize and run pipeline
pipeline = AnnotationPipeline(output_dir="processed_data")
pipeline.process_datasets()
```

### 2. Quality Control
```python
from src.quality_control import QualityController
import pandas as pd

# Load data
data = pd.read_csv("your_data.csv")

# Initialize controller
controller = QualityController()

# Validate data
passed, issues, metrics = controller.validate_data(data)

# Check results
print(f"Passed: {passed}")
print(f"Issues: {issues}")
print(f"Overall Score: {metrics.overall_score}")
```

### 3. Annotation System
```python
from src.annotation_system import AnnotationSystem

# Initialize
system = AnnotationSystem()

# Process text
text = "Apple Inc. is planning to release a new iPhone next year."
result = system.annotate_text(text)

# View results
print("Entities:", result['entities'])
print("Sentiment:", result['sentiment'])
print("Topics:", result['topics'])
```

## Testing

Run all tests:
```bash
python -m unittest discover tests
```

Run specific test suites:
```bash
python -m unittest tests/test_cleaner.py
python -m unittest tests/test_annotation_system.py
python -m unittest tests/test_quality_control.py
```

## Output Structure

### Quality Reports
```
quality_reports/
├── quality_report_dataset1_20241110_123456.json
├── quality_report_dataset1_20241110_123456.txt
├── quality_report_dataset2_20241110_123457.json
└── quality_report_dataset2_20241110_123457.txt
```

### Processed Data
```
processed_data/
├── processed_dataset1.csv
├── processed_dataset1_metrics.json
├── processed_dataset2.csv
└── processed_dataset2_metrics.json
```

## Development Timeline

### Day 1 ✅
- [x] Environment setup
- [x] Core infrastructure
- [x] Basic cleaning pipeline
- [x] Initial testing

### Day 2 ✅
- [x] Annotation system implementation
- [x] Quality control metrics
- [x] Validation rules
- [x] Integration testing

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Troubleshooting

Common issues and solutions:

1. Import Errors:
```bash
pip install -e .  # Install package in editable mode
```

2. Model Download Issues:
```bash
python -m spacy download en_core_web_sm --force
```

3. Quality Control Failures:
- Check the quality_reports directory for detailed error reports
- Adjust thresholds in config/quality_config.json
- Review the validation rules in the logs

## Support

For any issues:
1. Check the logs in:
   - pipeline.log
   - quality_control.log
2. Review the quality reports
3. Run the test suite
4. Create a GitHub issue with:
   - Detailed description
   - Relevant log excerpts
   - Steps to reproduce

## License

This project is licensed under the MIT License - see the LICENSE file for details.