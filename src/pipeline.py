# pipeline.py
from typing import Tuple, List
from pathlib import Path
import logging
import pandas as pd
from src.annotator import DataAnnotator
from src.s3_connector import S3Connector
from src.cleaner import AdvancedCleaner, CleaningConfig
from src.quality_controller import QualityController, QualityMetrics
import json

class AnnotationPipeline:
    def __init__(self, output_dir: str = "annotated_data"):
        # Initialize components for data processing pipeline
        self.annotator = DataAnnotator()
        self.connector = S3Connector()
        self.cleaner = AdvancedCleaner(CleaningConfig(
            min_text_length=20,
            max_text_length=1000,
            quality_threshold=0.8
        ))
        self.quality_controller = QualityController(threshold=0.8)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging for the pipeline"""
        log_file = self.output_dir / "pipeline.log"
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)

    def process_datasets(self):
        """Process each dataset through the complete pipeline"""
        self.logger.info("Starting dataset processing")
        try:
            datasets = self.connector.list_datasets()
            for dataset in datasets:
                self.logger.info(f"Processing dataset: {dataset}")
                
                # Create temporary file path
                temp_path = self.output_dir / f"temp_{Path(dataset).name}"
                
                # Download dataset
                self.connector.download_dataset(dataset, str(temp_path))
                
                # Read and process the dataset
                df, text_columns = self._read_dataset(temp_path)
                
                # Main processing loop for each text column
                for column in text_columns:
                    self.logger.info(f"Processing column: {column}")
                    
                    # Initialize containers for processed data
                    texts = df[column].tolist()
                    cleaned_texts = []
                    annotations = []
                    
                    # Process each text entry individually to maintain data integrity
                    for text in texts:
                        if text is None or not isinstance(text, str):
                            cleaned_texts.append("")
                            annotations.append([])
                            continue
                            
                        # Clean single text
                        cleaned = self.cleaner.clean_batch([text])
                        
                        if cleaned:
                            cleaned_text = cleaned[0]
                            cleaned_texts.append(cleaned_text)
                            # Annotate cleaned text
                            try:
                                annotated_data = self.annotator.process_batch(cleaned_text)
                                annotations.append(annotated_data)
                            except Exception as e:
                                self.logger.warning(f"Annotation failed for text: {e}")
                                annotations.append([])
                        else:
                            cleaned_texts.append("")
                            annotations.append([])
                    
                    # Verify lengths match
                    assert len(cleaned_texts) == len(df), "Cleaned texts length mismatch"
                    assert len(annotations) == len(df), "Annotations length mismatch"
                    
                    # Create new columns
                    cleaned_column = f"{column}_cleaned"
                    annotation_column = f"{column}_annotations"
                    
                    processed_df[cleaned_column] = cleaned_texts
                    processed_df[annotation_column] = annotations
                    
                    # Quality control for the column
                    column_df = pd.DataFrame({
                        'original': df[column],
                        'cleaned': cleaned_texts,
                        'has_annotation': [bool(ann) for ann in annotations]
                    })
                    
                    validated_data, metrics = self.quality_controller.validate_batch(column_df)
                    quality_metrics_dict[column] = {
                        'completeness': metrics.completeness,
                        'consistency': metrics.consistency,
                        'validity': metrics.validity,
                        'uniqueness': metrics.uniqueness,
                        'nlp_quality': metrics.nlp_quality
                    }
                
                # Save processed data
                output_base = self.output_dir / f"processed_{Path(dataset).stem}"
                
                # Save the processed DataFrame
                processed_df.to_csv(f"{output_base}.csv", index=False)
                
                # Save quality metrics
                with open(f"{output_base}_metrics.json", 'w') as f:
                    json.dump(quality_metrics_dict, f, indent=2)
                
                # Cleanup
                temp_path.unlink()
                
                self.logger.info(f"Completed processing dataset: {dataset}")
                break
                
        except Exception as e:
            self.logger.error(f"Error in pipeline: {e}", exc_info=True)
            raise

    def _read_dataset(self, file_path: Path) -> Tuple[pd.DataFrame, List[str]]:
        """Read dataset and identify text columns."""
        ext = file_path.suffix.lower()
        try:
            # Support multiple file formats
            if ext == '.csv':
                df = pd.read_csv(file_path)
            elif ext == '.json':
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
            
            # Identify columns containing meaningful text data
            text_columns = []
            for column in df.columns:
                if df[column].dtype == 'object':
                    sample = df[column].dropna().iloc[0] if not df[column].empty else None
                    if sample and isinstance(sample, str):
                        # Filter out short strings that are likely categorical
                        avg_length = df[column].str.len().mean()
                        if avg_length > 10:
                            text_columns.append(column)
            
            if not text_columns:
                self.logger.warning(f"No text columns found. Available columns: {df.columns.tolist()}")
            else:
                self.logger.info(f"Detected text columns: {text_columns}")
            
            return df, text_columns
            
        except Exception as e:
            self.logger.error(f"Error reading dataset {file_path}: {e}")
            raise
