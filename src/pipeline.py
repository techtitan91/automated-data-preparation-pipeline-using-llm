# pipeline.py
from typing import Tuple, List
from pathlib import Path
import logging
import pandas as pd
from src.annotator import DataAnnotator
from src.s3_connector import S3Connector
from src.cleaner import AdvancedCleaner, CleaningConfig
from src.quality_controller import QualityController, QualityMetrics, CompletenessMetrics
from src.benchmarker import DatasetBenchmarker
import json

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, CompletenessMetrics):
            return obj.to_dict()  # or however you want to represent it
        return super().default(obj)
    
class AnnotationPipeline:
    def __init__(self, output_dir: str = "annotated_data"):
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

        self.benchmarker = DatasetBenchmarker(self.output_dir)

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
                self.benchmarker.start_benchmark()
                
                # Create temporary file path
                temp_path = self.output_dir / f"temp_{Path(dataset).name}"
                
                # Download dataset
                self.connector.download_dataset(dataset, str(temp_path))
                
                # Read and process the dataset
                df, text_columns = self._read_dataset(temp_path)
                
                # Process each text column
                processed_df = df.copy()
                quality_metrics_dict = {}
                
                for column in text_columns:
                    self.logger.info(f"Processing column: {column}")
                    
                    # Clean the texts
                    texts = df[column].tolist()
                    cleaned_texts = []
                    annotations = []
                    
                    # Process each text while maintaining the original length
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
                
                # End benchmark and save results
                benchmark_metrics = self.benchmarker.end_benchmark(
                    dataset_name=Path(dataset).stem,
                    original_df=df,
                    processed_df=processed_df,
                    quality_metrics=quality_metrics_dict
                )
                
                # Convert BenchmarkMetrics object to dictionary
                metrics_dict = self.benchmarker._metrics_to_dict(benchmark_metrics)
                
                self.logger.info(f"Benchmark results for {dataset}:")
                self.logger.info(f"Processing time: {metrics_dict['processing_time_seconds']:.2f} seconds")
                self.logger.info(f"Memory usage: {metrics_dict['memory_usage_mb']:.2f} MB")
                self.logger.info(f"Input records: {metrics_dict['input_records']}")
                self.logger.info(f"Output records: {metrics_dict['output_records']}")
                self.logger.info(f"Cleaning ratio: {metrics_dict['cleaning_ratio']:.2%}")
                self.logger.info(f"Annotation coverage: {metrics_dict['annotation_coverage']:.2%}")

                # Log quality scores if available
                if 'quality_scores' in metrics_dict:
                    quality = metrics_dict['quality_scores']
                    self.logger.info("Quality Metrics:")
                    self.logger.info(f"  Accuracy: {quality['accuracy']:.2%}")
                    self.logger.info(f"  Precision: {quality['precision']:.2%}")
                    self.logger.info(f"  Recall: {quality['recall']:.2%}")
                    self.logger.info(f"  F1 Score: {quality['f1_score']:.2%}")

                # Save processed data
                output_base = self.output_dir / f"processed_{Path(dataset).stem}"
                
                # Save the processed DataFrame
                processed_df.to_csv(f"{output_base}.csv", index=False)
                
                # Save quality metrics
                with open(f"{output_base}_metrics.json", 'w') as f:
                    json.dump(quality_metrics_dict, f, indent=2, cls=CustomEncoder)
                
                # Cleanup
                temp_path.unlink()
                
                self.logger.info(f"Completed processing dataset: {dataset}")
                break
                
        except Exception as e:
            self.logger.error(f"Error in pipeline: {e}", exc_info=True)
            raise

    def _read_dataset(self, file_path: Path) -> Tuple[pd.DataFrame, List[str]]:
        """
        Read dataset and identify text columns.
        
        Args:
            file_path (Path): Path to the dataset file
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: 
                - The loaded dataframe
                - List of column names containing text data
        """
        ext = file_path.suffix.lower()
        try:
            # Read the file based on extension
            if ext == '.csv':
                df = pd.read_csv(file_path)
            elif ext == '.json':
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
            
            # Detect text columns
            text_columns = []
            for column in df.columns:
                # Check if column is object type (string)
                if df[column].dtype == 'object':
                    # Additional checks to confirm it's actually text
                    sample = df[column].dropna().iloc[0] if not df[column].empty else None
                    if sample and isinstance(sample, str):
                        # Check if it's a reasonably sized text (not just a category)
                        avg_length = df[column].str.len().mean()
                        if avg_length > 10:  # Adjustable threshold
                            text_columns.append(column)
            
            if not text_columns:
                self.logger.warning(f"No text columns found. Available columns: {df.columns.tolist()}")
            else:
                self.logger.info(f"Detected text columns: {text_columns}")
            
            return df, text_columns
            
        except Exception as e:
            self.logger.error(f"Error reading dataset {file_path}: {e}")
            raise

    def process_single_column(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Process a single column and return processed dataframe"""
        processed_df = df.copy()
        texts = df[column].tolist()
        cleaned_texts = []
        annotations = []
        
        # Process texts
        for text in texts:
            if text is None or not isinstance(text, str):
                cleaned_texts.append("")
                annotations.append([])
                continue
                
            cleaned = self.cleaner.clean_batch([text])
            if cleaned:
                cleaned_text = cleaned[0]
                cleaned_texts.append(cleaned_text)
                try:
                    annotated_data = self.annotator.process_batch(cleaned_text)
                    annotations.append(annotated_data)
                except Exception as e:
                    self.logger.warning(f"Annotation failed for text: {e}")
                    annotations.append([])
            else:
                cleaned_texts.append("")
                annotations.append([])
        
        # Add processed columns
        processed_df[f"{column}_cleaned"] = cleaned_texts
        processed_df[f"{column}_annotations"] = annotations
        
        return processed_df