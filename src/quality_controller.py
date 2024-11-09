from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging

@dataclass
class QualityMetrics:
    completeness: float
    consistency: float
    validity: float
    uniqueness: float
    nlp_quality: float
    
class QualityController:
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)
        
    def calculate_completeness(self, data: pd.DataFrame) -> float:
        """Calculate the completeness score of the dataset"""
        return 1 - data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        
    def calculate_consistency(self, data: pd.DataFrame) -> float:
        """Calculate consistency score based on data patterns"""
        consistency_scores = []
        
        for column in data.columns:
            if data[column].dtype == 'object':
                # Check string pattern consistency
                lengths = data[column].str.len()
                consistency_scores.append(1 - lengths.std() / lengths.mean() if lengths.mean() > 0 else 0)
                
        return np.mean(consistency_scores) if consistency_scores else 1.0
        
    def calculate_validity(self, data: pd.DataFrame) -> float:
        """Calculate validity score based on business rules"""
        validity_scores = []
        
        for column in data.columns:
            valid_count = data[column].notna().sum()
            validity_scores.append(valid_count / len(data))
            
        return np.mean(validity_scores)
        
    def calculate_uniqueness(self, data: pd.DataFrame) -> float:
        """Calculate uniqueness score (duplicate detection)"""
        duplicate_rows = data.duplicated().sum()
        return 1 - duplicate_rows / len(data)
        
    def calculate_nlp_quality(self, text_column: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
        """Calculate NLP quality score and fix low-quality text"""
        # try:
        nlp_scores = []
        fixed_data = text_column.copy()
        
        for column in text_column.columns:
            if text_column[column].dtype == 'object':
                series = text_column[column]
                
                # Calculate initial quality metrics
                avg_word_count = series.str.split().str.len().mean()
                non_empty_ratio = (series.str.strip().str.len() > 0).mean()
                special_char_ratio = (series.str.contains(r'[^a-zA-Z0-9\s]')).mean()
                all_caps_ratio = (series.str.isupper()).mean()
                
                column_score = (
                    (0.3 * min(avg_word_count / 10, 1.0)) +
                    (0.3 * non_empty_ratio) +
                    (0.2 * (1 - special_char_ratio)) +
                    (0.2 * (1 - all_caps_ratio))
                )
                
                # If quality is low, apply fixes
                if column_score < 0.9:
                    self.logger.info(f"Fixing low quality text in column {column}")
                    
                    # Apply text improvements
                    fixed_data[column] = (series
                        # Fix whitespace issues
                        .str.strip()
                        .str.replace(r'\s+', ' ')
                        
                        # Fix capitalization
                        .str.replace(r'([A-Z]{2,})', lambda x: x.group(1).title(), regex=True)  # Convert ALL CAPS to Title Case
                        .str.replace(r'^[a-z]', lambda x: x.group(0).upper(), regex=True)       # Capitalize first letter
                        
                        # Clean special characters but preserve important punctuation
                        .str.replace(r'[^\w\s.,!?;:()-]', '')
                        
                        # Ensure proper spacing around punctuation
                        .str.replace(r'\s*([.,!?;:])\s*', r'\1 ')
                        
                        # Remove redundant punctuation
                        .str.replace(r'([.,!?;:]){2,}', r'\1')
                        
                        # Handle empty/null values
                        .fillna('')
                    )
                    
                    # Remove entries that are too short or empty
                    mask = (fixed_data[column].str.split().str.len() >= 3) & (fixed_data[column].str.strip() != '')
                    fixed_data = fixed_data[mask]
                    
                    # Recalculate score after fixes
                    series = fixed_data[column]
                    avg_word_count = series.str.split().str.len().mean()
                    non_empty_ratio = (series.str.strip().str.len() > 0).mean()
                    special_char_ratio = (series.str.contains(r'[^a-zA-Z0-9\s]')).mean()
                    all_caps_ratio = (series.str.isupper()).mean()
                    
                    column_score = (
                        (0.3 * min(avg_word_count / 10, 1.0)) +
                        (0.3 * non_empty_ratio) +
                        (0.2 * (1 - special_char_ratio)) +
                        (0.2 * (1 - all_caps_ratio))
                    )
                
                nlp_scores.append(column_score)
        
        final_score = np.mean(nlp_scores) if nlp_scores else 0.0
        
        return final_score, fixed_data
            
        # except Exception as e:
        #     self.logger.error(f"Error calculating/fixing NLP quality: {e}")
        #     return 0.0, text_column

    def validate_batch(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, QualityMetrics]:
        """Validate a batch of data and return quality metrics"""
        try:
            # Get just the score from the nlp_quality tuple
            nlp_quality_score, fixed_data = self.calculate_nlp_quality(data)
            # print(fixed_data)
            # for row in fixed_data.iterrows():
            #     print(row[1]['original'])
            #     print(row[1]['cleaned'])
            metrics = QualityMetrics(
                completeness=self.calculate_completeness(data),
                consistency=self.calculate_consistency(data),
                validity=self.calculate_validity(data),
                uniqueness=self.calculate_uniqueness(data),
                nlp_quality=nlp_quality_score  # Use just the score
            )
            
            # Calculate overall quality score
            overall_score = np.mean([
                metrics.completeness,
                metrics.consistency,
                metrics.validity,
                metrics.uniqueness,
                metrics.nlp_quality
            ])
            
            # Filter data based on quality threshold
            if overall_score >= self.threshold:
                return fixed_data, metrics
            else:
                self.logger.warning(f"Data quality below threshold: {overall_score:.2f}")
                return pd.DataFrame(), metrics
                
        except Exception as e:
            self.logger.error(f"Error in quality validation: {e}")
            raise