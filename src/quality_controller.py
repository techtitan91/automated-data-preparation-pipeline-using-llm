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
        
    def validate_batch(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, QualityMetrics]:
        """Validate a batch of data and return quality metrics"""
        try:
            metrics = QualityMetrics(
                completeness=self.calculate_completeness(data),
                consistency=self.calculate_consistency(data),
                validity=self.calculate_validity(data),
                uniqueness=self.calculate_uniqueness(data)
            )
            
            # Calculate overall quality score
            overall_score = np.mean([
                metrics.completeness,
                metrics.consistency,
                metrics.validity,
                metrics.uniqueness
            ])
            
            # Filter data based on quality threshold
            if overall_score >= self.threshold:
                return data, metrics
            else:
                self.logger.warning(f"Data quality below threshold: {overall_score:.2f}")
                return pd.DataFrame(), metrics
                
        except Exception as e:
            self.logger.error(f"Error in quality validation: {e}")
            raise