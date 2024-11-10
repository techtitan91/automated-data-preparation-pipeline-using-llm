from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import re
import logging
from enum import Enum
from scipy import stats  # Add this import
from .base import DataQualityDimension

class DataQualityDimension(Enum):
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    TIMELINESS = "timeliness"
    INTEGRITY = "integrity"
    ACCURACY = "accuracy"

@dataclass
class QualityMetrics:
    completeness: float
    consistency: float
    validity: float
    uniqueness: float
    timeliness: float
    integrity: float
    accuracy: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'completeness': self.completeness,
            'consistency': self.consistency,
            'validity': self.validity,
            'uniqueness': self.uniqueness,
            'timeliness': self.timeliness,
            'integrity': self.integrity,
            'accuracy': self.accuracy
        }
    
    @property
    def overall_score(self) -> float:
        """Calculate overall quality score"""
        weights = {
            'completeness': 0.2,
            'consistency': 0.15,
            'validity': 0.15,
            'uniqueness': 0.15,
            'timeliness': 0.1,
            'integrity': 0.15,
            'accuracy': 0.1
        }
        
        scores = self.to_dict()
        weighted_sum = sum(scores[metric] * weight for metric, weight in weights.items())
        return weighted_sum

class QualityController:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
    def _default_config(self) -> Dict:
        return {
            'min_completeness': 0.8,
            'min_consistency': 0.7,
            'min_validity': 0.8,
            'min_uniqueness': 0.9,
            'max_data_age_days': 30,
            'min_accuracy': 0.8,
            'text_length_threshold': 10,
            'min_overall_score': 0.75
        }
    
    def calculate_completeness(self, data: pd.DataFrame) -> float:
        """Calculate data completeness score"""
        if data.empty:
            return 0.0
        
        return 1.0 - data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
    
    def calculate_consistency(self, data: pd.DataFrame) -> float:
        """Calculate data consistency score"""
        if data.empty:
            return 0.0
            
        consistency_scores = []
        
        for column in data.columns:
            if data[column].dtype == 'object':
                # Calculate length variance for text columns
                lengths = data[column].str.len()
                if not lengths.empty and lengths.mean() > 0:
                    variance_score = 1 - (lengths.std() / lengths.mean())
                    consistency_scores.append(max(0, variance_score))
                
        return np.mean(consistency_scores) if consistency_scores else 1.0
    
    def calculate_validity(self, data: pd.DataFrame) -> float:
        """Calculate data validity score"""
        if data.empty:
            return 0.0
            
        validity_scores = []
        
        for column in data.columns:
            valid_count = data[column].notna().sum()
            
            if data[column].dtype == 'object':
                # Check for minimum text length
                if self.config['text_length_threshold'] > 0:
                    valid_length = data[column].astype(str).str.len() >= self.config['text_length_threshold']
                    valid_count = valid_length.sum()
                
            validity_scores.append(valid_count / len(data))
            
        return np.mean(validity_scores)
    
    def calculate_uniqueness(self, data: pd.DataFrame) -> float:
        """Calculate uniqueness score"""
        if data.empty:
            return 0.0
            
        # Check for duplicate rows
        duplicate_rows = data.duplicated().sum()
        row_uniqueness = 1 - (duplicate_rows / len(data))
        
        # Check for near-duplicates in text columns
        text_columns = data.select_dtypes(include=['object']).columns
        text_uniqueness_scores = []
        
        for column in text_columns:
            unique_values = data[column].nunique()
            total_values = len(data[column])
            text_uniqueness_scores.append(unique_values / total_values)
        
        text_uniqueness = np.mean(text_uniqueness_scores) if text_uniqueness_scores else 1.0
        
        return 0.7 * row_uniqueness + 0.3 * text_uniqueness
    
    def calculate_timeliness(self, data: pd.DataFrame, timestamp_column: str = None) -> float:
        """Calculate timeliness score"""
        if data.empty or (timestamp_column and timestamp_column not in data.columns):
            return 1.0  # Default to 1.0 if no timestamp column
            
        if timestamp_column:
            try:
                timestamps = pd.to_datetime(data[timestamp_column])
                current_time = pd.Timestamp.now()
                
                # Calculate age in days
                ages = (current_time - timestamps).dt.total_seconds() / (24 * 3600)
                
                # Calculate score based on age
                max_age = self.config['max_data_age_days']
                timeliness_scores = 1 - (ages / max_age).clip(0, 1)
                
                return float(timeliness_scores.mean())
            except Exception as e:
                self.logger.warning(f"Error calculating timeliness: {e}")
                return 1.0
        
        return 1.0
    
    def calculate_integrity(self, data: pd.DataFrame) -> float:
        """Calculate data integrity score"""
        if data.empty:
            return 0.0
            
        integrity_scores = []
        
        # Check for broken references
        for column in data.columns:
            if column.endswith('_id') or column.endswith('_ref'):
                referenced_values = set(data[column].dropna())
                valid_references = True  # Implement reference checking logic
                integrity_scores.append(1.0 if valid_references else 0.0)
        
        return np.mean(integrity_scores) if integrity_scores else 1.0
    
    def calculate_accuracy(self, data: pd.DataFrame) -> float:
        """Calculate accuracy score"""
        if data.empty:
            return 0.0
            
        accuracy_scores = []
        
        # Check for outliers in numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            non_null_data = data[column].dropna()
            if not non_null_data.empty:
                zscore = np.abs(stats.zscore(non_null_data))
                outlier_percentage = (zscore > 3).mean()
                accuracy_scores.append(1 - outlier_percentage)
        
        # Check text quality for string columns
        text_columns = data.select_dtypes(include=['object']).columns
        for column in text_columns:
            quality_score = self._check_text_quality(data[column])
            accuracy_scores.append(quality_score)
        
        return np.mean(accuracy_scores) if accuracy_scores else 0.0
    
    def _check_text_quality(self, series: pd.Series) -> float:
        """Check quality of text data"""
        quality_scores = []
        
        for text in series.dropna():
            if not isinstance(text, str):
                quality_scores.append(0.0)
                continue
                
            score = 1.0
            
            # Check for excessive special characters
            special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s]', text)) / len(text) if text else 0
            if special_char_ratio > 0.3:
                score *= 0.7
            
            # Check for reasonable length
            if len(text) < self.config['text_length_threshold']:
                score *= 0.8
            
            # Check for repetitive patterns
            if re.search(r'(.)\1{4,}', text):  # Repeated character more than 4 times
                score *= 0.9
                
            quality_scores.append(score)
            
        return np.mean(quality_scores) if quality_scores else 0.0
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, List[str], QualityMetrics]:
        """Validate data against all quality dimensions"""
        try:
            metrics = QualityMetrics(
                completeness=self.calculate_completeness(data),
                consistency=self.calculate_consistency(data),
                validity=self.calculate_validity(data),
                uniqueness=self.calculate_uniqueness(data),
                timeliness=self.calculate_timeliness(data),
                integrity=self.calculate_integrity(data),
                accuracy=self.calculate_accuracy(data)
            )
            
            # Check against thresholds
            issues = []
            if metrics.completeness < self.config['min_completeness']:
                issues.append(f"Completeness score {metrics.completeness:.2f} below threshold {self.config['min_completeness']}")
                
            if metrics.consistency < self.config['min_consistency']:
                issues.append(f"Consistency score {metrics.consistency:.2f} below threshold {self.config['min_consistency']}")
            
            # Calculate overall pass/fail
            passed = bool(len(issues) == 0 and metrics.overall_score >= self.config['min_overall_score'])
            
            return passed, issues, metrics
            
        except Exception as e:
            self.logger.error(f"Error in data validation: {e}")
            raise