from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging

@dataclass
class CompletenessMetrics:
    null_score: float
    content_score: float
    effort_score: float
    sentence_score: float
    
    def to_dict(self):
        return {
            'null_score': self.null_score,
            'content_score': self.content_score,
            'effort_score': self.effort_score,
            'sentence_score': self.sentence_score
        }

@dataclass
class QualityMetrics:
    completeness: CompletenessMetrics
    consistency: float
    validity: float
    uniqueness: float
    nlp_quality: float
    
    def to_dict(self):
        return {
            'completeness': self.completeness.to_dict(),
            'consistency': self.consistency,
            'validity': self.validity,
            'uniqueness': self.uniqueness,
            'nlp_quality': self.nlp_quality
        }

class QualityController:
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)
        
    def calculate_completeness(self, data: pd.DataFrame) -> CompletenessMetrics:
        """Calculate detailed completeness metrics for conversational data"""
        column_metrics = {}
        
        for column in data.columns:
            if data[column].dtype == 'object':
                # 1. Basic null/NA check
                null_score = 1 - data[column].isnull().mean()
                
                # 2. Content length check
                meaningful_content = (
                    data[column].astype(str)
                    .str.strip()
                    .str.len() > 10
                )
                content_score = meaningful_content.mean()
                
                # 3. Low-effort response check
                low_effort_patterns = [
                    r'^(yes|no|maybe|idk|i don\'t know)$',
                    r'^[^a-zA-Z]*$',
                    r'^.{1,5}$'
                ]
                effort_score = 1 - (
                    data[column].astype(str)
                    .str.lower()
                    .str.strip()
                    .str.match('|'.join(low_effort_patterns))
                    .fillna(False)
                    .mean()
                )
                
                # 4. Sentence structure check
                sentence_score = (
                    data[column].astype(str)
                    .str.contains(r'[.!?]')
                    .fillna(False)
                    .mean()
                )
                
                column_metrics[column] = CompletenessMetrics(
                    null_score=null_score,
                    content_score=content_score,
                    effort_score=effort_score,
                    sentence_score=sentence_score
                )
                
                print(f"Column: {column}")
                print(f"Null score: {null_score:.3f}")
                print(f"Content score: {content_score:.3f}")
                print(f"Effort score: {effort_score:.3f}")
                print(f"Sentence score: {sentence_score:.3f}")
        
        # Average metrics across all columns
        avg_metrics = CompletenessMetrics(
            null_score=np.mean([m.null_score for m in column_metrics.values()]),
            content_score=np.mean([m.content_score for m in column_metrics.values()]),
            effort_score=np.mean([m.effort_score for m in column_metrics.values()]),
            sentence_score=np.mean([m.sentence_score for m in column_metrics.values()])
        )
        
        return avg_metrics
        
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
        """Calculate validity score based on multiple business rules:
        1. Null check (original)
        2. Data type conformity
        3. Range checks for numeric columns
        4. Pattern matching for string columns
        5. Date format validation
        """
        validity_scores = []
        
        for column in data.columns:
            column_scores = []
            
            # 1. Original null check
            valid_count = data[column].notna().sum()
            column_scores.append(valid_count / len(data))
            
            # 2. Data type conformity
            if data[column].dtype in ['int64', 'float64']:
                # Check if numeric values are within reasonable bounds
                numeric_valid = (~data[column].isin([np.inf, -np.inf])) & (data[column].abs() < 1e9)
                column_scores.append(numeric_valid.mean())
                
                # 3. Range checks (assuming values should be positive)
                if data[column].dtype == 'int64':
                    positive_score = (data[column] >= 0).mean()
                    column_scores.append(positive_score)
                
            elif data[column].dtype == 'object':
                non_empty = data[column].astype(str).str.strip().str.len() > 0
                column_scores.append(non_empty.mean())
                
                # 4. Pattern matching for common formats
                if 'email' in column.lower():
                    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                    pattern_score = data[column].str.match(email_pattern).mean()
                    column_scores.append(pattern_score)
                    
                elif 'phone' in column.lower():
                    phone_pattern = r'^\+?1?\d{9,15}$'
                    pattern_score = data[column].str.match(phone_pattern).mean()
                    column_scores.append(pattern_score)
                    
            # 5. Date validation
            elif data[column].dtype == 'datetime64[ns]':
                valid_dates = pd.to_datetime(data[column], errors='coerce').notna()
                future_dates = data[column] <= pd.Timestamp.now()
                column_scores.append(valid_dates.mean())
                column_scores.append(future_dates.mean())
            
            # Calculate average score for this column
            validity_scores.append(np.mean(column_scores))
            
        return np.mean(validity_scores)
        
    def calculate_uniqueness(self, data: pd.DataFrame) -> float:
        """Calculate uniqueness score considering multiple factors:
        1. Full row duplicates
        2. Column-level uniqueness
        3. Near-duplicate detection for text columns
        """
        scores = []
        
        # 1. Full row duplicates
        row_uniqueness = 1 - data.duplicated().sum() / len(data)
        scores.append(row_uniqueness)
        
        # 2. Column-level uniqueness
        for column in data.columns:
            column_uniqueness = 1 - data[column].duplicated().sum() / len(data)
            scores.append(column_uniqueness)
            
            # 3. For text columns, check for near-duplicates
            if data[column].dtype == 'object':
                # Convert to lowercase and remove extra spaces
                cleaned = data[column].str.lower().str.strip()
                # Check for similar text after basic normalization
                normalized_uniqueness = 1 - cleaned.duplicated().sum() / len(data)
                scores.append(normalized_uniqueness)
                
                # Optional: Add fuzzy matching for similar strings
                # Note: This can be computationally expensive for large datasets
                if len(data) < 100000:  # Only for smaller datasets
                    from difflib import SequenceMatcher
                    similar_pairs = 0
                    unique_values = cleaned.unique()
                    for i in range(len(unique_values)):
                        for j in range(i + 1, len(unique_values)):
                            similarity = SequenceMatcher(None, unique_values[i], unique_values[j]).ratio()
                            if similarity > 0.9:  # Threshold for similarity
                                similar_pairs += 1
                    fuzzy_uniqueness = 1 - (similar_pairs / len(data))
                    scores.append(fuzzy_uniqueness)
        
        # Return weighted average of all uniqueness scores
        # Give more weight to full row uniqueness
        weights = [2.0] + [1.0] * (len(scores) - 1)
        return np.average(scores, weights=weights)
        
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
            nlp_quality_score, fixed_data = self.calculate_nlp_quality(data)
            
            metrics = QualityMetrics(
                completeness=self.calculate_completeness(data),
                consistency=self.calculate_consistency(data),
                validity=self.calculate_validity(data),
                uniqueness=self.calculate_uniqueness(data),
                nlp_quality=nlp_quality_score
            )
            
            # Calculate overall quality score using average of completeness components
            completeness_avg = np.mean([
                metrics.completeness.null_score,
                metrics.completeness.content_score,
                metrics.completeness.effort_score,
                metrics.completeness.sentence_score
            ])
            
            overall_score = np.mean([
                completeness_avg,
                metrics.consistency,
                metrics.validity,
                metrics.uniqueness,
                metrics.nlp_quality
            ])
            
            if overall_score >= self.threshold:
                return fixed_data, metrics
            else:
                self.logger.warning(f"Data quality below threshold: {overall_score:.2f}")
                return pd.DataFrame(), metrics
                
        except Exception as e:
            self.logger.error(f"Error in quality validation: {e}")
            raise