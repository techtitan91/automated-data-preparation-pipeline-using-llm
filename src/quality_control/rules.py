from typing import List, Dict, Any
import pandas as pd
import re
import numpy as np
from dataclasses import dataclass
from .base import ValidationRule, DataQualityDimension
from langdetect import detect, detect_langs
import logging

@dataclass
class SpecialCharacterConfig:
    max_special_char_ratio: float = 0.3
    allowed_special_chars: set = None
    exclude_urls: bool = True
    exclude_code_blocks: bool = True

    def __post_init__(self):
        if self.allowed_special_chars is None:
            self.allowed_special_chars = {'.', ',', '!', '?', '-', '_', ':', ';', '(', ')', '[', ']', '"', "'"}

class SpecialCharacterRule(ValidationRule):
    """Validates the presence and distribution of special characters in text data"""
    
    def __init__(self, config: SpecialCharacterConfig = None):
        super().__init__(
            name="Special Character Validation",
            dimension=DataQualityDimension.VALIDITY,
            description="Checks for excessive or invalid special characters",
            severity="medium"
        )
        self.config = config or SpecialCharacterConfig()
        self.logger = logging.getLogger(__name__)
        
    def validate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validates special characters in text columns
        Returns: Dict with validation results
        """
        results = {
            'passed': True,
            'issues': [],
            'stats': {}
        }
        
        try:
            text_columns = data.select_dtypes(include=['object']).columns
            
            for column in text_columns:
                column_issues = []
                special_char_ratios = []
                
                for text in data[column].dropna():
                    if not isinstance(text, str):
                        continue
                        
                    # Skip URLs if configured
                    if self.config.exclude_urls:
                        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
                    
                    # Skip code blocks if configured
                    if self.config.exclude_code_blocks:
                        text = re.sub(r'```[\s\S]*?```', '', text)
                        text = re.sub(r'`.*?`', '', text)
                    
                    # Count special characters
                    special_chars = [c for c in text if not c.isalnum() and not c.isspace() and c not in self.config.allowed_special_chars]
                    ratio = len(special_chars) / len(text) if len(text) > 0 else 0
                    special_char_ratios.append(ratio)
                    
                    if ratio > self.config.max_special_char_ratio:
                        column_issues.append(f"High special character ratio ({ratio:.2f}) in text: {text[:50]}...")
                
                # Add column statistics
                results['stats'][column] = {
                    'avg_special_char_ratio': np.mean(special_char_ratios) if special_char_ratios else 0,
                    'max_special_char_ratio': max(special_char_ratios) if special_char_ratios else 0,
                    'violations': len(column_issues)
                }
                
                results['issues'].extend(column_issues)
                
            # Explicitly set passed to bool
            results['passed'] = bool(len(results['issues']) == 0)
            return results
            
        except Exception as e:
            self.logger.error(f"Error in special character validation: {e}")
            results['passed'] = False
            results['issues'].append(f"Error in validation: {str(e)}")
            return results

class LanguageConsistencyRule(ValidationRule):
    """Validates language consistency across text fields"""
    
    def __init__(self, primary_language: str = 'en', threshold: float = 0.8):
        super().__init__(
            name="Language Consistency",
            dimension=DataQualityDimension.CONSISTENCY,
            description="Checks if text content is consistently in the expected language",
            severity="high"
        )
        self.primary_language = primary_language
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)
        
    def validate(self, data: pd.DataFrame) -> Dict[str, Any]:
        results = {
            'passed': True,
            'issues': [],
            'stats': {}
        }
        
        try:
            text_columns = data.select_dtypes(include=['object']).columns
            
            for column in text_columns:
                column_languages = []
                column_issues = []
                
                for text in data[column].dropna():
                    if not isinstance(text, str) or len(text.strip()) < 10:
                        continue
                        
                    try:
                        # Detect language and confidence
                        langs = detect_langs(text)
                        primary_lang = langs[0]
                        column_languages.append(primary_lang.lang)
                        
                        if primary_lang.lang != self.primary_language:
                            if primary_lang.prob > 0.8:  # High confidence in different language
                                column_issues.append(
                                    f"Text detected as {primary_lang.lang} (confidence: {primary_lang.prob:.2f}): {text[:50]}..."
                                )
                                
                    except Exception as lang_e:
                        self.logger.warning(f"Language detection failed for text: {str(lang_e)}")
                        continue
                
                # Calculate language statistics
                if column_languages:
                    primary_lang_ratio = column_languages.count(self.primary_language) / len(column_languages)
                    results['stats'][column] = {
                        'primary_language_ratio': primary_lang_ratio,
                        'detected_languages': list(set(column_languages)),
                        'violations': len(column_issues)
                    }
                    
                    if primary_lang_ratio < self.threshold:
                        column_issues.append(
                            f"Column {column} has low primary language ratio: {primary_lang_ratio:.2f}"
                        )
                
                results['issues'].extend(column_issues)
            
            results['passed'] = len(results['issues']) == 0
            return results
            
        except Exception as e:
            self.logger.error(f"Error in language consistency validation: {e}")
            results['passed'] = False
            results['issues'].append(f"Error in validation: {str(e)}")
            return results

class DataFormatRule(ValidationRule):
    """Validates data format consistency across fields"""
    
    def __init__(self, format_patterns: Dict[str, str] = None):
        super().__init__(
            name="Data Format",
            dimension=DataQualityDimension.VALIDITY,
            description="Validates data format consistency",
            severity="high"
        )
        self.format_patterns = format_patterns or {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone': r'^\+?1?\d{9,15}$',
            'date': r'^\d{4}-\d{2}-\d{2}$',
            'url': r'^https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)$'
        }
        self.logger = logging.getLogger(__name__)
        
    def validate(self, data: pd.DataFrame) -> Dict[str, Any]:
        results = {
            'passed': True,
            'issues': [],
            'stats': {}
        }
        
        try:
            # Detect column types based on column names and content
            for column in data.columns:
                column_lower = column.lower()
                format_type = None
                
                # Try to identify column type from name
                for type_name, pattern in self.format_patterns.items():
                    if type_name in column_lower:
                        format_type = type_name
                        break
                
                if format_type:
                    pattern = self.format_patterns[format_type]
                    invalid_values = []
                    
                    for value in data[column].dropna():
                        if not isinstance(value, str):
                            invalid_values.append(str(value))
                            continue
                            
                        if not re.match(pattern, value):
                            invalid_values.append(value)
                    
                    if invalid_values:
                        results['issues'].append(
                            f"Column {column} ({format_type}) has {len(invalid_values)} invalid format values"
                        )
                        results['stats'][column] = {
                            'format_type': format_type,
                            'invalid_count': len(invalid_values),
                            'invalid_examples': invalid_values[:5]
                        }
            
            results['passed'] = len(results['issues']) == 0
            return results
            
        except Exception as e:
            self.logger.error(f"Error in data format validation: {e}")
            results['passed'] = False
            results['issues'].append(f"Error in validation: {str(e)}")
            return results

class CrossFieldConsistencyRule(ValidationRule):
    """Validates consistency between related fields"""
    
    def __init__(self, field_relationships: List[Dict[str, Any]] = None):
        super().__init__(
            name="Cross-field Consistency",
            dimension=DataQualityDimension.CONSISTENCY,
            description="Validates consistency between related fields",
            severity="high"
        )
        self.field_relationships = field_relationships or []
        self.logger = logging.getLogger(__name__)
        
    def add_relationship(self, fields: List[str], validation_func: callable, description: str):
        """Add a new field relationship to validate"""
        self.field_relationships.append({
            'fields': fields,
            'validation_func': validation_func,
            'description': description
        })
        
    def validate(self, data: pd.DataFrame) -> Dict[str, Any]:
        results = {
            'passed': True,
            'issues': [],
            'stats': {}
        }
        
        try:
            # Validate each defined relationship
            for relationship in self.field_relationships:
                fields = relationship['fields']
                
                # Check if all required fields exist
                if not all(field in data.columns for field in fields):
                    missing_fields = [f for f in fields if f not in data.columns]
                    results['issues'].append(
                        f"Missing required fields for validation: {missing_fields}"
                    )
                    continue
                
                # Apply validation function
                try:
                    validation_result = relationship['validation_func'](data[fields])
                    
                    if not validation_result['passed']:
                        results['issues'].extend(validation_result['issues'])
                        results['stats']['+'.join(fields)] = validation_result['stats']
                        
                except Exception as val_e:
                    results['issues'].append(
                        f"Validation failed for fields {fields}: {str(val_e)}"
                    )
            
            results['passed'] = len(results['issues']) == 0
            return results
            
        except Exception as e:
            self.logger.error(f"Error in cross-field validation: {e}")
            results['passed'] = False
            results['issues'].append(f"Error in validation: {str(e)}")
            return results

# Example usage with the CrossFieldConsistencyRule
def validate_date_consistency(data: pd.DataFrame) -> Dict[str, Any]:
    """Example validation function for date-related fields"""
    results = {
        'passed': True,
        'issues': [],
        'stats': {
            'total_violations': 0,
            'violation_types': {}
        }
    }
    
    try:
        # Convert dates to datetime
        start_dates = pd.to_datetime(data['start_date'])
        end_dates = pd.to_datetime(data['end_date'])
        
        # Check date order
        invalid_orders = start_dates > end_dates
        if invalid_orders.any():
            violation_count = invalid_orders.sum()
            results['passed'] = False
            results['issues'].append(
                f"Found {violation_count} cases where start_date is after end_date"
            )
            results['stats']['violation_types']['invalid_order'] = violation_count
            
        return results
        
    except Exception as e:
        results['passed'] = False
        results['issues'].append(f"Error in date validation: {str(e)}")
        return results