# src/quality_control/base.py
from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd
from enum import Enum

class DataQualityDimension(Enum):
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    TIMELINESS = "timeliness"
    INTEGRITY = "integrity"
    ACCURACY = "accuracy"

class ValidationRule:
    """Base class for all validation rules"""
    def __init__(self, 
                 name: str,
                 dimension: DataQualityDimension,
                 description: str,
                 severity: str):
        self.name = name
        self.dimension = dimension
        self.description = description
        self.severity = severity

    def validate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the data according to this rule
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dict containing:
                - passed: bool indicating if validation passed
                - issues: list of issue descriptions
                - stats: dict of relevant statistics
        """
        raise NotImplementedError("Validation logic must be implemented in subclasses")