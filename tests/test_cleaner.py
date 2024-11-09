# tests/test_cleaner.py
import unittest
import sys
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.cleaner import AdvancedCleaner, CleaningConfig
from src.quality_controller import QualityController
import pandas as pd
import unicodedata

class TestAdvancedCleaner(unittest.TestCase):
    def setUp(self):
        self.cleaner = AdvancedCleaner(CleaningConfig())
        
    def test_remove_boilerplate(self):
        text = "Important content\nBest regards,\nJohn Doe"
        cleaned = self.cleaner.remove_boilerplate(text)
        self.assertEqual(cleaned, "Important content")
        
    def test_standardize_format(self):
        text = "Multiple    spaces   and\n\n\nextra breaks"
        cleaned = self.cleaner.standardize_format(text)
        self.assertEqual(cleaned, "Multiple spaces and\n\nextra breaks")
        
    def test_meets_quality_standards(self):
        good_text = "This is a good quality text with proper length and structure."
        bad_text = "too short"
        
        self.assertTrue(self.cleaner.meets_quality_standards(good_text))
        self.assertFalse(self.cleaner.meets_quality_standards(bad_text))
        
    def test_clean_batch(self):
        texts = [
            "Good quality text example",
            "too short",
            None,
            "Another good example with proper length"
        ]
        cleaned = self.cleaner.clean_batch(texts)
        self.assertEqual(len(cleaned), 2)

class TestQualityController(unittest.TestCase):
    def setUp(self):
        self.qc = QualityController()
        
    def test_calculate_completeness(self):
        data = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': ['a', 'b', 'c', 'd']
        })
        completeness = self.qc.calculate_completeness(data)
        self.assertEqual(completeness, 0.875)
        
    def test_validate_batch(self):
        data = pd.DataFrame({
            'text': ['good text 1', 'good text 2'],
            'label': ['A', 'B']
        })
        validated_data, metrics = self.qc.validate_batch(data)
        self.assertFalse(validated_data.empty)
        self.assertTrue(all([
            metrics.completeness >= 0,
            metrics.consistency >= 0,
            metrics.validity >= 0,
            metrics.uniqueness >= 0,
            metrics.nlp_quality >= 0
        ]))

if __name__ == '__main__':
    unittest.main()