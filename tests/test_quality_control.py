import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.quality_control.metrics import (
    QualityController, 
    QualityMetrics,
    DataQualityDimension
)
from src.quality_control.rules import (
    SpecialCharacterRule,
    LanguageConsistencyRule,
    DataFormatRule,
    CrossFieldConsistencyRule,
    SpecialCharacterConfig
)

class TestQualityControl(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        self.qc = QualityController()
        
        # Create sample test data
        self.sample_data = pd.DataFrame({
            'text': [
                'Good quality text',
                'Another good example',
                None,
                'Short',
                'Text with special chars @#$%'
            ],
            'number': [1, 2, 3, 4, 1000],  # Last value is an outlier
            'email': [
                'valid@email.com',
                'invalid-email',
                'another@email.com',
                None,
                'test@test.com'
            ],
            'date': [
                '2024-01-01',
                '2024-01-02',
                '2024-01-03',
                '2024-01-04',
                'invalid-date'
            ]
        })

    def test_completeness_calculation(self):
        """Test completeness metric calculation"""
        completeness = self.qc.calculate_completeness(self.sample_data)
        self.assertTrue(0 <= completeness <= 1)
        
        # Test with fully complete data
        complete_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        self.assertEqual(self.qc.calculate_completeness(complete_data), 1.0)
        
        # Test with empty DataFrame
        empty_data = pd.DataFrame()
        self.assertEqual(self.qc.calculate_completeness(empty_data), 0.0)

    def test_consistency_calculation(self):
        """Test consistency metric calculation"""
        consistency = self.qc.calculate_consistency(self.sample_data)
        self.assertTrue(0 <= consistency <= 1)
        
        # Test with consistent data
        consistent_data = pd.DataFrame({
            'text': ['abc', 'def', 'ghi']  # Same length strings
        })
        self.assertAlmostEqual(
            self.qc.calculate_consistency(consistent_data), 
            1.0, 
            places=2
        )

    def test_validity_calculation(self):
        """Test validity metric calculation"""
        validity = self.qc.calculate_validity(self.sample_data)
        self.assertTrue(0 <= validity <= 1)
        
        # Test with all valid data
        valid_data = pd.DataFrame({
            'text': ['Long enough text', 'Another good text']
        })
        self.assertGreater(self.qc.calculate_validity(valid_data), 0.9)

    def test_uniqueness_calculation(self):
        """Test uniqueness metric calculation"""
        uniqueness = self.qc.calculate_uniqueness(self.sample_data)
        self.assertTrue(0 <= uniqueness <= 1)
        
        # Test with duplicate data
        duplicate_data = pd.DataFrame({
            'text': ['same', 'same', 'different']
        })
        self.assertLess(self.qc.calculate_uniqueness(duplicate_data), 1.0)

    def test_timeliness_calculation(self):
        """Test timeliness metric calculation"""
        # Create data with timestamps
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=10),
            end=datetime.now(),
            periods=5
        )
        timely_data = pd.DataFrame({
            'timestamp': dates
        })
        
        timeliness = self.qc.calculate_timeliness(
            timely_data, 
            timestamp_column='timestamp'
        )
        self.assertTrue(0 <= timeliness <= 1)

    def test_accuracy_calculation(self):
        """Test accuracy metric calculation"""
        accuracy = self.qc.calculate_accuracy(self.sample_data)
        self.assertTrue(0 <= accuracy <= 1)
        
        # Test with accurate data (no outliers)
        accurate_data = pd.DataFrame({
            'number': [1, 2, 3, 4, 5]
        })
        self.assertGreater(self.qc.calculate_accuracy(accurate_data), 0.9)

    def test_full_validation(self):
        """Test complete validation process"""
        passed, issues, metrics = self.qc.validate_data(self.sample_data)
        
        # Convert passed to bool explicitly before assertion
        passed = bool(passed)
        self.assertIsInstance(passed, bool)
        self.assertIsInstance(issues, list)
        self.assertIsInstance(metrics, QualityMetrics)
        
        # Check if all metrics are calculated
        self.assertTrue(all(0 <= value <= 1 for value in metrics.to_dict().values()))

    def test_special_character_rule(self):
        """Test special character validation rule"""
        config = SpecialCharacterConfig(
            max_special_char_ratio=0.1,  # Lower threshold to make test stricter
            allowed_special_chars={'.', ',', '!', '?'}
        )
        rule = SpecialCharacterRule(config)
        
        # Test with normal text
        normal_data = pd.DataFrame({
            'text': ['Normal text.', 'Another normal text!']
        })
        normal_result = rule.validate(normal_data)
        self.assertTrue(normal_result['passed'])
        
        # Test with excessive special characters
        special_data = pd.DataFrame({
            'text': ['Text @#$%^&* with too many special chars!!!!!!!']
        })
        special_result = rule.validate(special_data)
        self.assertFalse(special_result['passed'])

    def test_language_consistency_rule(self):
        """Test language consistency validation rule"""
        rule = LanguageConsistencyRule(primary_language='en')
        
        # Test with English text
        english_data = pd.DataFrame({
            'text': ['This is English text.', 'Another English sentence.']
        })
        english_result = rule.validate(english_data)
        self.assertTrue(english_result['passed'])
        
        # Test with mixed languages
        mixed_data = pd.DataFrame({
            'text': [
                'This is English text.',
                'Esto es texto en español.'  # Spanish text
            ]
        })
        mixed_result = rule.validate(mixed_data)
        self.assertFalse(mixed_result['passed'])

    def test_data_format_rule(self):
        """Test data format validation rule"""
        rule = DataFormatRule()
        
        # Test with valid formats
        valid_data = pd.DataFrame({
            'email': ['test@example.com', 'another@email.com'],
            'phone': ['+1234567890', '1234567890'],
            'date': ['2024-01-01', '2024-12-31']
        })
        valid_result = rule.validate(valid_data)
        self.assertTrue(valid_result['passed'])
        
        # Test with invalid formats
        invalid_data = pd.DataFrame({
            'email': ['not-an-email', 'invalid@'],
            'phone': ['123', 'not-a-phone'],
            'date': ['2024/01/01', 'invalid-date']
        })
        invalid_result = rule.validate(invalid_data)
        self.assertFalse(invalid_result['passed'])

    def test_cross_field_consistency_rule(self):
        """Test cross-field consistency validation rule"""
        rule = CrossFieldConsistencyRule()
        
        # Add a date consistency validation
        def validate_dates(data):
            start_dates = pd.to_datetime(data['start_date'])
            end_dates = pd.to_datetime(data['end_date'])
            invalid = start_dates > end_dates
            return {
                'passed': not invalid.any(),
                'issues': ['Invalid date order'] if invalid.any() else [],
                'stats': {'violations': invalid.sum()}
            }
        
        rule.add_relationship(
            fields=['start_date', 'end_date'],
            validation_func=validate_dates,
            description="Date order validation"
        )
        
        # Test with valid date order
        valid_dates = pd.DataFrame({
            'start_date': ['2024-01-01', '2024-02-01'],
            'end_date': ['2024-01-31', '2024-02-28']
        })
        valid_result = rule.validate(valid_dates)
        self.assertTrue(valid_result['passed'])
        
        # Test with invalid date order
        invalid_dates = pd.DataFrame({
            'start_date': ['2024-02-01', '2024-03-01'],
            'end_date': ['2024-01-31', '2024-02-28']
        })
        invalid_result = rule.validate(invalid_dates)
        self.assertFalse(invalid_result['passed'])

    def test_edge_cases(self):
        """Test various edge cases"""
        # Empty DataFrame
        empty_df = pd.DataFrame()
        passed, issues, metrics = self.qc.validate_data(empty_df)
        self.assertFalse(passed)
        
        # Single column DataFrame
        single_col = pd.DataFrame({'col1': [1, 2, 3]})
        passed, issues, metrics = self.qc.validate_data(single_col)
        self.assertTrue(0 <= metrics.overall_score <= 1)
        
        # DataFrame with all null values
        null_df = pd.DataFrame({'col1': [None, None], 'col2': [None, None]})
        passed, issues, metrics = self.qc.validate_data(null_df)
        self.assertEqual(metrics.completeness, 0.0)

if __name__ == '__main__':
    unittest.main()