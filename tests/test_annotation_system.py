import unittest
from src.annotation_system.base import AnnotationSystem, AnnotationConfig
import spacy

class TestAnnotationSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = AnnotationConfig(
            enable_entities=True,
            enable_sentiment=True,
            enable_topics=True,
            enable_keywords=True,
            enable_language=True,
            enable_instruction=True
        )
        cls.system = AnnotationSystem(cls.config)
        
    def test_basic_annotation(self):
        text = "Apple Inc. is planning to release a new iPhone next year. The company's CEO Tim Cook is very excited about it."
        result = self.system.annotate_text(text)
        
        self.assertIn('entities', result)
        self.assertIn('sentiment', result)
        self.assertIn('topics', result)
        self.assertIn('keywords', result)
        self.assertIn('language', result)
        
    def test_entity_recognition(self):
        text = "Microsoft CEO Satya Nadella announced new AI features."
        result = self.system.annotate_text(text)
        
        entities = result['entities']
        self.assertTrue(any(entity['text'] == 'Microsoft' for entity in entities))
        self.assertTrue(any(entity['text'] == 'Satya Nadella' for entity in entities))
        
    def test_sentiment_analysis(self):
        positive_text = "This is absolutely wonderful news!"
        negative_text = "This is terrible and disappointing."
        
        pos_result = self.system.annotate_text(positive_text)
        neg_result = self.system.annotate_text(negative_text)
        
        self.assertEqual(pos_result['sentiment']['label'], 'positive')
        self.assertEqual(neg_result['sentiment']['label'], 'negative')
        
    def test_instruction_detection(self):
        instruction_text = "Can you explain how neural networks work?"
        regular_text = "Neural networks are computational models."
        
        instr_result = self.system.annotate_text(instruction_text)
        reg_result = self.system.annotate_text(regular_text)
        
        self.assertTrue(instr_result['instruction']['is_instruction'])
        self.assertFalse(reg_result['instruction']['is_instruction'])
        
    def test_batch_processing(self):
        texts = [
            "Apple is a tech company.",
            "Google develops Android.",
            ""  # Empty text
        ]
        results = self.system.process_batch(texts)
        
        self.assertEqual(len(results), 3)
        self.assertNotEqual(results[0], {})  # Should have annotations
        self.assertNotEqual(results[1], {})  # Should have annotations
        self.assertEqual(results[2], {})  # Should be empty for empty text
        
    def test_error_handling(self):
        # Test with invalid input
        result = self.system.annotate_text(None)
        self.assertEqual(result, {})
        
        # Test with non-string input
        result = self.system.annotate_text(123)
        self.assertEqual(result, {})
        
if __name__ == '__main__':
    unittest.main()