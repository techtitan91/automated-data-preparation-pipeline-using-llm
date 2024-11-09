from typing import List, Dict, Any, Tuple
import spacy
import re
from dataclasses import dataclass
import logging
import unicodedata

@dataclass
class CleaningConfig:
    min_text_length: int = 20
    max_text_length: int = 1000
    quality_threshold: float = 0.8
    remove_urls: bool = True
    fix_spacing: bool = True
    remove_special_chars: bool = True

class AdvancedCleaner:
    def __init__(self, config: CleaningConfig = None):
        self.config = config or CleaningConfig()
        self.nlp = spacy.load("en_core_web_sm")
        self.logger = logging.getLogger(__name__)
        
    def remove_boilerplate(self, text: str) -> str:
        """Remove common boilerplate content"""
        # Remove common headers/footers
        text = re.sub(r'(?i)(confidential|private|all rights reserved|copyright)', '', text)
        # Remove email signatures
        text = re.sub(r'(?i)best regards.*$', '', text, flags=re.MULTILINE | re.DOTALL)
        return text.strip()
    
    def standardize_format(self, text: str) -> str:
        """Fix common formatting issues"""
        if self.config.fix_spacing:
            # Preserve double newlines first
            text = re.sub(r'\n\s*\n', '\n\n', text)
            # Fix multiple spaces (but don't touch the preserved newlines)
            lines = text.split('\n')
            lines = [re.sub(r'\s+', ' ', line.strip()) for line in lines]
            text = '\n'.join(lines)
        return text.strip()
    
    def clean_special_chars(self, text: str) -> str:
        """Handle special characters"""
        if self.config.remove_special_chars:
            # Remove control characters
            text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
            # Replace smart quotes with standard quotes
            text = text.replace('"', '"').replace('"', '"')
            text = text.replace(''', "'").replace(''', "'")
        return text
    
    def meets_quality_standards(self, text: str) -> bool:
        """Check if text meets quality standards"""
        if not text or not isinstance(text, str):
            return False
            
        text_length = len(text)
        if text_length < self.config.min_text_length or text_length > self.config.max_text_length:
            return False
            
        # Check for minimum word count
        word_count = len(text.split())
        if word_count < 3:
            return False
            
        # Check for language quality using spaCy
        doc = self.nlp(text)
        if len(doc.ents) == 0 and len(doc) > 50:  # Long text with no entities might be garbage
            return False
            
        return True
        
    def clean_batch(self, texts: List[str]) -> List[str]:
        """Process a batch of texts with advanced cleaning"""
        cleaned_texts = []
        for text in texts:
            try:
                if not isinstance(text, str):
                    continue
                    
                # Remove boilerplate content
                text = self.remove_boilerplate(text)
                
                # Fix formatting issues
                text = self.standardize_format(text)
                
                # Handle special characters
                text = self.clean_special_chars(text)
                
                # Validate text quality
                if self.meets_quality_standards(text):
                    cleaned_texts.append(text)
                
            except Exception as e:
                self.logger.error(f"Error cleaning text: {e}")
                continue
                
        return cleaned_texts