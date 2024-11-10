from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import spacy
from transformers import pipeline
import torch
from collections import Counter
import re
from .processors import (
    EntityProcessor,
    SentimentProcessor,
    TopicProcessor,
    KeywordProcessor,
    LanguageProcessor,
    InstructionProcessor
)

class AnnotationType(Enum):
    ENTITIES = "entities"
    SENTIMENT = "sentiment"
    TOPICS = "topics"
    KEYWORDS = "keywords"
    LANGUAGE = "language"
    INSTRUCTION = "instruction"

@dataclass
class AnnotationConfig:
    enable_entities: bool = True
    enable_sentiment: bool = True
    enable_topics: bool = True
    enable_keywords: bool = True
    enable_language: bool = True
    enable_instruction: bool = True
    min_keyword_score: float = 0.3
    max_keywords: int = 10
    sentiment_threshold: float = 0.1
    model_path: str = "en_core_web_sm"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32

class AnnotationSystem:
    """Main annotation system that coordinates different annotation processors"""
    
    def __init__(self, config: Optional[AnnotationConfig] = None):
        self.config = config or AnnotationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize processors
        try:
            self._initialize_processors()
            self.logger.info("Successfully initialized annotation processors")
        except Exception as e:
            self.logger.error(f"Error initializing annotation processors: {e}")
            raise
            
    def _initialize_processors(self):
        """Initialize all annotation processors"""
        # Load spaCy model
        self.nlp = spacy.load(self.config.model_path)
        
        # Initialize individual processors
        self.processors = {
            AnnotationType.ENTITIES: EntityProcessor(self.nlp),
            AnnotationType.SENTIMENT: SentimentProcessor(device=self.config.device),
            AnnotationType.TOPICS: TopicProcessor(self.nlp),
            AnnotationType.KEYWORDS: KeywordProcessor(
                self.nlp,
                min_score=self.config.min_keyword_score,
                max_keywords=self.config.max_keywords
            ),
            AnnotationType.LANGUAGE: LanguageProcessor(self.nlp),
            AnnotationType.INSTRUCTION: InstructionProcessor()
        }
        
    def annotate_text(self, text: str) -> Dict[str, Any]:
        """
        Annotate a single text with all enabled processors
        """
        try:
            if not text or not isinstance(text, str):
                return {}
                
            doc = self.nlp(text)
            annotations = {}
            
            # Run enabled processors
            if self.config.enable_entities:
                annotations['entities'] = self.processors[AnnotationType.ENTITIES].process(doc)
                
            if self.config.enable_sentiment:
                annotations['sentiment'] = self.processors[AnnotationType.SENTIMENT].process(text)
                
            if self.config.enable_topics:
                annotations['topics'] = self.processors[AnnotationType.TOPICS].process(doc)
                
            if self.config.enable_keywords:
                annotations['keywords'] = self.processors[AnnotationType.KEYWORDS].process(doc)
                
            if self.config.enable_language:
                annotations['language'] = self.processors[AnnotationType.LANGUAGE].process(doc)
                
            if self.config.enable_instruction:
                annotations['instruction'] = self.processors[AnnotationType.INSTRUCTION].process(text)
                
            return annotations
            
        except Exception as e:
            self.logger.error(f"Error annotating text: {e}")
            return {}
            
    def process_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Process a batch of texts with all enabled processors
        """
        annotations = []
        
        try:
            # Process texts in batches
            for i in range(0, len(texts), self.config.batch_size):
                batch = texts[i:i + self.config.batch_size]
                batch_annotations = []
                
                for text in batch:
                    if isinstance(text, str) and text.strip():
                        annotation = self.annotate_text(text)
                        batch_annotations.append(annotation)
                    else:
                        batch_annotations.append({})
                        
                annotations.extend(batch_annotations)
                
        except Exception as e:
            self.logger.error(f"Error processing batch: {e}")
            
        return annotations
        
    def get_supported_types(self) -> List[str]:
        """Get list of supported annotation types"""
        return [ant_type.value for ant_type in AnnotationType]