from typing import List, Dict, Any
import spacy
from transformers import pipeline
from collections import Counter
import re

class BaseProcessor:
    """Base class for annotation processors"""
    
    def __init__(self):
        self.name = self.__class__.__name__
        
    def process(self, input_data: Any) -> Dict[str, Any]:
        raise NotImplementedError("Processors must implement process method")
        
class EntityProcessor(BaseProcessor):
    def __init__(self, nlp):
        super().__init__()
        self.nlp = nlp
        
    def process(self, doc) -> List[Dict[str, str]]:
        return [{
            'text': ent.text,
            'label': ent.label_,
            'start': ent.start_char,
            'end': ent.end_char
        } for ent in doc.ents]
        
class SentimentProcessor(BaseProcessor):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=device
        )
        
    def process(self, text: str) -> Dict[str, Any]:
        result = self.sentiment_analyzer(text)[0]
        score = result['score']
        if result['label'] == 'NEGATIVE':
            score = -score
            
        return {
            'score': score,
            'label': 'positive' if score > 0.1 else 'negative' if score < -0.1 else 'neutral'
        }
        
class TopicProcessor(BaseProcessor):
    def __init__(self, nlp):
        super().__init__()
        self.nlp = nlp
        
    def process(self, doc) -> List[str]:
        # Extract noun chunks and entities
        topics = []
        seen = set()
        
        # Add noun chunks
        for chunk in doc.noun_chunks:
            if chunk.text.lower() not in seen:
                topics.append(chunk.text)
                seen.add(chunk.text.lower())
                
        # Add named entities
        for ent in doc.ents:
            if ent.text.lower() not in seen:
                topics.append(ent.text)
                seen.add(ent.text.lower())
                
        return topics
        
class KeywordProcessor(BaseProcessor):
    def __init__(self, nlp, min_score: float = 0.3, max_keywords: int = 10):
        super().__init__()
        self.nlp = nlp
        self.min_score = min_score
        self.max_keywords = max_keywords
        
    def process(self, doc) -> List[Dict[str, Any]]:
        keywords = []
        seen = set()
        
        for token in doc:
            if (not token.is_stop and not token.is_punct and token.pos_ in ['NOUN', 'PROPN', 'ADJ']):
                importance = (token.prob * -1)  # Using negative log probability
                
                if importance > self.min_score:
                    if token.text.lower() not in seen:
                        keywords.append({
                            'text': token.text,
                            'pos': token.pos_,
                            'score': float(importance)
                        })
                        seen.add(token.text.lower())
                        
        # Sort by importance and limit
        keywords.sort(key=lambda x: x['score'], reverse=True)
        return keywords[:self.max_keywords]
        
class LanguageProcessor(BaseProcessor):
    def __init__(self, nlp):
        super().__init__()
        self.nlp = nlp
        
    def process(self, doc) -> Dict[str, Any]:
        return {
            'language': doc.lang_,
            'confidence': 1.0 if doc.lang_ else 0.0
        }
        
class InstructionProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.instruction_patterns = [
            r"^(how to|what is|explain|describe|tell me|can you|please)",
            r"\?(.*?)$",
            r"^(if|when|why|where|who|whose|which|whatever|whom)"
        ]
        
    def process(self, text: str) -> Dict[str, Any]:
        is_instruction = any(re.search(pattern, text.lower()) for pattern in self.instruction_patterns)
        
        parts = {
            'is_instruction': is_instruction,
            'has_question_mark': '?' in text,
            'command_words': self._extract_command_words(text)
        }
        
        return parts
        
    def _extract_command_words(self, text: str) -> List[str]:
        command_words = [
            'explain', 'describe', 'list', 'analyze', 'compare',
            'define', 'elaborate', 'illustrate', 'show', 'tell'
        ]
        
        found_commands = []
        for word in text.lower().split():
            if word in command_words:
                found_commands.append(word)
                
        return found_commands