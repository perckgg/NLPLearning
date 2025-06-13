import re
import spacy
from nltk.tokenize import RegexpTokenizer

nlp = spacy.load("en_core_web_sm")

class AdvancedEmailTokenizer:
    def __init__(self):
        # Patterns for special tokens
        self.patterns = {
            'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            'url': r'https?://\S+|www\.\S+',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',
            'time': r'\b\d{1,2}:\d{2}(?::\d{2})?\b(?: [aApP][mM])?\b'
        }
        
        # Combined pattern for special tokens
        self.special_pattern = re.compile(
            '(' + '|'.join(f'(?P<{name}>{pattern})' 
                          for name, pattern in self.patterns.items()) + ')',
            flags=re.IGNORECASE
        )
        
        # Regular tokenizer for non-special text
        self.word_tokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')
    
    def tokenize(self, text):
        # Find all special tokens and their positions
        matches = list(self.special_pattern.finditer(text))
        
        # Split text into segments
        segments = []
        prev_end = 0
        for match in matches:
            start, end = match.span()
            if start > prev_end:
                segments.append(('text', text[prev_end:start]))
            segments.append(('special', match.group()))
            prev_end = end
        if prev_end < len(text):
            segments.append(('text', text[prev_end:]))
        
        # Tokenize each segment appropriately
        tokens = []
        for seg_type, content in segments:
            if seg_type == 'special':
                tokens.append(content)
            else:
                tokens.extend(self.word_tokenizer.tokenize(content))
        
        return tokens

# Example usage
email_text = """
Dear All,

The budget meeting is rescheduled to 03/25/2023 at 3:00 PM in Conference Room B. 
Please confirm attendance to abc+23@xy3-z.vn by 03/22. 
Documents available at: http://company.net/budget/q1-2023

Best,
Finance Team
+84911449664
"""

tokenizer = AdvancedEmailTokenizer()
tokens = tokenizer.tokenize(email_text)
print("\nHybrid Email Tokenizer Results:")
for i, token in enumerate(tokens, 1):
    print(f"{i}. {token}")
