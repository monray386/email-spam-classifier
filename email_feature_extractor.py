import re
import numpy as np
import pandas as pd

class EmailFeatureExtractor:
    """
    Extract features from raw email text for the Spambase dataset.
    Automatically excludes low-importance or non-generalizable columns.
    """
    
    WORDS = [
        "make", "address", "all", "3d", "our", "over", "remove", "internet",
        "order", "mail", "receive", "will", "people", "report", "addresses",
        "free", "business", "email", "you", "credit", "your", "font", "000",
        "money", "data", "technology", "pm", "cs", "meeting",
        "original", "project", "re", "edu", "conference"
    ]
    
    # Special characters mapping
    CHAR_RENAME = {
        ";": "semicolon",
        "(": "parentheses",
        "[": "sqbrackets",
        "!": "exclamation",
        "$": "dollar",
        "#": "hashtag"
    }
    
    CHARS = list(CHAR_RENAME.keys())
    
    def __init__(self, email_text):
        self.email_text = email_text.lower()
        self.features = {}
        self._extract()
    
    def _extract(self):
        # Word frequencies
        words_list = self.email_text.split()
        total_words = len(words_list)
        for word in self.WORDS:
            count = words_list.count(word)
            self.features[f"word_freq_{word}"] = (count / total_words * 100) if total_words > 0 else 0
        
        # Character frequencies
        total_chars = len(self.email_text)
        for char in self.CHARS:
            count = self.email_text.count(char)
            self.features[f"char_freq_{char}"] = (count / total_chars * 100) if total_chars > 0 else 0
        
        # Capital run lengths
        caps = re.findall(r"[A-Z]+", self.email_text)
        self.features["capital_run_length_average"] = np.mean([len(c) for c in caps]) if caps else 0
        self.features["capital_run_length_longest"] = max([len(c) for c in caps]) if caps else 0
        self.features["capital_run_length_total"] = sum([len(c) for c in caps]) if caps else 0
    
    def to_dict(self):
        return self.features
    
    def to_dataframe(self):
        df = pd.DataFrame([self.features])
        # Rename character columns to match training set
        rename_dict = {f"char_freq_{k}": f"char_freq_{v}" for k,v in self.CHAR_RENAME.items()}
        df = df.rename(columns=rename_dict)
        return df
