import spacy
from typing import List

class ImprovedTokenizer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        self.MEANINGFUL_BIGRAMS = {
            ("ADJ", "NOUN"),
            ("NOUN", "NOUN"),
            ("VERB", "NOUN"),
        }
    # Keep only the most meaningful syntactic bigram patterns
    def __call__(self, text: str, max_tokens=200) -> List[str]:
        doc = self.nlp(text[:3000])  # truncate long docs for speed
        tokens = [(t.text, t.lemma_.lower(), t.pos_) for t in doc if t.is_alpha]
       
        bigrams = []
        for i in range(len(tokens) - 1):
            word1, lemma1, pos1 = tokens[i]
            word2, lemma2, pos2 = tokens[i + 1]
            if (pos1, pos2) in self.MEANINGFUL_BIGRAMS:
                # Optionally lowercase both words to normalize
                bigrams.append(f"{lemma1} {lemma2}")
       
        return bigrams