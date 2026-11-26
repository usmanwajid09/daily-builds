"""Skip-gram word embeddings (word2vec) from scratch."""

from word2vec_skipgram.model import STOPWORDS, SkipGramWord2Vec, tokenize

__all__ = ["SkipGramWord2Vec", "tokenize", "STOPWORDS"]
