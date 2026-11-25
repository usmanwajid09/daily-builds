"""Search a small document set with TF-IDF + cosine similarity.

Fits the vectorizer on six short documents (cats, dogs, machine learning),
then searches for "machine learning models" and prints the ranked hits
with their cosine similarity scores.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tfidf_vectorizer import TfidfVectorizer

DOCUMENTS = [
    "Cats sleep all day and chase mice in the garden",
    "Dogs love walks and bark at strangers in the park",
    "Machine learning models learn patterns from data",
    "Deep learning uses neural networks to classify images",
    "Cats and dogs are popular pets in many homes",
    "Training a model requires data and careful evaluation",
]
QUERY = "machine learning models"


def main():
    vectorizer = TfidfVectorizer()
    vectorizer.fit(DOCUMENTS)
    results = vectorizer.search(QUERY, top_k=3)
    print(f"Documents: {len(DOCUMENTS)}, vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f'Query: "{QUERY}"\n')
    for rank, (doc_index, score) in enumerate(results, start=1):
        print(f"  #{rank}  doc {doc_index}  score {score:.4f}")
        print(f"        {DOCUMENTS[doc_index]}")
    top_index, top_score = results[0]
    print(f"\nTop hit: doc {top_index} (score {top_score:.4f})")


if __name__ == "__main__":
    main()
