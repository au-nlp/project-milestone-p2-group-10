
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer

from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired

def topic_modeling():
  # Step 1 - Extract embeddings
  embedding_model = SentenceTransformer("all-mpnet-base-v2")

  # Step 2 - Reduce dimensionality
  umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)

  # Step 3 - Cluster reduced embeddings
  hdbscan_model = HDBSCAN(min_cluster_size=15, min_samples = 10, metric='euclidean', cluster_selection_method='leaf', prediction_data=True)

  # Step 4 - Tokenize topics
  vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(2,3))

  # Step 5 - Create topic representation
  ctfidf_model = ClassTfidfTransformer()

  # Step 6 - (Optional) Fine-tune topic representations with
  # a `bertopic.representation` model
  representation_model = KeyBERTInspired()

  # All steps together
  topic_model = BERTopic(
    embedding_model=embedding_model,          # Step 1 - Extract embeddings
    umap_model=umap_model,                    # Step 2 - Reduce dimensionality
    hdbscan_model=hdbscan_model,              # Step 3 - Cluster reduced embeddings
    vectorizer_model=vectorizer_model,        # Step 4 - Tokenize topics
    ctfidf_model=ctfidf_model,                # Step 5 - Extract topic words
    representation_model=representation_model, # Step 6 - (Optional) Fine-tune topic representations
    verbose = True
  )

  return topic_model