import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Stopwords to ignore (expand this list as needed)
CUSTOM_STOPWORDS = [
    'the', 'and', 'to', 'of', 'a', 'in', 'that', 'is', 'it', 'for', 'you', 'we', 
    'on', 'this', 'are', 'as', 'with', 'be', 'have', 'but', 'not', 'they', 'at',
    'so', 'was', 'if', 'or', 'what', 'do', 'can', 'about', 'just', 'like', 'my',
    'your', 'all', 'there', 'people', 'would', 'know', 'from', 'get', 'going',
    'think', 'really', 'one', 'time', 'because', 'intro', 'episode', 'podcast'
]

def explain_drift_drivers(folder_path, top_n=10):
    print(f"--- Extracting Vocabulary Drivers for {folder_path} ---")
    
    # 1. Aggregate Text by Year
    text_by_year = {} # {2015: ["text segment 1", "text segment 2"], 2016: ...}
    
    files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    
    for filename in files:
        try:
            data = np.load(os.path.join(folder_path, filename), allow_pickle=True).item()
            timestamp = data.get('timestamp')
            text_snippets = data.get('text_snippets', [])
            
            if timestamp is None or not text_snippets:
                continue
                
            year = pd.to_datetime(timestamp).year
            
            # Filter for reasonable years
            if year < 2010 or year > 2025:
                continue
                
            if year not in text_by_year:
                text_by_year[year] = []
            
            # Combine all segments into one massive string for that file
            full_text = " ".join(text_snippets)
            text_by_year[year].append(full_text)
            
        except Exception as e:
            continue

    if not text_by_year:
        print("No text data found.")
        return

    # 2. Prepare Corpus for TF-IDF
    # We treat "All text from 2015" as one document, "All text from 2016" as another.
    sorted_years = sorted(text_by_year.keys())
    corpus = [" ".join(text_by_year[y]) for y in sorted_years]
    
    print(f"Analyzing {len(corpus)} years of content: {sorted_years}")

    # 3. TF-IDF to find DISTINCTIVE words
    # (Words that are high frequency in THIS year, but low frequency in OTHERS)
    vectorizer = TfidfVectorizer(
        stop_words=CUSTOM_STOPWORDS, 
        max_df=0.7,       # Ignore words that appear in >70% of years (removes generic words)
        min_df=1,         # Keep words even if they only appear in 1 year
        ngram_range=(1,2) # Look for single words AND bigrams (e.g., "mental health")
    )
    
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    # 4. Extract Top Words per Year
    print("\n" + "="*40)
    print("  KEYWORD EVOLUTION (Distinctive Terms)")
    print("="*40)
    
    df_keywords = pd.DataFrame()

    for i, year in enumerate(sorted_years):
        # Get the row for this year
        year_vector = tfidf_matrix[i]
        
        # Sort indices by score (highest first)
        sorted_indices = year_vector.toarray()[0].argsort()[::-1]
        
        # Get top N words
        top_words = feature_names[sorted_indices][:top_n]
        
        print(f"\n📅 {year}: {', '.join(top_words)}")
        
        # Save to dataframe for report
        df_keywords[year] = top_words

    return df_keywords