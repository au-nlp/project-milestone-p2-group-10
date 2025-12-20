import os
import json
import re
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from wtpsplit import SaT
# We import the logic we built in the Hybrid script
# (Ensure Hybrid_TextTiling_Segmenter.py is in the same folder)
from Hybrid_TextTiling_Segmenter import find_topical_segments, sat_model, embed_model # importing models from already loaded module
from coherence_eval import calculate_segmentation_quality

# --- Configuration ---
# Path to your massive 25GB JSONL file
JSONL_PATH = r"C:\Users\Sadik\Desktop\NLP Project\episodeLevelDataSample.jsonl"
OUTPUT_DIR = "processed_vectors_sample2_500"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# How many episodes to test with?
MAX_EPISODES_TO_PROCESS = 500


# --- 2. Streaming Processing Loop ---
print("--- Streaming Processing Loop ---")
print(f"Starting stream processing of: {JSONL_PATH}")

episodes_processed = 0
pattern = r"\[.*?\]"


# Open the HUGE file in read mode. This does NOT load it into RAM.
# 'encoding="utf-8"' is crucial for text data.
with open(JSONL_PATH, 'r', encoding='utf-8') as f:
    
    # We loop through the file one line at a time
    for line_idx, line in enumerate(f):
        
        # Stop if we hit our test limit
        if episodes_processed >= MAX_EPISODES_TO_PROCESS:
            print(f"\nReached limit of {MAX_EPISODES_TO_PROCESS} episodes. Stopping.")
            break
            
        try:
            # Parse the single line as JSON
            data = json.loads(line)
            
            # Access the transcript (adjust key if your data uses something else)
            text = data.get("transcript", "")
            
            #getting the datetime for dynamic topic modeling
            episode_date = data.get("oldestEpisodeDate", None)

            # Use 'episode_id' if available, otherwise use line index
            episode_id = data.get("episode_id", f"episode_{line_idx}")
            
            # Skip empty transcripts
            if not text or len(text) < 500:
                continue

            print(f"\n ✅ Processing Episode {episodes_processed + 1} (ID: {episode_id})")

            # --- A. Pre-calculate Length for Dynamic K ---
            # We use sat_model to count sentences first

            print("---- 1. Cleaning up with wtpsplit and regex ----")
            raw_sents = re.sub(pattern," ", text)
            raw_sents = sat_model.split(raw_sents, do_paragraph_segmentation=False)
            clean_sents = [s for s in raw_sents if len(s.split()) > 3]
            num_sents = len(clean_sents)
            
            
            if num_sents < 30:
                print(f"  -> Skipped: Too short ({num_sents} sentences)")
                continue

            best_score = -float('inf')
            best_segments = []
            best_k = 0

            k_list = [3, 5, 7, 9, 11, 15]

            # using complete search to find the optimal k 
            for k_size in k_list:

                segments, _ = find_topical_segments(
                    clean_sents, 
                    k=k_size,
                    min_sentence_len=3,
                    smooth_passes=1,
                    smooth_window=2,
                    threshold_factor=0.5
                )

                if not segments:
                    print("  -> No segments found.")
                    continue

                # customized segmentation quality evaluator
                score = calculate_segmentation_quality(segments, embed_model, sat_model, penalty_weight=0.2)
                
                # Check if winner
                if score > best_score:
                    best_score = score
                    best_segments = segments
                    best_k = k_size
                
                if not best_segments:
                    print("  -> No valid segments found after tuning.")
                    continue

            print(f"\n\n  -> Episode id: {episode_id} Sentence in episode: {num_sents} \
                  \n Best k={best_k} (Score: {best_score:.3f}). Found {len(best_segments)} segments.")

            # --- C. Embed Segments ---
            # Create vectors for the segments
            segment_vectors = embed_model.encode(best_segments)
            
            # --- D. Save Lightweight Data ---
            save_filename = f"{episode_id}_vectors.npy"
            # Sanitize filename just in case
            save_filename = "".join([c for c in save_filename if c.isalnum() or c in (' ', '.', '_')]).strip()
            save_path = os.path.join(OUTPUT_DIR, save_filename)
            seg_len = len(best_segments)
            data_package = {
                "episode_id": episode_id,
                "timestamp": episode_date,   # for dynamic modeling overtime
                "vectors": segment_vectors,  # The math (for global clustering)
                "text_snippets": best_segments, # entire segments
                "segment_count": seg_len
            }
            
            np.save(save_path, data_package)
            print(f"  -> Saved {seg_len} segments to {save_filename}")
            
            episodes_processed += 1

        except json.JSONDecodeError:
            print(f"  -> Error: Line {line_idx} is not valid JSON. Skipping.")
        except Exception as e:
            print(f"  -> Error processing line {line_idx}: {e}")

print("\nProcessing complete!")