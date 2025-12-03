import os
import json
import re
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from wtpsplit import SaT
# We import the logic we built in the Hybrid script
# (Ensure Hybrid_TextTiling_Segmenter.py is in the same folder)
from Hybrid_TextTiling_Segmenter import find_topical_segments, sat_model, embed_model

# --- Configuration ---
# Path to your massive 25GB JSONL file
JSONL_PATH = r"C:\Users\Sadik\Desktop\NLP Project\episodeLevelDataSample.jsonl"
OUTPUT_DIR = "processed_vectors"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# How many episodes to test with?
MAX_EPISODES_TO_PROCESS = 500

# --- 1. Load Models Once ---
# print("Loading 'sat-12l-sm' (wtpsplit) and all-mpnet-base-v2 model...")
# sat_model = SaT("sat-12l-sm")
# embed_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# try:
#     if torch.cuda.is_available():
#         print("CUDA available. Moving models to GPU.")
#         sat_model.to("cuda")
#         sat_model.half()
#         embed_model.to("cuda")
#     else:
#         print("Running on CPU.")
# except Exception as e:
#     print(f"GPU setup failed: {e}")

# --- Helper: Dynamic K (Block Size) ---
def get_dynamic_k(num_sentences):
    if num_sentences < 30: return None 
    elif num_sentences < 100: return 3 
    elif num_sentences < 300: return 5
    elif num_sentences < 600: return 7
    elif num_sentences < 800: return 9
    elif num_sentences < 1000: return 11
    else: return 15

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
            
            k_size = get_dynamic_k(num_sents)
            
            if k_size is None:
                print(f"  -> Skipped: Too short ({num_sents} sentences)")
                continue

            print(f"  -> Length: {num_sents} sentences. Using k={k_size}")

            # --- B. Segment ---
            # Note: We pass the 'dynamic k' here
            segments, _ = find_topical_segments(
                clean_sents, 
                k=k_size,
                min_sentence_len=3,
                smooth_passes=1,
                smooth_window=1,
                threshold_factor=0.5
            )

            if not segments:
                print("  -> No segments found.")
                continue

            # --- C. Embed Segments ---
            # Create vectors for the segments
            segment_vectors = embed_model.encode(segments)
            
            # --- D. Save Lightweight Data ---
            save_filename = f"{episode_id}_vectors.npy"
            # Sanitize filename just in case
            save_filename = "".join([c for c in save_filename if c.isalnum() or c in (' ', '.', '_')]).strip()
            save_path = os.path.join(OUTPUT_DIR, save_filename)
            
            data_package = {
                "episode_id": episode_id,
                "vectors": segment_vectors,  # The math (for global clustering)
                "text_snippets": segments, #[s[:100] + "..." for s in segments], # Preview
                "segment_count": len(segments)
            }
            
            np.save(save_path, data_package)
            print(f"  -> Saved {len(segments)} segments to {save_filename}")
            
            episodes_processed += 1

        except json.JSONDecodeError:
            print(f"  -> Error: Line {line_idx} is not valid JSON. Skipping.")
        except Exception as e:
            print(f"  -> Error processing line {line_idx}: {e}")

print("\nProcessing complete!")