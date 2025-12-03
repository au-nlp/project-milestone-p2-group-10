import os
import json
import re
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from wtpsplit import SaT
# We import the logic we built in the Hybrid script
from Hybrid_TextTiling_Segmenter import find_topical_segments, sat_model, embed_model

# --- Configuration ---
# Path to your massive 25GB JSONL file
JSONL_PATH = r"C:\Users\Sadik\Desktop\NLP Project\episodeLevelDataSample.jsonl"

# How many episodes to test with?
MAX_EPISODES_TO_PROCESS = 15
# --- 2. Streaming Processing Loop ---
print("--- Streaming Processing Loop ---")
print(f"Starting stream processing of: {JSONL_PATH}")

episodes_processed = 0
pattern = r"\[.*?\]"


def calculate_segmentation_quality(segments, embed_model, sat_model, penalty_weight=0.1):
    """
    Calculates a Balanced Quality Score using robust sentence splitting.
    
    Args:
        segments (list[str]): The list of text segments to evaluate.
        embed_model: The SentenceTransformer model.
        sat_model: The wtpsplit SaT model (for robust sentence counting).
        penalty_weight (float): How much to punish over-segmentation.
        
    Returns:
        float: The quality score.
    """
    if len(segments) < 2:
        return 0.0
        
    # 1. Inter-Segment Similarity (We want this LOW)
    # Get embeddings for each full segment (treat segment as one unit)
    seg_embeddings = embed_model.encode(segments, show_progress_bar=False)
    
    inter_sims = []
    for i in range(len(seg_embeddings) - 1):
        sim = util.cos_sim(seg_embeddings[i], seg_embeddings[i+1]).item()
        inter_sims.append(sim)
    avg_inter = np.mean(inter_sims) if inter_sims else 0.0
    
    # 2. Intra-Segment Similarity (We want this HIGH)
    intra_sims = []
    total_sentences = 0
    
    for segment in segments:
        # --- CRITICAL FIX: Use wtpsplit for robust splitting ---
        # We use the model to find true sentence boundaries
        sents = sat_model.split(segment, do_paragraph_segmentation=False)
        
        # Filter tiny noise (like "Okay.") to avoid inflating the count
        valid_sents = [s for s in sents if len(s.split()) > 3]
        
        num_valid = len(valid_sents)
        total_sentences += num_valid
        
        if num_valid < 2:
            # If segment is just 1 sentence, it's perfectly coherent by definition,
            # but we don't want to reward this too much (penalty will handle it).
            intra_sims.append(1.0)
            continue
            
        sent_embs = embed_model.encode(valid_sents, show_progress_bar=False)
        centroid = np.mean(sent_embs, axis=0)
        
        # Calculate how close each sentence is to the segment's "center"
        sims = util.cos_sim(sent_embs, centroid).mean().item()
        intra_sims.append(sims)
        
    avg_intra = np.mean(intra_sims) if intra_sims else 0.0
    
    # 3. The Fragmentation Penalty
    # We use the robust 'total_sentences' count we just calculated
    fragmentation_ratio = len(segments) / max(1, total_sentences)
    penalty = penalty_weight * fragmentation_ratio
    
    # Final Balanced Score
    quality_score = (avg_intra - avg_inter) - penalty


    # DEBUG PRINT (Remove later)
    print(f"    -> Intra: {avg_intra:.3f} | Inter: {avg_inter:.3f} | Pen: {penalty:.3f} | Score: {quality_score:.3f}")
    
    return quality_score



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
            
            
            if num_sents < 30:
                print(f"  -> Skipped: Too short ({num_sents} sentences)")
                continue
            
            best_score = -float('inf')
            best_segments = []
            best_k = 0

            k_list = [3, 5, 7, 9, 11, 15]

            for k_size in k_list:

                # Note: We pass the 'dynamic k' here
                segments, _ = find_topical_segments(
                    clean_sents, 
                    k=k_size,
                    min_sentence_len=3,
                    smooth_passes=1,
                    smooth_window=3,
                    threshold_factor=0.5
                )

                if not segments:
                    print("  -> No segments found.")
                    continue

                score = calculate_segmentation_quality(segments, embed_model, sat_model, penalty_weight=0.5)
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
            
            episodes_processed += 1

        except json.JSONDecodeError:
            print(f"  -> Error: Line {line_idx} is not valid JSON. Skipping.")
        except Exception as e:
            print(f"  -> Error processing line {line_idx}: {e}")

print("\nProcessing complete!")