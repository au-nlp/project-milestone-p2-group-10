import os
import json
import re
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from wtpsplit import SaT

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
        # --- Use wtpsplit for robust splitting ---
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


    # DEBUG PRINT
    print(f"    -> Intra: {avg_intra:.3f} | Inter: {avg_inter:.3f} | Pen: {penalty:.3f} | Score: {quality_score:.3f}")
    
    return quality_score