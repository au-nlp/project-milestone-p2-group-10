import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from wtpsplit import SaT
import matplotlib.pyplot as plt

# --- Model Loading ---
# We load both models globally to be efficient.

# Model 1: For splitting text into clean sentences
print("Loading 'sat-12l-sm' (wtpsplit) model...")
sat_model = SaT("sat-12l-sm")

# Model 2: For getting sentence embeddings
print("Loading 'all-mpnet-base-v2' (embedding) model...")
embed_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Move models to GPU if available
try:
    if torch.cuda.is_available():
        print("CUDA available. Moving models to GPU.")
        sat_model.to("cuda")
        sat_model.half()
        embed_model.to("cuda")
    else:
        print("Running on CPU.")
except Exception as e:
    print(f"GPU setup failed: {e}")

# --- TextTiling Algorithm Functions (from paper) ---

def block_comparison_score(embs_stack, k=10): # <-- PARAMETER RENAMED
    """
    Implements the block comparison from TextTiling.
    Calculates similarity score for each "gap" between sentences.
    
    Args:
        embs_stack (torch.Tensor): A 2D Tensor of (num_sentences, emb_dim).
                                   This is PRE-STACKED.
        k (int): The block size (number of sentences to compare on
                 either side of the gap).
    
    Returns:
        list[float]: A list of similarity scores.
    """
    if len(embs_stack) < k * 2:
        # Not enough sentences to compare
        return []
        
    scores = []
    
    # Iterate from the first possible gap (after k sentences)
    # to the last possible gap (before k sentences from the end)
    for i in range(k, len(embs_stack) - k):
        # Score gap at (i): compare [s(i-k)...s(i-1)] with [s(i+1)...s(i+k)].
        
        # Block before: k sentences ending at i
        block_before = embs_stack[i-k : i]
        
        # Block after: k sentences starting at i+1
        block_after = embs_stack[i+1 : i+k+1]
        
        # Pool the embeddings (average) to get a topic vector
        pool_before = torch.mean(block_before, dim=0)
        pool_after = torch.mean(block_after, dim=0)
        
        # Calculate cosine similarity
        sim = util.cos_sim(pool_before, pool_after).item()
        scores.append(sim)
        
    return scores

def depth_score(timeseries):
    """
    Calculates the "depth" of each valley (potential break).
    A deep valley means a significant topic change.
    (This function is a direct implementation from the paper's core.py)
    """
    depth_scores = []
    for i in range(1, len(timeseries) - 1):
        left, right = i - 1, i + 1
        while left > 0 and timeseries[left - 1] > timeseries[left]:
            left -= 1
        while (
            right < (len(timeseries) - 1) and timeseries[right + 1] > timeseries[right]
        ):
            right += 1
        
        # Score is the sum of the "cliffs" on either side
        depth = (timeseries[left] - timeseries[i]) + (timeseries[right] - timeseries[i])
        depth_scores.append(depth)
    return depth_scores

def smooth(timeseries, n=2, s=1):
    """
    Applies a simple moving average filter.
    (This function is from the paper's core.py)
    """
    if not timeseries:
        return []
    smoothed_timeseries = timeseries[:]
    for _ in range(n):
        for index in range(len(smoothed_timeseries)):
            neighbours = smoothed_timeseries[
                max(0, index - s) : min(len(timeseries) - 1, index + s) + 1
            ]
            smoothed_timeseries[index] = sum(neighbours) / len(neighbours)
    return smoothed_timeseries

def get_local_maxima(array):
    """
    Finds local maxima (peaks) in a list.
    (This function is from the paper's core.py)
    """
    local_maxima_indices = []
    local_maxima_values = []
    for i in range(1, len(array) - 1):
        if array[i - 1] < array[i] and array[i] > array[i + 1]:
            local_maxima_indices.append(i)
            local_maxima_values.append(array[i])
    return local_maxima_indices, local_maxima_values

# --- Main Segmentation Function ---

def find_topical_segments(
    sents, 
    k=5, 
    min_sentence_len=4,
    smooth_passes=1, 
    smooth_window=1, 
    threshold_factor=0.6
):
    """
    Combines all steps into the full hybrid pipeline.
    
    Args:
        text (str): The full podcast transcript.
        k (int): The block size for TextTiling.
        min_sentence_len (int): Filter out sentences shorter than this.
        smooth_passes (int): Number of smoothing passes.
        smooth_window (int): Window size for smoothing.
        threshold_factor (float): Threshold for break detection, as a
                                  factor of the max depth score.
    
    Returns:
        list[str]: The final list of topic segments.
        dict: A dictionary of intermediate scores for plotting.
    """
    
    # --- Step 1: Clean sentences with wtpsplit ---
    #print("\n--- 1. Cleaning text with wtpsplit ---")
    #sents = sat_model.split(text, do_paragraph_segmentation=False)
    #print(f"Found {len(sents)} raw sentences.")

    # --- Step 2: Pre-process and Filter (CRITICAL) ---
    # This is from the paper: filter out very short,
    # non-semantic "noise" sentences.
    original_sents = sents
    #sents = [s.strip() for s in sents if len(s.split()) > min_sentence_len]
    #print(f"Filtered down to {len(sents)} sentences (>{min_sentence_len} words).")
    
    if len(sents) < k * 2:
        print("Not enough sentences to perform segmentation.")
        return [" ".join(original_sents)], {}
    
    # --- Step 3: Get Embeddings ---
    print("\n--- 2. Calculating Embeddings ---")
    embs = embed_model.encode(
        sents, 
        convert_to_tensor=True, 
        show_progress_bar=True
    )

    # --- Step 4: Run TextTiling Algorithm ---
    print("\n--- 3. Running TextTiling Algorithm ---")
    
    # 4a. Get block comparison scores
    sim_scores = block_comparison_score(embs, k=k) # 'embs' is already a stack
    
    # 4b. Smooth the scores
    smooth_scores = smooth(sim_scores, n=smooth_passes, s=smooth_window)
    
    # 4c. Calculate "depth" of valleys
    depth_scores = depth_score(smooth_scores)
    
    print(f"Calculated {len(sim_scores)} block similarity scores.")
    print(f"Calculated {len(depth_scores)} depth scores.")

    # --- Step 5: Find Breakpoints (Thresholding) ---
    print("\n--- 4. Finding Breakpoints ---")
    
    # Find the peaks of the depth scores
    local_maxima_indices, local_maxima_values = get_local_maxima(depth_scores)
    
    if not local_maxima_values:
        print("No significant breaks found.")
        return [" ".join(original_sents)], {}

    # Adaptive threshold:
    # A break must be at least 60% as "deep" as the deepest break.
    threshold = threshold_factor * max(local_maxima_values)
    
    # Find all breaks that pass the threshold
    break_indices = [
        local_maxima_indices[i] 
        for i, val in enumerate(local_maxima_values) 
        if val > threshold
    ]
    
    print(f"Found {len(local_maxima_indices)} potential breaks.")
    print(f"Max depth score: {max(local_maxima_values):.4f}")
    print(f"Calculated threshold: {threshold:.4f}")
    print(f"Filtered down to {len(break_indices)} final breaks.")

    # --- Step 6: Map indices and create segments ---
    print("\n--- 5. Creating Final Segments ---")
    
    # This maps the depth_score indices back to the original sentence indices.
    # A depth_score index 'idx' corresponds to a sim_score index 'idx+1'.
    # A sim_score index 'idx' corresponds to a sentence gap 'idx+k'.
    final_break_indices = [k + idx + 1 for idx in break_indices]
    
    # Create the final list of "slice" points
    idx_list = sorted(list(set([0] + final_break_indices + [len(sents)])))
    
    segments = []
    for i in range(len(idx_list) - 1):
        start = idx_list[i]
        end = idx_list[i+1]
        segments.append(" ".join(sents[start:end]))
        
    # Data for plotting
    plot_data = {
        'sim_scores': sim_scores,
        'smooth_scores': smooth_scores,
        'depth_scores': depth_scores,
        'threshold': threshold,
        'breaks': break_indices,
        'score_x_axis': list(range(len(depth_scores))),
        'break_x_axis': break_indices
    }
        
    return segments, plot_data

def plot_segmentation_scores(plot_data, k=5):
    """
    Uses matplotlib to visualize the algorithm's scores.
    """
    if not plot_data:
        print("\nNo plot data to show.")
        return
        
    print("\nGenerating plot...")
    
    sim_scores = plot_data['sim_scores']
    smooth_scores = plot_data['smooth_scores']
    depth_scores = plot_data['depth_scores']
    threshold = plot_data['threshold']
    break_x_axis = plot_data['break_x_axis']

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    fig.suptitle('Hybrid TextTiling Segmentation Analysis', fontsize=16)

    # --- Plot 1: Similarity Scores ---
    # X-axis needs to be offset to align with gaps
    sim_x_axis = [i + k for i in range(len(sim_scores))]
    smooth_x_axis = [i + k for i in range(len(smooth_scores))]

    ax1.plot(sim_x_axis, sim_scores, 'b-', label='Raw Similarity')
    ax1.plot(smooth_x_axis, smooth_scores, 'r-', linewidth=2, label='Smoothed Similarity')
    ax1.set_ylabel('Similarity (Higher is similar)')
    ax1.set_title('Block Similarity Scores (Valleys = Topic Breaks)')
    ax1.legend()
    ax1.grid(True, linestyle=':')

    # --- Plot 2: Depth Scores ---
    # X-axis needs to be offset to align with gaps
    depth_x_axis = [i + k + 1 for i in range(len(depth_scores))]
    
    ax2.plot(depth_x_axis, depth_scores, 'g-', linewidth=2, label='Valley Depth Score')
    ax2.plot(
        depth_x_axis, 
        [threshold] * len(depth_scores), 
        'r--', 
        label=f'Threshold ({threshold:.2f})'
    )
    
    # Plot the breaks we found
    break_points_x = [depth_x_axis[i] for i in break_x_axis]
    break_points_y = [depth_scores[i] for i in break_x_axis]
    ax2.plot(break_points_x, break_points_y, 'kx', markersize=10, mew=3, label='Final Topic Break')
    
    ax2.set_xlabel('Sentence Gap Index')
    ax2.set_ylabel('Depth Score (Higher is better break)')
    ax2.set_title('Depth Score Analysis (Peaks = Topic Breaks)')
    ax2.legend()
    ax2.grid(True, linestyle=':')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    #plt.savefig("segmentation_plot.png")
    print("Plot saved to segmentation_plot.png")
    # To display the plot in a notebook, you might use plt.show()
    plt.show()

# # --- Main execution ---
# if __name__ == "__main__":

#     # --- A. Sample Data (Noisier) ---
#     # This text includes short interjections that our
#     # old method would fail on, but this method should handle.
#     sample_transcript = (
#         "Welcome to 'TechForward', I'm your host, Alex. "
#         "Today we're diving deep into the world of artificial intelligence. "
#         "It's a field that's moving at lightning speed. "
#         "We're seeing AI that can generate photorealistic video. "
#         "But what does this mean for artists? "
#         "Sarah, you're an expert in AI ethics, what's your take? "
#         "Well Alex, it's a valid concern, but I see it more as a tool for augmentation, not replacement. "
#         "Right. " # NOISE
#         "Think of it like the synthesizer in music. "
#         "It didn't replace musicians, it created new genres.\n"
#         "That's a great point. Okay. " # NOISE
#         "Let's switch gears a bit. "
#         "I want to talk about hardware. "
#         "None of this amazing AI would be possible without the incredible advancements in GPU technology. "
#         "These new tensor cores are specifically designed for matrix multiplication. "
#         "It's fascinating how software and hardware are co-evolving.\n"
#         "I see. " # NOISE
#         "Speaking of hardware, I've just been testing the new 'QuantumByte' laptop. "
#         "It's got one of these new chips. "
#         "The battery life, however, is another story. "
#         "That's always the trade-off, isn't it? "
#         "For our listeners thinking of upgrading, I'd say wait for the second-gen reviews.\n"
#         "Absolutely. Okay, before we wrap up, let's look at what's coming next week. "
#         "We'll be interviewing the CEO of 'GreenSolar', a company revolutionizing home energy storage. "
#         "It's a bold claim, and we're going to dig into the science behind it. Thanks for tuning in."
#     )
    
#     # --- B. Set Hyperparameters ---
#     # We use a smaller 'k' because our text is short.
#     # For a long podcast, the paper's default of k=15 is better.
#     K_BLOCK_SIZE = 4           # How many sentences in a block
#     MIN_SENTENCE_LEN = 3     # Filter sentences shorter than 3 words
#     SMOOTH_PASSES = 1        # Number of smoothing passes
#     SMOOTH_WINDOW = 1        # Window for smoothing
#     THRESHOLD_FACTOR = 0.5   # 0.6 = break must be 60% of max depth
    
#     # --- C. Run Full Pipeline ---
#     segments, plot_data = find_topical_segments(
#         sample_transcript,
#         k=K_BLOCK_SIZE,
#         min_sentence_len=MIN_SENTENCE_LEN,
#         smooth_passes=SMOOTH_PASSES,
#         smooth_window=SMOOTH_WINDOW,
#         threshold_factor=THRESHOLD_FACTOR
#     )

#     # --- D. Print Results ---
#     print("\n" + "="*30)
#     print("HYBRID TEXTTILING SEGMENTATION RESULTS")
#     print("="*30 + "\n")
    
#     for i, segment in enumerate(segments):
#         print(f"--- TOPICAL SEGMENT {i+1} ---\n")
#         print(segment.strip())
#         print("\n" + "-"*20 + "\n")

#     # --- E. Plot Results ---
#     if 'matplotlib' in globals():
#         plot_segmentation_scores(plot_data, k=K_BLOCK_SIZE)
#     else:
#         print("\nMatplotlib not installed. Skipping plot.")
#         print("To see the plot, run: pip install matplotlib")