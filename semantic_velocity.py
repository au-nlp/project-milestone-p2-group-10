import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import seaborn as sns

# 1. (Existing Logic) - Calculate the Data
def analyze_semantic_velocity(folder_path):
    results = []
    files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    print(f"Analyzing pacing for {len(files)} episodes...")

    for filename in files:
        file_path = os.path.join(folder_path, filename)
        try:
            # Load the data dictionary
            data = np.load(file_path, allow_pickle=True).item()
            vectors = data['vectors']
            episode_id = data['episode_id']

            # We need at least 2 segments to calculate movement
            if len(vectors) < 2:
                continue
            
            # 1. Calculate distances between sequential segments
            # (Distance between Seg 0 & Seg 1, Seg 1 & Seg 2, etc.)
            distances = []
            for i in range(len(vectors) - 1):
                dist = euclidean(vectors[i], vectors[i+1])
                distances.append(dist)
            
            results.append({
                "episode_id": episode_id,
                "avg_velocity": np.mean(distances),      # Speed
                "peak_velocity": np.max(distances),      # Max Jump
                "pacing_variance": np.var(distances),    # Steadiness
                "segment_count": len(vectors)            # Length/Depth
            })
            
        except Exception:
            continue

    return pd.DataFrame(results)

# 2. (NEW) - Visualization Function
def plot_pacing_analysis(df):
    """
    Generates a 2-panel report visualization:
    1. An 'Archetype Map' (Scatter) classifying episodes by speed and stability.
    2. A Distribution plot showing the most common pacing for this category.
    """
    # Set a clean style for the report
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # --- PLOT A: The Archetype Map (Velocity vs. Variance) ---
    # X = How fast they move (Velocity)
    # Y = How consistent they are (Variance)
    # Size = Number of Segments (Duration/Depth)
    sns.scatterplot(
        data=df, 
        x="avg_velocity", 
        y="pacing_variance", 
        size="segment_count", 
        sizes=(50, 400), 
        alpha=0.7, 
        hue="avg_velocity", 
        palette="viridis", 
        ax=ax1,
        legend="brief"
    )

    # Annotate the extreme outliers (Fastest & Slowest)
    # We find the top 3 and bottom 3 to label them on the chart
    top_fast = df.nlargest(3, 'avg_velocity')
    top_slow = df.nsmallest(3, 'avg_velocity')
    outliers = pd.concat([top_fast, top_slow])

    for _, row in outliers.iterrows():
        ax1.text(
            row['avg_velocity'], 
            row['pacing_variance'], 
            f" {row['episode_id']}", 
            fontsize=9, 
            weight='bold', 
            color='black'
        )

    ax1.set_title("Episode Archetype Map: Speed vs. Stability", fontsize=14)
    ax1.set_xlabel("Semantic Velocity (Information Density)", fontsize=12)
    ax1.set_ylabel("Pacing Variance (Conversational Drift)", fontsize=12)
    
    # Add quadrant labels for interpretation
    # (Positions are approximate based on your data range, adjust 1.0/0.015 as needed)
    ax1.text(df['avg_velocity'].max(), 0, "Fast & Steady\n(Scripted/News)", 
             ha='right', va='bottom', color='green', alpha=0.6, fontsize=10)
    ax1.text(df['avg_velocity'].min(), 0, "Slow & Steady\n(Deep Dive)", 
             ha='left', va='bottom', color='blue', alpha=0.6, fontsize=10)

    # --- PLOT B: Pacing Distribution ---
    sns.histplot(df['avg_velocity'], kde=True, bins=15, color="teal", ax=ax2)
    ax2.axvline(df['avg_velocity'].mean(), color='red', linestyle='--', label=f"Category Mean: {df['avg_velocity'].mean():.2f}")
    
    ax2.set_title("Distribution of Information Density", fontsize=14)
    ax2.set_xlabel("Average Velocity", fontsize=12)
    ax2.legend()

    plt.tight_layout()
    plt.show()