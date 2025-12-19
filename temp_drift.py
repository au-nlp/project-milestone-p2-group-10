import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean

def analyze_temporal_drift(folder_path):
    print(f"--- Loading Data from {folder_path} ---")
    
    # 1. Load Data
    records = []
    files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    
    for filename in files:
        try:
            data = np.load(os.path.join(folder_path, filename), allow_pickle=True).item()
            
            # Extract basic fields
            vectors = data['vectors'] # (N segments, 768 dim)
            timestamp = data.get('timestamp')
            
            # Skip if no timestamp or vectors
            if timestamp is None or len(vectors) == 0:
                continue
                
            # Compute the "Episode Vector" (Mean of all segments in the episode)
            # This represents the episode's overall topic
            episode_vector = np.mean(vectors, axis=0)
            
            records.append({
                "date": pd.to_datetime(timestamp),
                "vector": episode_vector
            })
            
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue

    if not records:
        print("No valid data found.")
        return

    df = pd.DataFrame(records)
    
    # 2. Extract Year and Group
    df['year'] = df['date'].dt.year
    
    # Filter out weird dates (e.g., 1970 or future) if your data has noise
    df = df[(df['year'] > 2000) & (df['year'] <= 2025)]
    
    print(f"Loaded {len(df)} episodes across {df['year'].nunique()} years.")
    
    # 3. Calculate Annual Centroids
    # We find the "Average Episode" for each year
    yearly_centroids = df.groupby('year')['vector'].apply(lambda x: np.mean(np.vstack(x), axis=0)).sort_index()
    
    years = yearly_centroids.index.tolist()
    centroid_matrix = np.vstack(yearly_centroids.values) # Shape: (Num_Years, 768)

    # 4. Calculate Drift Distance (How far did we move from the previous year?)
    drift_distances = [0] # First year is 0 movement
    for i in range(1, len(centroid_matrix)):
        dist = euclidean(centroid_matrix[i-1], centroid_matrix[i])
        drift_distances.append(dist)
        
    # 5. PCA Projection (768D -> 2D) for Visualization
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(centroid_matrix)
    
    # --- VISUALIZATION ---
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # PLOT A: The Evolution Path
    # We plot the centroids as points and connect them with arrows
    ax1.plot(coords_2d[:, 0], coords_2d[:, 1], linestyle='--', color='gray', alpha=0.5)
    
    # Scatter plot with color gradient for time
    scatter = ax1.scatter(
        coords_2d[:, 0], 
        coords_2d[:, 1], 
        c=years, 
        cmap='viridis', 
        s=200, 
        edgecolor='black',
        zorder=10
    )
    
    # Add year labels and arrows
    for i, year in enumerate(years):
        # Label the dot
        ax1.text(
            coords_2d[i, 0] + 0.02, 
            coords_2d[i, 1] + 0.02, 
            str(year), 
            fontsize=11, 
            fontweight='bold'
        )
        
        # Draw arrow from previous year
        if i > 0:
            ax1.arrow(
                coords_2d[i-1, 0], coords_2d[i-1, 1],
                coords_2d[i, 0] - coords_2d[i-1, 0], 
                coords_2d[i, 1] - coords_2d[i-1, 1],
                color='black', alpha=0.3, width=0.002, head_width=0.02, length_includes_head=True
            )

    ax1.set_title("Evolution of Thought: Topic Drift Map (PCA)", fontsize=14)
    ax1.set_xlabel("PCA Component 1", fontsize=12)
    ax1.set_ylabel("PCA Component 2", fontsize=12)
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Year')

    # PLOT B: Magnitude of Shift
    # How much did the conversation change compared to the previous year?
    sns.barplot(x=years, y=drift_distances, palette="viridis", ax=ax2)
    ax2.set_title("Magnitude of Topic Shift (Year-over-Year)", fontsize=14)
    ax2.set_ylabel("Euclidean Shift Distance", fontsize=12)
    ax2.set_xlabel("Year", fontsize=12)

    plt.tight_layout()
    plt.show()