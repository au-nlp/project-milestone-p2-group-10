import os
import glob
import numpy as np
import pandas as pd


def merge_dateset(folder_name):

    # --- Run the harveser.py ---
    # it creates simantic segmentation and pre-compute the embeddings for topic modeling.
    # embeddings will be saved in the processe_vectorss folder. Single ".npy" file for each episode
    # the idea is to segment locally for each episode with varied lengths since hyperparameter K in our TexTilling algorithm is
    # very sensitive to length. 

    # --- Load the Harvested Vectors ---
    VECTOR_DIR = folder_name
    files = sorted(glob.glob(os.path.join(VECTOR_DIR, "*.npy")))

    all_texts = []       # saves all the segments from each episode for every episode
    all_embeddings = []  # saves all the eembeddings from each episode for every episode
    timestamps = []      # To track which episode each segment belongs to
    episode_names = []   # To track episode IDs

    episode_indices = [] # Integer timestamp (0, 1, 2...)
    episode_labels = []  # String label ("Episode 1", "Episode 2"...)

    print("Loading data...")
    # for fpath in files:
    #     # Load the .npy file
    #     data = np.load(fpath, allow_pickle=True).item()
        
    #     # Extract data
    #     vectors = data['vectors']          # The embeddings
    #     snippets = data['text_snippets']   # The text segments
    #     ep_id = data.get('episode_id', os.path.basename(fpath))
        
    #     # Append to our global lists
    #     # Note: We assume 'vectors' and 'snippets' are aligned lists/arrays
    #     for i, vec in enumerate(vectors):
    #         all_texts.append(snippets[i])
    #         all_embeddings.append(vec)
    #         episode_names.append(ep_id)
    #         timestamps.append(i) # Simple index as timestamp proxy


    for i, fpath in enumerate(files):
        data = np.load(fpath, allow_pickle=True).item()
        
        vectors = data['vectors']
        snippets = data['text_snippets']
        ep_id = data.get('episode_id', os.path.basename(fpath))
        
   
        # Extract the date string (e.g., "2020-01-04 04:00:00")
        date_str = data.get('timestamp') 
        
        # Handle missing dates (fallback to a default or skip)
        if date_str is None:
            # Option A: Skip this episode
            # continue 
            # Option B: Use a dummy date (Be careful, this messes up the plot)
            date_obj = pd.Timestamp("2020-01-01")
            print(f"episode {i} doesn't have any datetime info")
        else:
            # Convert string to Pandas Timestamp
            date_obj = pd.to_datetime(date_str)
        
        count = len(vectors)
        
        all_texts.extend(snippets)
        all_embeddings.extend(vectors)
        
        # Assign THIS episode's date to ALL its segments
        timestamps.extend([date_obj] * count)
        
        episode_labels.append(ep_id)

    # Convert embeddings to a single numpy array
    # BERTopic expects shape (n_samples, embedding_dim)
    X_embeddings = np.array(all_embeddings)

    print(f"Loaded {len(all_texts)} total segments.")
    print(f"Embedding Matrix Shape: {X_embeddings.shape}")

    return all_texts, X_embeddings, timestamps