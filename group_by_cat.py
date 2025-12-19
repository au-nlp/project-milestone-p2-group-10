import os
import shutil
import pandas as pd
import re

# ---------------- CONFIGURATION ---------------- #
# Update this to point to your actual dataset file if needed
# Assuming dataset_sample is already loaded in your environment
# If not, uncomment the line below:
# dataset_sample = pd.read_csv("path/to/your/dataset.csv") 

def group_by_category(dataset_sample):

    SOURCE_FOLDER = "processed_vectors_sample2_500"
    HEALTH_FOLDER = "grouped_health"
    RELIGION_FOLDER = "grouped_religion"
    EDUCATION_FOLDER = "grouped_education"

    # target keywords
    TARGET_CATEGORIES = ["health", "religion", "education"]

    # ---------------- SETUP ---------------- #

    # 1. Create the destination folders if they don't exist
    os.makedirs(HEALTH_FOLDER, exist_ok=True)
    os.makedirs(RELIGION_FOLDER, exist_ok=True)
    os.makedirs(EDUCATION_FOLDER, exist_ok=True)

    # 2. Get list of all npy files in the source folder
    all_files = [f for f in os.listdir(SOURCE_FOLDER) if f.endswith('.npy')]

    print(f"Found {len(all_files)} files in {SOURCE_FOLDER}. Starting processing...")

    # ---------------- MAIN LOOP ---------------- #

    count_health = 0
    count_religion = 0
    count_education = 0

    for filename in all_files:
        # 3. Extract the episode number from the filename
        # Looks for digits between 'episode_' and '_vectors'
        match = re.search(r'episode_(\d+)_vectors\.npy', filename)
        
        if match:
            episode_idx = int(match.group(1))
            
            # 4. Check if this index exists in the dataframe
            # We use a try-except block to handle cases where the dataframe 
            # might be shorter than the episode number (inconsistency check)
            try:
                row = dataset_sample.iloc[episode_idx]
            except IndexError:
                print(f"Skipping {filename}: Index {episode_idx} out of bounds for dataframe.")
                continue

            # 5. Extract all categories for this row into a set for easy searching
            # We look at columns category1 through category10
            # We convert to lowercase to ensure case-insensitive matching
            row_categories = set()
            for i in range(1, 11):
                col_name = f"category{i}"
                if col_name in row and pd.notna(row[col_name]):
                    row_categories.add(str(row[col_name]).lower().strip())

            # 6. Check for keywords and copy files
            src_path = os.path.join(SOURCE_FOLDER, filename)

            if "health" in row_categories:
                dst_path = os.path.join(HEALTH_FOLDER, filename)
                shutil.copy2(src_path, dst_path)
                count_health += 1
                # print(f"Copied {filename} to {HEALTH_FOLDER}")

            if "religion" in row_categories:
                dst_path = os.path.join(RELIGION_FOLDER, filename)
                shutil.copy2(src_path, dst_path)
                count_religion += 1
                # print(f"Copied {filename} to {RELIGION_FOLDER}")

            if "education" in row_categories:
                dst_path = os.path.join(EDUCATION_FOLDER, filename)
                shutil.copy2(src_path, dst_path)
                count_education += 1
                # print(f"Copied {filename} to {RELIGION_FOLDER}")

    print("Processing complete.")
    print(f"Total files copied to Health: {count_health}")
    print(f"Total files copied to Religion: {count_religion}")
    print(f"Total files copied to Education: {count_education}")