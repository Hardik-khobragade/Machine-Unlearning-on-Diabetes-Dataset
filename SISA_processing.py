import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# =======================================================
# CELL 1: DATA PREPROCESSOR
# =======================================================
def _detect_user_id_column(dataframe: pd.DataFrame) -> str:
    """Helper function to find a user ID column."""
    possible_names = ['user_id', 'userid', 'user', 'uid', 'id']
    lower_cols = {col.lower(): col for col in dataframe.columns}
    for name in possible_names:
        if name in lower_cols:
            return lower_cols[name]
    # Raise error only if no column is found, to be caught below
    raise ValueError("No user identifier column found.")

def prepare_dataset(filepath):
    #Loads, cleans, and prepares the dataset.
    print(" Running data preprocessing...")
    df = pd.read_csv(filepath)
    print(" Dataset loaded successfully")

    if df.isnull().values.any():
        df = df.fillna(df.mean(numeric_only=True))
    else:
        print("No missing values found")

    non_numeric_cols = df.select_dtypes(exclude=['number']).columns
    if len(non_numeric_cols) > 0:
        encoder = LabelEncoder()
        for col in non_numeric_cols:
            df[col] = encoder.fit_transform(df[col].astype(str))

    # Try to find a user_id column. If not found, create one.
    try:
        user_id_col = _detect_user_id_column(df)
        print(f" Existing user ID column ('{user_id_col}') found")
    except ValueError:
        print(" No user ID column found. Creating a new 'user_id' column")
        df["user_id"] = np.arange(1, len(df) + 1)

    df["index"] = df.index
    print("  - Preprocessing complete")
    return df

# =======================================================
# CELL 2: SISA LOGIC
# =======================================================
def create_splits_and_mapping(dataframe, num_shards, splits_per_shard):
    """Creates shards, splits, and a DETAILED mapping of users to their data locations."""
    print("\n Running SISA splitting and mapping")
    user_id_col = _detect_user_id_column(dataframe)
    shuffled_df = dataframe.sample(frac=1, random_state=42).reset_index(drop=True)
    shards = np.array_split(shuffled_df, num_shards)

    all_splits = []
    user_mapping = {}

    for shard_idx, shard_df in enumerate(shards):
        if shard_df.empty: continue
        splits = np.array_split(shard_df, splits_per_shard)

        for split_idx, split_df in enumerate(splits):
            if split_df.empty: continue
            all_splits.append(split_df)

            for _, row in split_df.iterrows():
                user_id = int(row[user_id_col])
                original_index = int(row['index'])

                if user_id not in user_mapping:
                    user_mapping[user_id] = {
                        'original_rows': [],
                        'locations': {}
                    }

                user_mapping[user_id]['original_rows'].append(original_index)

                location_key = (shard_idx, split_idx)
                if location_key not in user_mapping[user_id]['locations']:
                    user_mapping[user_id]['locations'][location_key] = {
                        'shard': shard_idx,
                        'split': split_idx,
                        'rows': []
                    }

                user_mapping[user_id]['locations'][location_key]['rows'].append(original_index)

    #locations dict to the required list format
    for user_id in user_mapping:
        user_mapping[user_id]['locations'] = list(user_mapping[user_id]['locations'].values())
        # Also sort the original_rows list for consistency
        user_mapping[user_id]['original_rows'].sort()

    print("SISA logic with detailed mapping completed successfully.")
    return all_splits, user_mapping

# =======================================================
# CELL 3 & 4: MAIN EXECUTION AND VALIDATION
# =======================================================
def main():
    """Main function to run the entire pipeline."""
    # --- Configuration
    INPUT_DATASET = 'diabetes_with_users_reordered.csv'
    OUTPUT_DIR = 'sisa_data/'
    NUM_SHARDS = 4
    SPLITS_PER_SHARD = 3

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f" Directory '{OUTPUT_DIR}' is ready.")

    prepared_df = prepare_dataset(INPUT_DATASET)
    all_splits, user_map = create_splits_and_mapping(prepared_df, NUM_SHARDS, SPLITS_PER_SHARD)

    print("\n Saving all output files...")
    split_counter = 0
    for shard_idx in range(NUM_SHARDS):
        for split_idx in range(SPLITS_PER_SHARD):
            if split_counter < len(all_splits):
                split_df = all_splits[split_counter]
                file_name = f'shard_{shard_idx}_split_{split_idx}.csv'
                file_path = os.path.join(OUTPUT_DIR, file_name)
                split_df.to_csv(file_path, index=False)
                split_counter += 1

    mapping_file_path = os.path.join(OUTPUT_DIR, 'user_mapping.json')
    with open(mapping_file_path, 'w') as f:
        json.dump(user_map, f, indent=4)

    print("\n" + "="*50)
    print(" TEAM 1 EXECUTION COMPLETE ")
    print("="*50)
    print(f"  {split_counter} split files have been saved in the '{OUTPUT_DIR}' folder.")
    print(f"  User mapping for {len(user_map)} users saved to '{mapping_file_path}'.")
    print("="*50)


if __name__ == "__main__":
  main()


# From Team 1 !!!!
