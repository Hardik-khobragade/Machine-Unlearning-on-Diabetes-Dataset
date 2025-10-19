# =======================================================
# TEAM 2 - SISA UNLEARNING IMPLEMENTATION
# =======================================================
import os
import json
import pandas as pd

# CONFIGURATION
SISA_DIR = "sisa_data"
USER_MAPPING_FILE = os.path.join(SISA_DIR, "user_mapping.json")
os.path.join(SISA_DIR, "user_mapping.json")
UNLEARNED_DIR = os.path.join(SISA_DIR, "unlearned_splits/")
os.makedirs(UNLEARNED_DIR, exist_ok=True)

def load_user_mapping():
    with open(USER_MAPPING_FILE, "r") as f:
        return json.load(f)

def remove_user_data(user_id, mapping):
    """
    Removes all rows belonging to a specific user_id
    and regenerates the affected split files.
    """
    if str(user_id) not in mapping:
        print(f"[ERROR] User ID {user_id} not found in mapping.")
        return
    
    user_info = mapping[str(user_id)]
    print(f"[INFO] Removing user {user_id} from affected splits...")

    for loc in user_info["locations"]:
        shard, split = loc["shard"], loc["split"]
        file_name = f"shard_{shard}_split_{split}.csv"
        file_path = os.path.join(SISA_DIR, file_name)
        
        if not os.path.exists(file_path):
            print(f"  - File not found: {file_name}")
            continue
        
        df = pd.read_csv(file_path)
        before = len(df)
        df = df[df["user_id"] != int(user_id)]
        after = len(df)
        
        new_path = os.path.join(UNLEARNED_DIR, file_name)
        df.to_csv(new_path, index=False)
        print(f"  - {file_name}: removed {before - after} rows, saved new file.")

    print(f"[SUCCESS] Unlearning for user {user_id} completed.\n")

def main():
    mapping = load_user_mapping()
    user_to_remove = int(input("Enter User ID to unlearn: "))
    remove_user_data(user_to_remove, mapping)

if __name__ == "__main__":
    main()
