import os
from pathlib import Path

# Base path to your phases folder
base_path = Path("/Users/jjcatulle/Desktop/ML-AI-learning")

# Find all 'completed' folders
for completed_folder in base_path.rglob("completed"):
    if completed_folder.is_dir():
        # Iterate through files in the completed folder
        for file in completed_folder.iterdir():
            if file.is_file() and file.name.startswith("STARTER_"):
                # Create new filename by replacing 'STARTER_' with 'MY_'
                new_name = file.name.replace("STARTER_", "MY_", 1)
                new_path = file.parent / new_name
                
                # Rename the file
                file.rename(new_path)
                print(f"Renamed: {file.name} → {new_name}")

print("Renaming complete!") 