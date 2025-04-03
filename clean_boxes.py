import os
import re
from pathlib import Path

def clean_box_files():
    # Path to the boxes directory
    boxes_dir = Path("data/boxes")
    
    # Valid filename pattern: BP followed by numbers, then _L or _R, then a number, then .png
    pattern = re.compile(r"BP\d+_[LR][1-6]\.png$")
    
    # List all files in the directory
    files = list(boxes_dir.glob("*.png"))
    
    print(f"Found {len(files)} PNG files in {boxes_dir}")
    print("Deleting files with old naming convention...")
    
    # Delete files that don't match the pattern
    deleted_count = 0
    for file in files:
        if not pattern.match(file.name):
            print(f"Deleting: {file.name}")
            file.unlink()
            deleted_count += 1
    
    print(f"\nDeleted {deleted_count} files")
    print(f"Remaining files: {len(files) - deleted_count}")

if __name__ == "__main__":
    clean_box_files() 