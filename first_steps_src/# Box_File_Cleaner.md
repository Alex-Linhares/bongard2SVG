# Box File Cleaner

This utility script (`clean_boxes.py`) helps maintain consistency in the Bongard problem image dataset by cleaning up box image files that don't follow the standard naming convention.

## Purpose

The script scans through the `data/boxes` directory and removes any PNG files that don't match the expected naming pattern. This ensures that only properly named box images remain in the dataset.

## Naming Convention

Files must follow this pattern:
- Start with "BP" followed by numbers
- Followed by either "_L" or "_R" (indicating Left or Right)
- Followed by a single digit from 1-6
- End with ".png"

Examples of valid filenames:
- `BP123_L1.png`
- `BP456_R3.png`

Any files not matching this pattern will be deleted.

## Usage

To run the script:

```bash
python clean_boxes.py
```

## Output

The script will:
1. Count and display the total number of PNG files found
2. Print the names of files being deleted
3. Show a summary of how many files were deleted and how many remain

## Directory Structure

The script expects the following directory structure:
```
.
└── data/
    └── boxes/
        └── *.png
```

## Warning

This script permanently deletes files that don't match the naming convention. Make sure to backup your data before running if you're unsure about the file names.

