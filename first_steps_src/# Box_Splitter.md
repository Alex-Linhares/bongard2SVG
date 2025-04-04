# Box Splitter

A Python module for splitting Bongard Problem images into individual boxes. The original images are 36 boxes arranged in 3 sets of 12 boxes each, from high-res scans of the Bongard problems book. This module processes images containing a grid of 36 boxes (arranged in 3 sets of 12 boxes each) and extracts each box as a separate image.

## Features

- Automatic detection of box grids in images
- Intelligent box extraction with border removal
- Handles multiple BP (Bongard Problem) sets per image
- Automatic naming convention (BP{number}_[L/R]{position}.png)
- Visual debugging with box detection visualization
- Error handling and progress reporting

## Dependencies

- OpenCV (cv2)
- NumPy
- pathlib

## Installation

1. Install required dependencies:
```bash
pip install opencv-python numpy
```

2. Copy `split_boxes.py` to your project directory

## Usage

### Basic Usage

```python
from split_boxes import BoxSplitter

# Create splitter instance with default paths
splitter = BoxSplitter()

# Process all images
splitter.process_all_images()
```

### Custom Directories

```python
# Specify custom input and output directories
splitter = BoxSplitter(
    input_folder="path/to/input",
    output_folder="path/to/output"
)
splitter.process_all_images()
```

### Directory Structure

project/
├── data/
│ ├── png/ # Default input folder
│ │ ├── binary_Bp79-81.png
│ │ └── ...
│ └── boxes/ # Default output folder
│ ├── BP79_L1.png
│ ├── BP79_R1.png
│ └── ...
└── split_boxes.py


## Methods

### BoxSplitter

#### __init__(input_folder="data/png", output_folder="data/boxes")
Initializes the splitter with input and output folder paths.

#### split_image(image_path)
Splits a single image into individual box images.

#### process_all_images()
Processes all binary PNG files in the input folder.

## Image Processing Steps

1. **Box Detection**
   - Finds contours in the image
   - Identifies boxes of similar size
   - Arranges boxes in a grid pattern
   - Separates into left (L) and right (R) sides

2. **Content Extraction**
   - Removes box borders
   - Extracts inner content
   - Applies margin adjustment

3. **File Organization**
   - Names files according to BP number and position
   - Separates left and right examples
   - Maintains set organization

## Naming Convention

Output files follow the pattern: `BP{number}_{side}{position}.png`
- `number`: Bongard Problem number
- `side`: L (left) or R (right)
- `position`: Position number (1-6)

Example: `BP79_L1.png` = Bongard Problem 79, Left side, Position 1

## Visual Debugging

- Shows detected boxes with green rectangles
- Labels each box with its position (L1-L6, R1-R6)
- Displays the processed image for verification

## Error Handling

- Validates box detection (expects 36 boxes)
- Checks for valid content extraction
- Reports processing errors
- Continues processing on error

## Example Output

Found 10 images to process
Processing: binary_Bp79-81.png
Detecting boxes...
Processing set 1 (BP79)
Saved: BP79_L1.png
Saved: BP79_R1.png
...
Processing set 2 (BP80)
...
Processing complete!


## Notes

- Input images should be binary (black and white)
- Expects 36 boxes per image (3 sets of 12)
- Box detection parameters can be adjusted in code
- Visual debugging can be disabled if needed

## Contributing

Feel free to submit issues and enhancement requests!

## License

GPL 3.0