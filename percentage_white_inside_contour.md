# Contour Analysis Module

This module provides functionality for analyzing pixel distributions within contours in images, particularly useful for determining if shapes in binary images are filled or outlined.

## Features

- Calculate the percentage of white pixels within a contour
- Analyze all contours in an image
- Support for both grayscale and binary images
- Configurable threshold for what constitutes a "white" pixel

## Usage

### Basic Usage

```python
from percentage_inside_contour import ContourAnalyzer

# Create an analyzer instance
analyzer = ContourAnalyzer()

# Analyze a specific contour in an image
percentage = analyzer.get_white_pixel_percentage_in_contour(
    image_path="path/to/image.png",
    contour=some_contour,
    threshold=127
)

# Analyze all contours in an image
results = analyzer.analyze_image_contours(
    image_path="path/to/image.png",
    threshold=127
)
```

### Methods

#### get_white_pixel_percentage_in_contour

```python
def get_white_pixel_percentage_in_contour(image_path, contour, threshold=127)
```

Calculates the percentage of pixels within a contour that are considered "white" (above the threshold).

**Parameters:**
- `image_path` (str or Path): Path to the image file
- `contour` (numpy.ndarray): Contour from cv2.findContours
- `threshold` (int): Minimum pixel value to be considered "white" (0-255)

**Returns:**
- `float`: Percentage of white pixels (0.0 to 1.0)

#### analyze_image_contours

```python
def analyze_image_contours(image_path, threshold=127)
```

Analyzes all contours in an image and returns their white pixel percentages.

**Parameters:**
- `image_path` (str or Path): Path to the image file
- `threshold` (int): Minimum pixel value to be considered "white"

**Returns:**
- `list`: List of tuples (contour, white_percentage)

## Dependencies

- OpenCV (cv2)
- NumPy
- pathlib

## Installation

1. Ensure you have the required dependencies:
```bash
pip install opencv-python numpy
```

2. Copy the `percentage_inside_contour.py` file to your project.

## Example

```python
from percentage_inside_contour import ContourAnalyzer

# Create analyzer instance
analyzer = ContourAnalyzer()

# Analyze all contours in an image
results = analyzer.analyze_image_contours("image.png", threshold=200)

# Print results
for i, (contour, percentage) in enumerate(results):
    print(f"Contour {i + 1} white pixel percentage: {percentage:.2f}")
```

## Notes

- The module assumes input images are readable by OpenCV
- Contours should be in the format provided by cv2.findContours
- White pixel percentage is calculated based on the specified threshold
- Returns 0.0 if the contour is empty or an error occurs