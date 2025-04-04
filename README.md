# Box to SVG Converter

A Python module for converting Bongard Problem box images to SVG format, with accurate shape detection and representation.

## Features

- Converts PNG images to SVG format
- Detects and classifies different shapes:
  - Triangles
  - Squares
  - Rectangles
  - Circles
  - Polygons
- Preserves shape properties:
  - Fill (solid or outline)
  - Dimensions
  - Position
- Handles multiple files in batch processing
- Maintains original image dimensions
- Supports shape analysis for complex patterns

## Dependencies

- OpenCV (cv2)
- NumPy
- svgwrite
- pathlib
- xml.etree.ElementTree
- percentage_inside_contour (local module)

## Installation

1. Ensure you have the required dependencies:
```bash
pip install opencv-python numpy svgwrite
```

2. Clone this repository or copy the following files to your project:
   - `box_to_svg.py`
   - `percentage_inside_contour.py`

## Usage

### Basic Usage

```python
from box_to_svg import BoxToSVGConverter

# Create converter instance
converter = BoxToSVGConverter(input_folder="boxes")

# Process all boxes in the input folder
converter.process_all_boxes()
```

### Directory Structure
project/
├── boxes/ # Input folder containing PNG files
│ ├── BP1_L1.png
│ ├── BP1_L2.png
│ └── ...
├── box_to_svg.py
└── percentage_inside_contour.py


### Configuration

The module has several configurable parameters:

- `SMALL_OBJECT_CONTOUR_AREA`: Minimum area threshold for detecting shapes (default: 1000)
- Input folder path can be specified in the constructor
- Thresholds for shape detection and fill analysis can be adjusted in the code

## Methods

### BoxToSVGConverter

#### __init__(input_folder="boxes")
Initializes the converter with the specified input folder.

#### process_box(box_path)
Processes a single box image and converts it to SVG.

#### process_all_boxes()
Processes all PNG files in the input folder that match the pattern "BP*.png".

## Shape Detection

The module detects shapes based on the following criteria:

- Triangles: 3 vertices
- Squares: 4 vertices with aspect ratio between 0.95 and 1.05
- Rectangles: 4 vertices with other aspect ratios
- Circles: More than 8 vertices and high circularity
- Polygons: Other shapes

## Fill Detection

Shape fill detection uses two approaches:
1. Mean pixel value analysis within the contour
2. White pixel percentage calculation (using percentage_inside_contour module)

## Output

- SVG files are created with the same name as the input PNG files
- Each SVG contains:
  - White background
  - Detected shapes with appropriate fill and stroke
  - Original image dimensions

## Example

```python
# Example with custom input folder
converter = BoxToSVGConverter(input_folder="custom_boxes")
converter.process_all_boxes()
```

## Notes

- Input images should be clear, binary-like images
- Small shapes and noise are filtered based on area threshold
- Shape detection parameters can be tuned for specific use cases
- The module handles both filled and outlined shapes
- Error handling is implemented for robustness

## Contributing

Feel free to submit issues and enhancement requests!

## License

GPL 3.0