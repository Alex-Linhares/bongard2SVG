# Batch Image Preprocessor

A Python module for preprocessing images in batch, converting them to binary format with noise reduction. This module is particularly useful for preparing images for further processing or analysis.

## Features

- Batch processing of multiple image formats
- Noise reduction using Gaussian blur
- Adaptive thresholding for handling varying lighting conditions
- Automatic output directory creation
- Support for multiple image formats (BMP, PNG, JPG, JPEG, TIFF)
- Progress tracking and reporting

## Dependencies

- OpenCV (cv2)
- NumPy
- pathlib

## Installation

1. Install required dependencies:
```bash
pip install opencv-python numpy
```

2. Copy `preprocess_images.py` to your project directory

## Usage

### Basic Usage

```python
from preprocess_images import BatchImagePreprocessor

# Create preprocessor instance with default input folder "data"
preprocessor = BatchImagePreprocessor()

# Process all images
preprocessor.process_all_images()
```

### Custom Input Directory

```python
# Specify custom input directory
preprocessor = BatchImagePreprocessor(input_folder="path/to/images")
preprocessor.process_all_images()
```

### Directory Structure
project/
├── data/ # Default input folder
│ ├── image1.png
│ ├── image2.jpg
│ └── binary/ # Output folder (created automatically)
│ ├── binary_image1.png
│ └── binary_image2.png
└── preprocess_images.py


## Methods

### BatchImagePreprocessor

#### __init__(input_folder="data")
Initializes the preprocessor with the specified input folder.

**Parameters:**
- `input_folder` (str): Path to the folder containing images

#### process_image(image_path)
Processes a single image: removes noise and converts to binary.

**Parameters:**
- `image_path` (Path): Path to the input image

**Returns:**
- `bool`: True if processing was successful, False otherwise

#### process_all_images()
Processes all supported images in the input folder.

## Image Processing Steps

1. **Image Loading**
   - Reads image in grayscale format
   - Checks for successful image loading

2. **Noise Reduction**
   - Applies Gaussian blur with 5x5 kernel
   - Reduces image noise while preserving edges

3. **Binary Conversion**
   - Uses adaptive thresholding (Gaussian)
   - Block size: 11
   - Constant subtraction: 2
   - Handles varying lighting conditions

4. **Output**
   - Creates binary/ subdirectory in input folder
   - Saves processed images with "binary_" prefix
   - Preserves original file extension

## Supported Image Formats

- PNG (.png, .PNG)
- JPEG (.jpg, .jpeg, .JPG, .JPEG)
- BMP (.bmp, .BMP)
- TIFF (.tiff, .TIFF)

## Example Output
Found 10 images to process
Processed: image1.png -> binary_image1.png
Processed: image2.jpg -> binary_image2.png
...
Processing complete!
Successfully processed: 10/10 images
Binary images saved in: data/binary


## Error Handling

- Checks for existence of input folder
- Verifies successful image loading
- Reports processing success/failure for each image
- Provides summary of processing results

## Notes

- Output images are saved in binary format
- Original images are preserved
- Processing parameters can be adjusted in the code
- Adaptive thresholding helps handle varying image conditions

## Contributing

Feel free to submit issues and enhancement requests!

## License

GPL 3.0