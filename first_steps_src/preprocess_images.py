import cv2
import numpy as np
import os
from pathlib import Path

class BatchImagePreprocessor:
    def __init__(self, input_folder="data"):
        """
        Initialize the preprocessor with input folder path.
        
        Args:
            input_folder (str): Path to the folder containing images
        """
        self.input_folder = Path(input_folder)
        self.output_folder = self.input_folder / "binary"
        
        # Create output folder if it doesn't exist
        self.output_folder.mkdir(exist_ok=True)
        
    def process_image(self, image_path):
        """
        Process a single image: remove noise and convert to binary.
        
        Args:
            image_path (Path): Path to the input image
        """
        # Read image in grayscale
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Could not open {image_path}")
            return False
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Apply adaptive thresholding for better results with varying lighting
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,  # Block size
            2    # Constant subtracted from mean
        )
        
        # Create output filename
        output_path = self.output_folder / f"binary_{image_path.name}"
        
        # Save the binary image
        cv2.imwrite(str(output_path), binary)
        print(f"Processed: {image_path.name} -> {output_path.name}")
        return True
        
    def process_all_images(self):
        """Process all images in the input folder."""
        # Supported image extensions
        image_extensions = ('.bmp', '.png', '.jpg', '.jpeg', '.tiff', '.BMP', '.PNG', '.JPG', '.JPEG', '.TIFF')
        
        # Get all image files
        image_files = [f for f in self.input_folder.iterdir() 
                      if f.is_file() and f.suffix in image_extensions]
        
        if not image_files:
            print(f"No images found in {self.input_folder}")
            return
        
        print(f"Found {len(image_files)} images to process")
        
        # Process each image
        successful = 0
        for image_path in image_files:
            if self.process_image(image_path):
                successful += 1
                
        print(f"\nProcessing complete!")
        print(f"Successfully processed: {successful}/{len(image_files)} images")
        print(f"Binary images saved in: {self.output_folder}")

if __name__ == "__main__":
    # Create preprocessor instance
    preprocessor = BatchImagePreprocessor()
    
    # Process all images
    preprocessor.process_all_images() 