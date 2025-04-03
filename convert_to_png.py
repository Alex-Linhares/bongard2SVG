import cv2
import os
from pathlib import Path

class ImageConverter:
    def __init__(self, input_folder="data/binary", output_folder="data/png"):
        """
        Initialize the converter with input and output folder paths.
        
        Args:
            input_folder (str): Path to the folder containing binary images
            output_folder (str): Path to save PNG images
        """
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        
        # Create output folder if it doesn't exist
        self.output_folder.mkdir(exist_ok=True, parents=True)
        
    def convert_image(self, image_path):
        """
        Convert a single image to PNG format.
        
        Args:
            image_path (Path): Path to the input image
        """
        # Read image
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Error: Could not open {image_path}")
            return False
        
        # Create output filename - replace original extension with .png
        output_filename = image_path.stem + ".png"
        output_path = self.output_folder / output_filename
        
        # Save as PNG
        success = cv2.imwrite(str(output_path), img)
        if success:
            print(f"Converted: {image_path.name} -> {output_path.name}")
            return True
        else:
            print(f"Error saving: {output_path}")
            return False

    def convert_all_images(self):
        """Convert all images in the input folder to PNG format."""
        # Get all image files
        image_files = [f for f in self.input_folder.iterdir() 
                      if f.is_file() and f.suffix.lower() in ('.bmp', '.jpg', '.jpeg', '.tiff')]
        
        if not image_files:
            print(f"No images found in {self.input_folder}")
            return
        
        print(f"Found {len(image_files)} images to convert")
        
        # Convert each image
        successful = 0
        for image_path in image_files:
            if self.convert_image(image_path):
                successful += 1
                
        print(f"\nConversion complete!")
        print(f"Successfully converted: {successful}/{len(image_files)} images")
        print(f"PNG images saved in: {self.output_folder}")

if __name__ == "__main__":
    # Create converter instance
    converter = ImageConverter()
    
    # Convert all images
    converter.convert_all_images() 