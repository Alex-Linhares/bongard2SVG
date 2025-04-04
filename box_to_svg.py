import cv2
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
import os
import svgwrite
from percentage_inside_contour import ContourAnalyzer

SMALL_OBJECT_CONTOUR_AREA = 1 # why are we filtering out small objects?

"""commit message = Problems:  tiny objects are not being detected.  Solution:  increased the contour area threshold.
                    small objects are being detected as filled, but they are not actually filled.
                    Lines are not being detected as lines (area is too small?).
                    detected shapes have a fine outer stroke and a fine inner stroke.
                    
                    
                    TODO: Refactor extract method on lines 185 to 190 and 41-43 to use the new threshold.
                          Refactor again to get the percentage of white pixels inside the shape to determine if it's filled.
                    """


class BoxToSVGConverter:
    def __init__(self, input_folder="boxes"):
        """Initialize with input folder path."""
        self.input_folder = Path(input_folder)
        self.analyzer = ContourAnalyzer()
        
    def process_box(self, box_path):
        """Process a single box image and convert it to SVG."""
        try:
            # Read the image
            image = cv2.imread(box_path)
            if image is None:
                print(f"Error: Could not read image {box_path}")
                return

            # Get original image dimensions
            height, width = image.shape[:2]

            # Convert to grayscale if needed
            if len(image.shape) > 2:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Create binary image for contour detection
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            # Create SVG file path
            svg_path = os.path.splitext(box_path)[0] + '.svg'

            # Create SVG with original dimensions
            dwg = svgwrite.Drawing(svg_path, size=(width, height))

            # Add white background
            dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill='white'))

            # Process each contour
            for contour in contours:
                # Filter out small contours
                if cv2.contourArea(contour) < SMALL_OBJECT_CONTOUR_AREA:
                    continue

                # Use the ContourAnalyzer instead of the local method
                if self.analyzer.get_white_pixel_percentage_in_contour(box_path, contour, threshold=230) < 0.65:
                    continue

                # Create a mask and check if the shape is filled
                mask = np.zeros(binary.shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], 0, 255, -1)
                mean_val = cv2.mean(binary, mask=mask)[0]
                is_filled = mean_val < 127

                # Approximate the contour to detect shape type
                epsilon = 0.04 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Convert points to SVG format
                points = []
                for point in approx:
                    x, y = point[0]
                    points.append((x, y))

                # Determine shape type based on number of vertices
                num_vertices = len(points)
                
                # Check if it's a circle
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                if circularity > 0.8 and num_vertices > 4:
                    # Draw circle
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    dwg.add(dwg.circle(center=(x, y), r=radius,
                                     fill='black' if is_filled else 'none',
                                     stroke='black',
                                     stroke_width=5))
                else:
                    # Draw polygon
                    path_data = 'M ' + ' L '.join([f'{x},{y}' for x, y in points]) + ' Z'
                    dwg.add(dwg.path(d=path_data,
                                   fill='black' if is_filled else 'none',
                                   stroke='black',
                                   stroke_width=5))

            # Save the SVG file
            dwg.save()
            print(f"Created: {os.path.basename(svg_path)}")

        except Exception as e:
            print(f"Error processing {box_path}: {str(e)}")

    def process_all_boxes(self):
        """Process all box images in the input folder."""
        # Get all PNG files
        box_files = list(self.input_folder.glob("BP*.png"))

        if not box_files:
            print(f"No PNG files found in {self.input_folder}")
            return

        print(f"Found {len(box_files)} box images to process")

        # Sort files by name
        box_files.sort(key=lambda x: x.name)

        # Process each box
        for box_path in box_files:
            try:
                self.process_box(box_path)
            except Exception as e:
                print(f"Error processing {box_path.name}: {e}")
                continue
        print("\nProcessing complete!")


if __name__ == "__main__":
    # Create converter instance
    converter = BoxToSVGConverter()
    
    # Process all boxes
    converter.process_all_boxes() 