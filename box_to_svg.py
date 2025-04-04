import cv2
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
import os
import svgwrite

SMALL_OBJECT_CONTOUR_AREA = 1000

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

def get_white_pixel_percentage(self, image, contour, threshold=230):
    """
    Calculates the percentage of pixels within a contour that are "almost white".

    Args:
        image (numpy.ndarray): The grayscale or binary image.
        contour (numpy.ndarray): The contour.
        threshold (int, optional): The minimum pixel value to be considered "almost white" (0-255).
                                 Defaults to 240.

    Returns:
        float: The percentage of "almost white" pixels (0.0 to 1.0) within the contour.
    """

    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, -1)
    white_pixels = np.sum(image[mask == 255] >= threshold)
    total_pixels = np.sum(mask == 255)
    return float(white_pixels) / total_pixels if total_pixels > 0 else 0.0


    def detect_shapes(self, image):
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Create binary image
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if hierarchy is None:
            return []
        
        shapes = []
        for i, contour in enumerate(contours):
            # Skip very small contours.  Why?
            if cv2.contourArea(contour) < SMALL_OBJECT_CONTOUR_AREA:
                continue
            
            # Get shape properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Approximate the contour to reduce noise
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Determine shape type based on number of vertices
            vertices = len(approx)
            if vertices == 3:
                shape_type = 'triangle'
            elif vertices == 4:
                # Check if it's a rectangle
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w)/h
                if 0.95 <= aspect_ratio <= 1.05:
                    shape_type = 'square'
                else:
                    shape_type = 'rectangle'
            elif vertices > 8:
                # Check if it's a circle
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                # Compare the contour area with the circle area
                circle_area = np.pi * radius * radius
                if abs(area - circle_area) / circle_area < 0.1:
                    shape_type = 'circle'
                else:
                    shape_type = 'polygon'
            else:
                shape_type = 'polygon'

            # Check if the shape is filled by examining the pixel values inside the contour
            # This is a hack to get the fill color.  It's not perfect, and it's failing when there are small things inside the shape.
            mask = np.zeros(binary.shape, dtype=np.uint8) # Creates a completely black image (mask) the same size as binary
            cv2.drawContours(mask, [contour], 0, 255, -1) # Draws the contour on the mask
            mean_val = cv2.mean(binary, mask=mask)[0]
            is_filled = mean_val > 127  # If mean value is less than 127, the shape is filled (black)
            # playing with the threshold to see if it works better
            # is_filled = mean_val < 375    

            '''prompt: on lines 71 to 74 you try to see if the shape is filled, but this fails if there's a small shaape inside it, can you see the entirety of the shape to determine if it's filled?'''

            # Get the points for the shape
            if shape_type == 'circle':
                points = [(center[0], center[1], radius)]
            else:
                points = approx.reshape(-1, 2).tolist()

            shapes.append({
                'type': shape_type,
                'points': points,
                'area': area,
                'perimeter': perimeter,
                'is_filled': is_filled
            })

        return shapes
        
    def create_svg(self, shapes, width, height):
        """Create an SVG file from detected shapes."""
        # Create SVG root element
        svg = ET.Element('svg')
        svg.set('width', str(width))
        svg.set('height', str(height))
        svg.set('xmlns', 'http://www.w3.org/2000/svg')
        
        # Add white background
        background = ET.SubElement(svg, 'rect')
        background.set('width', str(width))
        background.set('height', str(height))
        background.set('fill', 'white')
        
        # Add shapes
        for shape in shapes:
            if shape['type'] == 'circle':
                circle = ET.SubElement(svg, 'circle')
                circle.set('cx', str(shape['points'][0][0]))
                circle.set('cy', str(shape['points'][0][1]))
                circle.set('r', str(shape['points'][0][2]))
                circle.set('stroke', 'black')
                circle.set('stroke-width', '1')
                if shape['is_filled']:
                    circle.set('fill', 'black')
                else:
                    circle.set('fill', 'none')
                
            else:
                path = ET.SubElement(svg, 'path')
                points = shape['points']
                d = f"M {points[0][0]},{points[0][1]}"
                for point in points[1:]:
                    d += f" L {point[0]},{point[1]}"
                if shape['is_filled']:
                    d += " Z"
                path.set('d', d)
                path.set('stroke', 'black')
                path.set('stroke-width', '1')
                if shape['is_filled']:
                    path.set('fill', 'black')
                else:
                    path.set('fill', 'none')
        
        return svg
        
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

            # Find contours - use RETR_LIST to get all contours at same hierarchy level
            contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            # Create SVG file path
            svg_path = os.path.splitext(box_path)[0] + '.svg'

            # Create SVG with original dimensions
            dwg = svgwrite.Drawing(svg_path, size=(width, height))

            # Add white background
            dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill='white'))

            # Process each contour
            for contour in contours:
                # Skip very small contours.  Why?
                if cv2.contourArea(contour) < SMALL_OBJECT_CONTOUR_AREA and self.get_white_pixel_percentage(binary, contour) < 0.5:
                   continue

                # Create a mask and check if the shape is filled
                mask = np.zeros(binary.shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], 0, 255, -1)
                mean_val = cv2.mean(binary, mask=mask)[0] # this only considers the pixels where the mask is non-zero (i.e., inside the white filled shape drawn in the previous step).
                is_filled = mean_val < 175  # If mean value is less than 127, the shape is filled (black)
                # playing with the threshold to see if it works better.  Note that 375 makes the entire box black.
                # is_filled = mean_val < 175 was the original threshold.
                if is_filled:
                    print(f"Filled shape detected with mean value {mean_val}, with white pixel percentage {self.get_white_pixel_percentage(binary, contour)}")

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
                
                if num_vertices == 3:
                    shape_type = 'triangle'
                elif num_vertices == 4:
                    # Check if it's a square or rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w)/h
                    shape_type = 'square' if 0.95 <= aspect_ratio <= 1.05 else 'rectangle'
                else:
                    # Check if it's a circle
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    shape_type = 'circle' if circularity > 0.8 else 'polygon'

                # Create SVG element based on shape type
                if shape_type == 'circle':
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    dwg.add(dwg.circle(center=(x, y), r=radius,
                                     fill='black' if is_filled else 'none',
                                     stroke='black',
                                     stroke_width=1))
                else:
                    # Create path for other shapes
                    path_data = 'M ' + ' L '.join([f'{x},{y}' for x, y in points]) + ' Z'
                    dwg.add(dwg.path(d=path_data,
                                   fill='black' if is_filled else 'none',
                                   stroke='black',
                                   stroke_width=1))

            # Save the SVG file
            dwg.save()
            print(f"Created: {os.path.basename(svg_path)}")

        except Exception as e:
            print(f"Error processing {box_path}: {str(e)}")

    """    
    def process_all_boxes(self):
        # Process all box images in the input folder.
        # Get all PNG files
        box_files = list(self.input_folder.glob("BP*.png"))
        
        if not box_files:
            print(f"No PNG files found in {self.input_folder}")
            return
            
        print(f"Found {len(box_files)} box images to process")
        
        # Process each box
        for box_path in box_files:
            try:
                self.process_box(box_path)
            except Exception as e:
                print(f"Error processing {box_path.name}: {e}")
                continue
                
        print("\nProcessing complete!")
    """


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