import cv2
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
import os

class BoxToSVGConverter:
    def __init__(self, input_folder="data/boxes"):
        """Initialize with input folder path."""
        self.input_folder = Path(input_folder)
        
    def detect_shapes(self, img):
        """Detect shapes in a binary image and return their properties."""
        # Convert to binary if not already
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        # Create a copy of the original image for outline detection
        outline_img = img.copy()
        
        # Find contours for filled shapes
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        filled_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find contours for outlines
        _, outline_binary = cv2.threshold(outline_img, 127, 255, cv2.THRESH_BINARY)
        outline_contours, _ = cv2.findContours(outline_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []
        
        # Process filled shapes
        for contour in filled_contours:
            # Skip very small contours
            if cv2.contourArea(contour) < 10:
                continue
                
            # Get shape properties
            shape_info = {
                'points': contour.reshape(-1, 2).tolist(),
                'area': cv2.contourArea(contour),
                'perimeter': cv2.arcLength(contour, True),
                'is_closed': cv2.arcLength(contour, True) > 0,
                'is_filled': True
            }
            
            # Try to fit different shapes
            approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
            
            if len(approx) == 3:
                shape_info['type'] = 'triangle'
            elif len(approx) == 4:
                shape_info['type'] = 'rectangle'
            elif len(approx) > 8:
                # Check if it's a circle
                (x, y), radius = cv2.minEnclosingCircle(contour)
                shape_info['type'] = 'circle'
                shape_info['center'] = (int(x), int(y))
                shape_info['radius'] = int(radius)
            else:
                shape_info['type'] = 'polygon'
                shape_info['vertices'] = len(approx)
            
            shapes.append(shape_info)
        
        # Process outline shapes
        for contour in outline_contours:
            # Skip very small contours
            if cv2.contourArea(contour) < 10:
                continue
                
            # Check if this contour is already processed as a filled shape
            # by comparing with filled contours
            is_duplicate = False
            for filled_contour in filled_contours:
                if cv2.matchShapes(contour, filled_contour, cv2.CONTOURS_MATCH_I1, 0.0) < 0.1:
                    is_duplicate = True
                    break
            
            if is_duplicate:
                continue
                
            # Get shape properties
            shape_info = {
                'points': contour.reshape(-1, 2).tolist(),
                'area': cv2.contourArea(contour),
                'perimeter': cv2.arcLength(contour, True),
                'is_closed': cv2.arcLength(contour, True) > 0,
                'is_filled': False
            }
            
            # Try to fit different shapes
            approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
            
            if len(approx) == 3:
                shape_info['type'] = 'triangle'
            elif len(approx) == 4:
                shape_info['type'] = 'rectangle'
            elif len(approx) > 8:
                # Check if it's a circle
                (x, y), radius = cv2.minEnclosingCircle(contour)
                shape_info['type'] = 'circle'
                shape_info['center'] = (int(x), int(y))
                shape_info['radius'] = int(radius)
            else:
                shape_info['type'] = 'polygon'
                shape_info['vertices'] = len(approx)
            
            shapes.append(shape_info)
            
        return shapes
        
    def create_svg(self, shapes, width, height, output_path):
        """Create an SVG file from the detected shapes."""
        # Create SVG root element
        svg = ET.Element('svg')
        svg.set('width', str(width))
        svg.set('height', str(height))
        svg.set('xmlns', 'http://www.w3.org/2000/svg')
        svg.set('viewBox', f'0 0 {width} {height}')
        
        # Add white background
        background = ET.SubElement(svg, 'rect')
        background.set('width', str(width))
        background.set('height', str(height))
        background.set('fill', 'white')
        
        # Add shapes
        for shape in shapes:
            if shape['type'] == 'circle':
                circle = ET.SubElement(svg, 'circle')
                circle.set('cx', str(shape['center'][0]))
                circle.set('cy', str(shape['center'][1]))
                circle.set('r', str(shape['radius']))
                
                if shape['is_filled']:
                    circle.set('fill', 'black')
                else:
                    circle.set('fill', 'none')
                    circle.set('stroke', 'black')
                    circle.set('stroke-width', '1')
            else:
                # Create path for other shapes
                path = ET.SubElement(svg, 'path')
                points = shape['points']
                d = f"M {points[0][0]},{points[0][1]}"
                for point in points[1:]:
                    d += f" L {point[0]},{point[1]}"
                if shape['is_closed']:
                    d += " Z"
                path.set('d', d)
                
                if shape['is_filled']:
                    path.set('fill', 'black')
                else:
                    path.set('fill', 'none')
                    path.set('stroke', 'black')
                    path.set('stroke-width', '1')
        
        # Create XML tree and save
        tree = ET.ElementTree(svg)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
                
    def process_box(self, image_path):
        """Process a single box image and create SVG file."""
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Error: Could not open {image_path}")
            return
            
        # Get image dimensions
        height, width = img.shape[:2]
        
        # Detect shapes
        shapes = self.detect_shapes(img)
        
        # Create output path
        base_path = image_path.with_suffix('')
        svg_path = base_path.with_suffix('.svg')
        
        # Create SVG
        self.create_svg(shapes, width, height, svg_path)
        
        print(f"Processed {image_path.name}")
        print(f"Created: {svg_path.name}")
        
    def process_all_boxes(self):
        """Process all box images in the input folder."""
        # Get all PNG files
        box_files = list(self.input_folder.glob("*.png"))
        
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

if __name__ == "__main__":
    # Create converter instance
    converter = BoxToSVGConverter()
    
    # Process all boxes
    converter.process_all_boxes() 