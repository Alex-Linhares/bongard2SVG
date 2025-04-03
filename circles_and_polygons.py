import cv2
import numpy as np
from xml.etree import ElementTree as ET

class ImagePreprocessor:
    def __init__(self):
        self.binary_image = None
        
    def load_and_binarize(self, image_path):
        """
        Loads a binary PNG image.
        
        Args:
            image_path (str): Path to the PNG file
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Load the image - since it's already binary, we just need to load it
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Could not open or find the image at {image_path}")
            return False
            
        self.binary_image = img
        self.height, self.width = img.shape
        
        # Optional: Show the loaded image for debugging
        cv2.imshow("Loaded Binary Image", self.binary_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return True

def create_svg(shapes, width, height, output_path):
    """Creates an SVG file from the detected shapes."""
    # Create the SVG root element
    svg = ET.Element('svg')
    svg.set('width', str(width))
    svg.set('height', str(height))
    svg.set('xmlns', 'http://www.w3.org/2000/svg')
    
    # Add white background
    background = ET.SubElement(svg, 'rect')
    background.set('width', str(width))
    background.set('height', str(height))
    background.set('fill', 'white')
    
    # Add shapes to SVG - all black filled
    for shape in shapes:
        if shape['type'] == 'contour':
            # Create path for complex shapes
            path = ET.SubElement(svg, 'path')
            # Convert contour points to SVG path data
            d = "M " + " L ".join([f"{pt[0]},{pt[1]}" for pt in shape['points']]) + " Z"
            path.set('d', d)
            path.set('fill', 'black')
    
    # Create XML tree and save to file
    tree = ET.ElementTree(svg)
    with open(output_path, 'wb') as f:
        tree.write(f, encoding='utf-8', xml_declaration=True)

def detect_shapes(image_path):
    """Detects shapes in a binary image."""
    print(f"\nAnalyzing image: {image_path}")
    print("-" * 50)
    
    # Create preprocessor instance
    preprocessor = ImagePreprocessor()
    if not preprocessor.load_and_binarize(image_path):
        return

    # Use the binary image for shape detection
    img = cv2.cvtColor(preprocessor.binary_image.copy(), cv2.COLOR_GRAY2BGR)
    binary_img = preprocessor.binary_image

    # List to store all shapes for SVG creation
    shapes = []

    # Detect shapes using contours
    _, binary_img = cv2.threshold(binary_img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    print("\nShapes detected:")
    for i, contour in enumerate(contours, 1):
        area = cv2.contourArea(contour)
        if area < 50:  # Filter out very small contours
            continue
            
        # Draw the contour
        cv2.drawContours(img, [contour], 0, (0, 0, 0), 2)
        
        # Get shape information
        x, y, w, h = cv2.boundingRect(contour)
        print(f"Shape {i}: Area={area:.2f}, Position=({x}, {y}), Size={w}x{h}")
        
        # Store contour points for SVG
        shapes.append({
            'type': 'contour',
            'points': [point[0] for point in contour]
        })

    # Create SVG file
    output_svg = image_path.replace('.png', '.svg')
    create_svg(shapes, preprocessor.width, preprocessor.height, output_svg)
    print(f"\nSVG file created: {output_svg}")

    # Show results
    cv2.imshow("Detected Shapes", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage:
if __name__ == "__main__":
    image_path = "data/png/binary_Bp79-81.png"
    detect_shapes(image_path)