import cv2
import numpy as np
import os
from pathlib import Path

class BoxSplitter:
    def __init__(self, input_folder="data/png", output_folder="data/boxes"):
        """Initialize with input and output folder paths."""
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True, parents=True)
        
    def extract_number(self, filename):
        """Extract the BP number from filename like 'binary_Bp79-81.png'."""
        num_str = filename.replace('binary_Bp', '').replace('.png', '')
        return int(num_str.split('-')[0])

    def detect_boxes(self, img):
        """Detect boxes of similar size arranged in a grid."""
        # Create a copy for visualization
        debug_img = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
        
        # Find all contours
        contours, _ = cv2.findContours(
            255 - img,  # Invert because boxes are white
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Get bounding rectangles for all contours
        rectangles = [cv2.boundingRect(c) for c in contours]
        
        if not rectangles:
            print("No rectangles found!")
            return []
            
        # Find the most common rectangle size (the box size we're looking for)
        areas = [w * h for x, y, w, h in rectangles]
        median_area = np.median(areas)
        
        # Filter rectangles by size (within 20% of median area)
        boxes = []
        for x, y, w, h in rectangles:
            area = w * h
            if 0.8 * median_area <= area <= 1.2 * median_area:
                boxes.append((x, y, w, h))
                
        # Sort boxes by position (top to bottom, then left to right)
        boxes.sort(key=lambda b: (b[1] // (img.shape[0] // 3), b[0]))
        
        # Verify we have exactly 36 boxes
        if len(boxes) != 36:
            print(f"Warning: Found {len(boxes)} boxes instead of expected 36")
        
        # Draw boxes for debugging
        for i, (x, y, w, h) in enumerate(boxes):
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Determine if box is on left or right side
            side = "L" if x < img.shape[1]/2 else "R"
            # Calculate position within its side (1-6)
            pos = (i % 12) % 6 + 1
            label = f"{side}{pos}"
            cv2.putText(debug_img, label, (x+5, y+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Show debug image
        cv2.imshow("Detected Boxes", debug_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Split into three sets
        sets = []
        height = img.shape[0] // 3
        for i in range(3):
            set_boxes = [b for b in boxes if i * height <= b[1] < (i + 1) * height]
            # Sort by x coordinate to separate left and right
            set_boxes.sort(key=lambda b: b[0])
            # Split into left and right sides
            mid_x = img.shape[1] / 2
            left_boxes = [b for b in set_boxes if b[0] < mid_x]
            right_boxes = [b for b in set_boxes if b[0] >= mid_x]
            # Sort each side by y coordinate
            left_boxes.sort(key=lambda b: b[1])
            right_boxes.sort(key=lambda b: b[1])
            # Combine sorted left and right boxes
            sets.append((left_boxes, right_boxes))
            
        return sets

    def extract_content(self, img, x, y, w, h):
        """Extract the content from inside a box, ignoring the border."""
        try:
            margin = 10  # Adjust this value to control how much of the border to remove
            
            # Ensure coordinates are within image bounds
            y1 = max(y + margin, 0)
            y2 = min(y + h - margin, img.shape[0])
            x1 = max(x + margin, 0)
            x2 = min(x + w - margin, img.shape[1])
            
            # Check if we have valid dimensions
            if x2 <= x1 or y2 <= y1:
                print(f"Warning: Invalid box dimensions: x1={x1}, x2={x2}, y1={y1}, y2={y2}")
                return None
                
            content = img[y1:y2, x1:x2]
            
            # Verify we got valid content
            if content.size == 0:
                print(f"Warning: Empty content extracted")
                return None
                
            return content
        except Exception as e:
            print(f"Error extracting content: {e}")
            return None

    def split_image(self, image_path):
        """Split a single image into left and right boxes for each set."""
        try:
            # Read image
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Error: Could not open {image_path}")
                return

            # Get the BP number
            bp_num = self.extract_number(image_path.name)
            
            # Detect boxes
            print(f"Detecting boxes in {image_path.name}...")
            box_sets = self.detect_boxes(img)
            
            # Process each set
            for set_idx, (left_boxes, right_boxes) in enumerate(box_sets):
                current_bp = bp_num + set_idx
                print(f"\nProcessing set {set_idx + 1} (BP{current_bp})")
                
                # Process left boxes
                for box_idx, (x, y, w, h) in enumerate(left_boxes, 1):
                    try:
                        # Extract the inner content of the left box
                        content = self.extract_content(img, x, y, w, h)
                        
                        if content is not None:
                            # Create output filename
                            output_filename = f"BP{current_bp}_L{box_idx}.png"
                            output_path = self.output_folder / output_filename
                            
                            # Save the content
                            cv2.imwrite(str(output_path), content)
                            print(f"Saved: {output_filename}")
                        else:
                            print(f"Skipping left box {box_idx} due to extraction error")
                            
                    except Exception as e:
                        print(f"Error processing left box {box_idx}: {e}")
                        continue
                
                # Process right boxes
                for box_idx, (x, y, w, h) in enumerate(right_boxes, 1):
                    try:
                        # Extract the inner content of the right box
                        content = self.extract_content(img, x, y, w, h)
                        
                        if content is not None:
                            # Create output filename
                            output_filename = f"BP{current_bp}_R{box_idx}.png"
                            output_path = self.output_folder / output_filename
                            
                            # Save the content
                            cv2.imwrite(str(output_path), content)
                            print(f"Saved: {output_filename}")
                        else:
                            print(f"Skipping right box {box_idx} due to extraction error")
                            
                    except Exception as e:
                        print(f"Error processing right box {box_idx}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    def process_all_images(self):
        """Process all binary PNG images in the input folder."""
        image_files = sorted(
            f for f in self.input_folder.glob("binary_Bp*.png")
            if f.is_file()
        )
        
        if not image_files:
            print(f"No binary PNG files found in {self.input_folder}")
            return
        
        print(f"Found {len(image_files)} images to process")
        
        for image_path in image_files:
            print(f"\nProcessing: {image_path.name}")
            try:
                self.split_image(image_path)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                print("Continuing with next image...")
                continue
            
        print("\nProcessing complete!")

if __name__ == "__main__":
    # Create splitter instance
    splitter = BoxSplitter()
    
    # Process all images
    splitter.process_all_images() 