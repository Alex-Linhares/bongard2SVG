import cv2
import numpy as np
from pathlib import Path
import os
import svgwrite
from percentage_inside_contour import ContourAnalyzer

class BoxToSVGConverter:
    def __init__(self, input_folder="boxes"):
        """Initialize with input folder path."""
        self.input_folder = Path(input_folder)
        self.analyzer = ContourAnalyzer()
        
    def is_similar_circle(self, x1, y1, r1, existing_circles, tolerance=0.35):
        """Check if a circle is similar to any existing circles."""
        for x2, y2, r2 in existing_circles:
            # Check if centers are close
            center_dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            # Check if one circle is inside the other and has similar radius
            if center_dist < max(r1, r2) and abs(r1 - r2) / max(r1, r2) < tolerance:
                return True
        return False

    def is_similar_polygon(self, points1, existing_polygons, tolerance=0.35):
        """Check if a polygon is similar to any existing polygons."""
        # Convert points to numpy array for easier computation
        poly1 = np.array(points1)
        center1 = np.mean(poly1, axis=0)
        
        for points2 in existing_polygons:
            poly2 = np.array(points2)
            center2 = np.mean(poly2, axis=0)
            
            # Check if centers are close
            center_dist = np.sqrt(np.sum((center1 - center2) ** 2))
            
            # Get bounding boxes
            x1, y1, w1, h1 = cv2.boundingRect(poly1.astype(np.float32))
            x2, y2, w2, h2 = cv2.boundingRect(poly2.astype(np.float32))
            
            # Compare sizes
            size1 = max(w1, h1)
            size2 = max(w2, h2)
            
            # If centers are close and sizes are similar
            if (center_dist < max(size1, size2) * tolerance and 
                abs(size1 - size2) / max(size1, size2) < tolerance):
                return True
                
        return False

    def is_similar_line(self, points1, existing_lines, tolerance=0.35):
        """Check if a line is similar to any existing lines."""
        if len(points1) != 2:
            return False
            
        start1, end1 = points1
        length1 = np.sqrt((end1[0] - start1[0])**2 + (end1[1] - start1[1])**2)
        
        for points2 in existing_lines:
            if len(points2) != 2:
                continue
                
            start2, end2 = points2
            length2 = np.sqrt((end2[0] - start2[0])**2 + (end2[1] - start2[1])**2)
            
            # Check if lines have similar length
            if abs(length1 - length2) / max(length1, length2) > tolerance:
                continue
                
            # Check if endpoints are close
            dist1 = min(
                np.sqrt((start1[0] - start2[0])**2 + (start1[1] - start2[1])**2),
                np.sqrt((start1[0] - end2[0])**2 + (start1[1] - end2[1])**2)
            )
            dist2 = min(
                np.sqrt((end1[0] - start2[0])**2 + (end1[1] - start2[1])**2),
                np.sqrt((end1[0] - end2[0])**2 + (end1[1] - end2[1])**2)
            )
            
            if dist1 < length1 * tolerance and dist2 < length1 * tolerance:
                return True
                
        return False

    def is_line(self, contour, points):
        """Check if a contour represents a line."""
        # Calculate area and perimeter
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # A line should have very small area but significant perimeter
        # This gives us a high perimeter-to-area ratio
        if area == 0:
            return perimeter > 1  # If area is zero, just check if perimeter is significant
        else:
            perimeter_area_ratio = perimeter / area  # Higher for line-like shapes
            # return perimeter_area_ratio > 400 and area < 100  # Thresholds can be adjusted
            return perimeter_area_ratio > area 

    def is_point_near_line(self, px, py, line_x1, line_y1, line_x2, line_y2, threshold=20):
        """Check if a point is near a line segment."""
        # Calculate distance from point to line segment
        # Based on the formula for point-to-line-segment distance
        line_length = np.sqrt((line_x2 - line_x1)**2 + (line_y2 - line_y1)**2)
        if line_length == 0:
            return np.sqrt((px - line_x1)**2 + (py - line_y1)**2) < threshold
            
        t = max(0, min(1, ((px - line_x1) * (line_x2 - line_x1) + 
                          (py - line_y1) * (line_y2 - line_y1)) / (line_length**2)))
        
        proj_x = line_x1 + t * (line_x2 - line_x1)
        proj_y = line_y1 + t * (line_y2 - line_y1)
        
        distance = np.sqrt((px - proj_x)**2 + (py - proj_y)**2)
        return distance < threshold

    def is_edge_near_any_line(self, x1, y1, x2, y2, existing_lines, threshold=20):
        """Check if a polygon edge is too close to any existing line."""
        # Check both endpoints of the edge against all existing lines
        for line_points in existing_lines:
            line_x1, line_y1 = line_points[0]
            line_x2, line_y2 = line_points[1]
            
            # If both endpoints are near the line, this edge is probably duplicate
            if (self.is_point_near_line(x1, y1, line_x1, line_y1, line_x2, line_y2, threshold) and
                self.is_point_near_line(x2, y2, line_x1, line_y1, line_x2, line_y2, threshold)):
                return True
        return False

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

            # Create binary image
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

            # Create SVG file path
            svg_path = os.path.splitext(box_path)[0] + '.svg'

            # Create SVG with original dimensions
            dwg = svgwrite.Drawing(svg_path, size=(width, height))

            # Add white background
            dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill='white'))

            # Keep track of shapes we've drawn
            existing_circles = []
            existing_polygons = []
            existing_lines = []
            polygon_edges = []  # Keep track of all polygon edges

            # First find all polygon edges
            contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                # Skip contours that don't meet the white pixel percentage criteria
                if self.analyzer.get_white_pixel_percentage_in_contour(box_path, contour, threshold=230) < 0.65:
                    continue

                # Approximate the contour
                epsilon = 0.04 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                points = [(point[0][0], point[0][1]) for point in approx]

                # Check if it's a circle
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                if circularity <= 0.8 or len(points) <= 4:  # If not a circle
                    # Add all edges of the polygon
                    for i in range(len(points)):
                        x1, y1 = points[i]
                        x2, y2 = points[(i + 1) % len(points)]
                        polygon_edges.append(((x1, y1), (x2, y2)))

            # Now detect lines using Hough Transform
            lines = cv2.HoughLinesP(
                255 - binary,  # Invert image since we want to detect black lines
                rho=1,
                theta=np.pi/180,
                threshold=80,
                minLineLength=60,
                maxLineGap=0
            )

            # Draw Hough lines that aren't near polygon edges
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    points = [(x1, y1), (x2, y2)]
                    
                    # Skip if this line is near any polygon edge
                    is_near_polygon = False
                    for edge_start, edge_end in polygon_edges:
                        if (self.is_point_near_line(x1, y1, edge_start[0], edge_start[1], edge_end[0], edge_end[1], threshold=20) and
                            self.is_point_near_line(x2, y2, edge_start[0], edge_start[1], edge_end[0], edge_end[1], threshold=20)):
                            is_near_polygon = True
                            break
                    
                    if not is_near_polygon and not self.is_similar_line(points, existing_lines):
                        path_data = f'M {x1},{y1} L {x2},{y2}'
                        dwg.add(dwg.path(d=path_data,
                                       fill='none',
                                       stroke='black',
                                       stroke_width=5))
                        existing_lines.append(points)

            # Process contours again to draw circles and polygons
            for contour in contours:
                if self.analyzer.get_white_pixel_percentage_in_contour(box_path, contour, threshold=230) < 0.65:
                    continue

                epsilon = 0.04 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                points = [(point[0][0], point[0][1]) for point in approx]

                # Create a mask and check if the shape is filled
                mask = np.zeros(binary.shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], 0, 255, -1)
                mean_val = cv2.mean(binary, mask=mask)[0]
                is_filled = mean_val < 127

                # Check if it's a circle
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                if circularity > 0.8 and len(points) > 4:
                    # Get circle parameters
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    
                    # Check if we already have a similar circle
                    if not self.is_similar_circle(x, y, radius, existing_circles):
                        # Draw circle
                        dwg.add(dwg.circle(center=(x, y), r=radius,
                                         fill='black' if is_filled else 'none',
                                         stroke='black',
                                         stroke_width=5))
                        # Remember this circle
                        existing_circles.append((x, y, radius))
                else:
                    # Check if we already have a similar polygon
                    if not self.is_similar_polygon(points, existing_polygons):
                        # Draw polygon
                        path_data = 'M ' + ' L '.join([f'{x},{y}' for x, y in points]) + ' Z'
                        dwg.add(dwg.path(d=path_data,
                                       fill='black' if is_filled else 'none',
                                       stroke='black',
                                       stroke_width=5))
                        # Remember this polygon
                        existing_polygons.append(points)

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