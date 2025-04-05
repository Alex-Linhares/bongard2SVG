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

    def is_point_near_line(self, px, py, line_x1, line_y1, line_x2, line_y2, threshold=50):
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

    def is_similar_ellipse(self, center1, size1, angle1, existing_ellipses, tolerance=0.35):
        """Check if an ellipse is similar to any existing ellipses."""
        for center2, size2, angle2 in existing_ellipses:
            # Check if centers are close
            center_dist = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
            
            # Compare sizes (major and minor axes)
            max_size1 = max(size1)
            max_size2 = max(size2)
            size_diff = abs(max_size1 - max_size2) / max(max_size1, max_size2)
            
            # If centers are close and sizes are similar
            if (center_dist < max(max_size1, max_size2) * tolerance and 
                size_diff < tolerance):
                return True
        return False

    def get_ellipse_points(self, center, axes, angle, num_points=32):
        """Get points along an ellipse's perimeter."""
        # Convert angle to radians
        angle_rad = angle * np.pi / 180.0
        
        # Generate points along the ellipse
        t = np.linspace(0, 2*np.pi, num_points)
        
        # Ellipse parametric equations before rotation
        x = axes[0]/2 * np.cos(t)
        y = axes[1]/2 * np.sin(t)
        
        # Rotate points
        x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
        y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)
        
        # Translate to center
        x_final = x_rot + center[0]
        y_final = y_rot + center[1]
        
        return list(zip(x_final, y_final))

    def get_line_angle(self, x1, y1, x2, y2):
        """Calculate the angle of a line in degrees."""
        return np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 360

    def get_ellipse_tangent_angles(self, center, axes, angle, num_points=32):
        """Get tangent angles at points along an ellipse."""
        # Convert angle to radians
        angle_rad = angle * np.pi / 180.0
        
        # Generate points along the ellipse parameter
        t = np.linspace(0, 2*np.pi, num_points)
        
        # Calculate tangent angles before rotation
        a, b = axes[0]/2, axes[1]/2
        tangent_angles = np.degrees(np.arctan2(-a * np.sin(t), b * np.cos(t)))
        
        # Adjust for ellipse rotation
        tangent_angles = (tangent_angles + angle) % 360
        
        return tangent_angles

    def angles_are_similar(self, angle1, angle2, tolerance=20):
        """Check if two angles are similar, considering the circular nature of angles."""
        diff = abs((angle1 - angle2 + 180) % 360 - 180)
        return diff < tolerance or diff > (360 - tolerance)

    def is_line_path_near_edge(self, x1, y1, x2, y2, edge_x1, edge_y1, edge_x2, edge_y2, num_points=10):
        """Check if any point along a line path is near an edge."""
        # Check multiple points along the line
        for t in np.linspace(0, 1, num_points):
            px = x1 + t * (x2 - x1)
            py = y1 + t * (y2 - y1)
            if self.is_point_near_line(px, py, edge_x1, edge_y1, edge_x2, edge_y2):
                return True
        return False

    def is_line_near_shape_edges(self, x1, y1, x2, y2, polygon_edges, ellipse_params, threshold=50):
        """Check if a line is near any polygon edge or ellipse edge."""
        line_angle = self.get_line_angle(x1, y1, x2, y2)
        line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Check against polygon edges
        for edge_start, edge_end in polygon_edges:
            # Check if any point along the line is near the edge
            if self.is_line_path_near_edge(x1, y1, x2, y2, edge_start[0], edge_start[1], edge_end[0], edge_end[1]):
                edge_angle = self.get_line_angle(edge_start[0], edge_start[1], edge_end[0], edge_end[1])
                if self.angles_are_similar(line_angle, edge_angle, tolerance=30):
                    return True

        # Check against ellipse edges
        for center, axes, angle in ellipse_params:
            # Get more points along the ellipse for better coverage
            ellipse_points = self.get_ellipse_points(center, axes, angle, num_points=64)
            tangent_angles = self.get_ellipse_tangent_angles(center, axes, angle, num_points=64)
            
            # Add extra points around the ellipse for a safety margin
            margin = 10
            expanded_axes = (axes[0] + margin, axes[1] + margin)
            outer_points = self.get_ellipse_points(center, expanded_axes, angle, num_points=64)
            
            # Check both the ellipse points and the expanded margin
            all_points = ellipse_points + outer_points
            
            # Check if any point along the line is near any ellipse segment
            for i in range(len(all_points)):
                p1 = all_points[i]
                p2 = all_points[(i + 1) % len(all_points)]
                
                if self.is_line_path_near_edge(x1, y1, x2, y2, p1[0], p1[1], p2[0], p2[1]):
                    # For actual ellipse points (not margin points), also check angle
                    if i < len(ellipse_points):
                        if self.angles_are_similar(line_angle, tangent_angles[i], tolerance=30):
                            return True
                    else:
                        return True  # If near margin points, reject anyway

        return False

    def point_inside_ellipse(self, px, py, center, axes, angle):
        """Check if a point is inside an ellipse."""
        # Translate point to origin
        px -= center[0]
        py -= center[1]
        
        # Rotate point to align with ellipse axes
        angle_rad = angle * np.pi / 180.0
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        
        x_rot = px * cos_angle + py * sin_angle
        y_rot = -px * sin_angle + py * cos_angle
        
        # Check if point is inside ellipse equation
        return ((x_rot / (axes[0]/2))**2 + (y_rot / (axes[1]/2))**2) <= 1.0

    def point_inside_polygon(self, px, py, points):
        """Check if a point is inside a polygon using ray casting algorithm."""
        n = len(points)
        inside = False
        j = n - 1
        
        for i in range(n):
            if (((points[i][1] > py) != (points[j][1] > py)) and
                (px < (points[j][0] - points[i][0]) * (py - points[i][1]) /
                 (points[j][1] - points[i][1]) + points[i][0])):
                inside = not inside
            j = i
            
        return inside

    def line_intersects_shape(self, x1, y1, x2, y2, polygon_points, ellipse_params, num_points=10):
        """Check if a line passes through the interior of any shape."""
        # Check multiple points along the line
        for t in np.linspace(0.1, 0.9, num_points):  # Skip endpoints to avoid edge cases
            px = x1 + t * (x2 - x1)
            py = y1 + t * (y2 - y1)
            
            # Check if point is inside any ellipse
            for center, axes, angle in ellipse_params:
                if self.point_inside_ellipse(px, py, center, axes, angle):
                    return True
            
            # Check if point is inside any polygon
            if polygon_points:
                for points in polygon_points:
                    if self.point_inside_polygon(px, py, points):
                        return True
        
        return False

    def create_shape_mask(self, binary_image, contours, margin=20):
        """Create a mask that excludes shapes and their surroundings."""
        height, width = binary_image.shape
        mask = np.ones((height, width), dtype=np.uint8) * 255
        
        for contour in contours:
            # Create a slightly larger contour for the margin
            hull = cv2.convexHull(contour)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(hull)
            
            # Expand the bounding box by margin
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(width - x, w + 2*margin)
            h = min(height - y, h + 2*margin)
            
            # Draw filled contour with margin
            cv2.rectangle(mask, (x, y), (x + w, y + h), 0, -1)
        
        return mask

    def is_elliptical(self, contour):
        """Check if a contour is truly elliptical rather than polygonal."""
        # Must have enough points for an ellipse
        if len(contour) < 5:
            return False
            
        try:
            # Fit an ellipse
            ellipse = cv2.fitEllipse(contour)
            center, axes, angle = ellipse
            
            # Create masks for comparison
            h, w = cv2.boundingRect(contour)[2:]
            mask_size = (h + 20, w + 20)  # Add padding
            
            # Draw the original contour
            contour_mask = np.zeros(mask_size, dtype=np.uint8)
            shifted_contour = contour.copy()
            shifted_contour = shifted_contour.reshape(-1, 2)
            shifted_contour[:, 0] += 10  # Shift to account for padding
            shifted_contour[:, 1] += 10
            shifted_contour = shifted_contour.reshape(-1, 1, 2)
            cv2.drawContours(contour_mask, [shifted_contour], 0, 255, -1)
            
            # Draw the fitted ellipse
            ellipse_mask = np.zeros(mask_size, dtype=np.uint8)
            center = (center[0] + 10, center[1] + 10)  # Shift center too
            cv2.ellipse(ellipse_mask, 
                       [int(center[0]), int(center[1])],
                       [int(axes[0]/2), int(axes[1]/2)],
                       angle, 0, 360, 255, -1)
            
            # Calculate overlap and areas
            overlap = cv2.bitwise_and(contour_mask, ellipse_mask)
            overlap_area = cv2.countNonZero(overlap)
            ellipse_area = cv2.countNonZero(ellipse_mask)
            contour_area = cv2.countNonZero(contour_mask)
            
            # Calculate fit quality
            fit_quality = overlap_area / max(ellipse_area, contour_area)
            
            # Calculate perimeter efficiency (closer to 1 for ellipses)
            contour_perimeter = cv2.arcLength(contour, True)
            theoretical_ellipse_perimeter = np.pi * (axes[0] + axes[1]) / 2
            perimeter_ratio = min(contour_perimeter, theoretical_ellipse_perimeter) / max(contour_perimeter, theoretical_ellipse_perimeter)
            
            # Check contour smoothness
            angles = []
            for i in range(len(contour)):
                p1 = contour[i][0]
                p2 = contour[(i + 1) % len(contour)][0]
                p3 = contour[(i + 2) % len(contour)][0]
                
                v1 = p2 - p1
                v2 = p3 - p2
                
                dot = np.dot(v1, v2)
                norms = np.linalg.norm(v1) * np.linalg.norm(v2)
                
                if norms > 0:
                    cos_angle = dot / norms
                    cos_angle = min(1.0, max(-1.0, cos_angle))
                    angle = np.arccos(cos_angle)
                    angles.append(angle)
            
            angle_std = np.std(angles)
            
            # Return True only if all criteria are met
            return (fit_quality > 0.9 and  # Very high fit quality
                    perimeter_ratio > 0.85 and  # Efficient perimeter
                    angle_std < 0.5)  # Smooth angle changes
                    
        except:
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

            # Create SVG with original dimensions and no default border
            dwg = svgwrite.Drawing(svg_path, size=(width, height), style='background-color: transparent;')
            dwg.viewbox(minx=0, miny=0, width=width, height=height)
            
            # Add a style to ensure no default borders
            dwg.defs.add(dwg.style("""
                svg {
                    shape-rendering: geometricPrecision;
                    border: none;
                    outline: none;
                }
                path, ellipse {
                    vector-effect: non-scaling-stroke;
                }
            """))

            # Find all contours first
            contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by white pixel percentage and check if they touch borders
            valid_contours = []
            for contour in contours:
                # Skip contours that form a rectangle around the entire image
                x, y, w, h = cv2.boundingRect(contour)
                is_border_rectangle = (
                    abs(x) <= 5 and abs(y) <= 5 and  # Close to top-left corner
                    abs(w - width) <= 10 and abs(h - height) <= 10  # Close to image dimensions
                )
                if is_border_rectangle:
                    continue

                # Check if contour touches image border
                touches_border = (x <= 0 or y <= 0 or x + w >= width or y + h >= height)
                
                # If it touches border, expand the points slightly inward
                if touches_border:
                    # Create a slightly smaller bounding box
                    margin = 2
                    new_x = max(0, x)
                    new_y = max(0, y)
                    new_w = min(width - new_x, w)
                    new_h = min(height - new_y, h)
                    
                    # Adjust contour points to stay within bounds
                    contour = contour.reshape(-1, 2)
                    contour[:, 0] = np.clip(contour[:, 0], margin, width - margin)
                    contour[:, 1] = np.clip(contour[:, 1], margin, height - margin)
                    contour = contour.reshape(-1, 1, 2)

                if self.analyzer.get_white_pixel_percentage_in_contour(box_path, contour, threshold=230) >= 0.65:
                    valid_contours.append(contour)

            # First analyze all contours to detect shapes
            found_shapes = False
            shapes_to_draw = []  # Will store tuples of (shape_type, parameters)

            for contour in valid_contours:
                # First check if the contour is truly elliptical
                if self.is_elliptical(contour):
                    ellipse = cv2.fitEllipse(contour)
                    center, axes, angle = ellipse
                    
                    # Check if the shape is filled
                    mask = np.zeros(binary.shape, dtype=np.uint8)
                    cv2.drawContours(mask, [contour], 0, 255, -1)
                    mean_val = cv2.mean(binary, mask=mask)[0]
                    is_filled = mean_val < 127
                    
                    shapes_to_draw.append(('ellipse', (center, axes, angle, is_filled)))
                    found_shapes = True
                    continue

                # If not elliptical, try as polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                points = [(point[0][0], point[0][1]) for point in approx]

                if len(points) > 2:
                    mask = np.zeros(binary.shape, dtype=np.uint8)
                    cv2.drawContours(mask, [contour], 0, 255, -1)
                    mean_val = cv2.mean(binary, mask=mask)[0]
                    is_filled = mean_val < 127
                    shapes_to_draw.append(('polygon', (points, is_filled)))
                    found_shapes = True

            # If we found shapes, draw them
            if found_shapes:
                existing_ellipses = []
                existing_polygons = []

                for shape_type, params in shapes_to_draw:
                    if shape_type == 'ellipse':
                        center, axes, angle, is_filled = params
                        if not self.is_similar_ellipse(center, axes, angle, existing_ellipses):
                            dwg.add(dwg.ellipse(center=center,
                                              r=(axes[0]/2, axes[1]/2),
                                              transform=f'rotate({angle}, {center[0]}, {center[1]})',
                                              fill='black' if is_filled else 'none',
                                              stroke='black',
                                              stroke_width=5))
                            existing_ellipses.append((center, axes, angle))
                    else:  # polygon
                        points, is_filled = params
                        if not self.is_similar_polygon(points, existing_polygons):
                            path_data = 'M ' + ' L '.join([f'{x},{y}' for x, y in points]) + ' Z'
                            dwg.add(dwg.path(d=path_data,
                                           fill='black' if is_filled else 'none',
                                           stroke='black',
                                           stroke_width=5))
                            existing_polygons.append(points)

            # Only look for lines if no shapes were found
            else:
                lines = cv2.HoughLinesP(
                    255 - binary,
                    rho=1,
                    theta=np.pi/180,
                    threshold=120,
                    minLineLength=60,
                    maxLineGap=0
                )

                if lines is not None:
                    filtered_lines = []
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        
                        is_similar = False
                        line_angle = self.get_line_angle(x1, y1, x2, y2)
                        
                        for existing_line in filtered_lines:
                            ex1, ey1, ex2, ey2 = existing_line
                            if self.is_line_path_near_edge(x1, y1, x2, y2, ex1, ey1, ex2, ey2):
                                existing_angle = self.get_line_angle(ex1, ey1, ex2, ey2)
                                if self.angles_are_similar(line_angle, existing_angle, tolerance=30):
                                    is_similar = True
                                    break
                        
                        if not is_similar:
                            filtered_lines.append([x1, y1, x2, y2])
                            path_data = f'M {x1},{y1} L {x2},{y2}'
                            dwg.add(dwg.path(d=path_data,
                                           fill='none',
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