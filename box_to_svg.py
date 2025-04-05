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
        
        # Create input folder if it doesn't exist
        if not self.input_folder.exists():
            print(f"Creating input folder: {self.input_folder}")
            self.input_folder.mkdir(parents=True, exist_ok=True)
        
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
            # Calculate basic shape metrics
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                return False
                
            # Calculate circularity (1.0 = perfect circle)
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Get bounding box and calculate aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w)/h if h > 0 else 0
            
            # Calculate solidity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                return False
            solidity = float(area)/hull_area
            
            # Calculate extent (area to bounding box ratio)
            extent = float(area)/(w*h) if w*h > 0 else 0
            
            # Fit an ellipse and get its parameters
            ellipse = cv2.fitEllipse(contour)
            center, axes, angle = ellipse
            
            # Calculate axes ratio
            axes_ratio = max(axes) / min(axes) if min(axes) > 0 else float('inf')
            
            # Print debug info
            print(f"  Shape Analysis:")
            print(f"    Area: {area:.1f}")
            print(f"    Circularity: {circularity:.3f}")
            print(f"    Aspect ratio: {aspect_ratio:.3f}")
            print(f"    Solidity: {solidity:.3f}")
            print(f"    Extent: {extent:.3f}")
            print(f"    Axes ratio: {axes_ratio:.3f}")
            
            # Different criteria based on size
            if area < 1000:  # Small shapes
                # Stricter criteria for small shapes
                is_ellipse = (
                    circularity > 0.85 and  # Very circular
                    0.8 < aspect_ratio < 1.2 and  # Nearly square bounding box
                    solidity > 0.92 and  # Very solid
                    extent > 0.7 and  # Fills bounding box well
                    axes_ratio < 1.3  # Nearly equal axes
                )
            else:  # Large shapes
                # Much more lenient criteria for large shapes
                is_ellipse = (
                    circularity > 0.65 and  # Much less strict circularity
                    0.5 < aspect_ratio < 2.0 and  # Allow more elongated shapes
                    solidity > 0.8 and  # Allow less solid shapes
                    extent > 0.55 and  # Allow less filling
                    axes_ratio < 2.0  # Allow more elongated ellipses
                )
            
            print(f"    Classified as: {'ellipse' if is_ellipse else 'not ellipse'}")
            return is_ellipse
            
        except Exception as e:
            print(f"Error in ellipse check: {str(e)}")
            return False

    def is_square_like(self, contour):
        """Check if a contour is likely to be a square or rectangle."""
        # Get the basic metrics
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return False
            
        # Get approximated polygon
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = [(point[0][0], point[0][1]) for point in approx]
        
        # Must have 4 points for a square/rectangle
        if len(points) != 4:
            return False
            
        # Calculate angles between sides
        angles = []
        for i in range(4):
            p1 = np.array(points[i])
            p2 = np.array(points[(i + 1) % 4])
            p3 = np.array(points[(i + 2) % 4])
            
            # Calculate vectors
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Calculate angle
            dot = np.dot(v1, v2)
            norm = np.linalg.norm(v1) * np.linalg.norm(v2)
            if norm == 0:
                return False
            cos_angle = dot / norm
            angle = np.abs(np.arccos(cos_angle) * 180 / np.pi)
            angles.append(angle)
        
        # Check if angles are close to 90 degrees
        is_rectangular = all(abs(angle - 90) < 15 for angle in angles)
        
        # Get bounding box properties
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w)/h if h > 0 else 0
        if aspect_ratio < 1:
            aspect_ratio = 1/aspect_ratio
            
        # Calculate how well the contour fills its bounding box
        extent = float(area)/(w*h) if w*h > 0 else 0
        
        # Calculate solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area)/hull_area if hull_area > 0 else 0
        
        print(f"  Square Analysis:")
        print(f"    Angles: {[round(a, 1) for a in angles]}")
        print(f"    Aspect ratio: {aspect_ratio:.3f}")
        print(f"    Extent: {extent:.3f}")
        print(f"    Solidity: {solidity:.3f}")
        
        # Criteria for being square-like:
        is_square = (
            is_rectangular and  # Angles close to 90 degrees
            aspect_ratio < 2.0 and  # Not too elongated
            extent > 0.8 and  # Fills bounding box well
            solidity > 0.85  # Very solid shape
        )
        
        print(f"    Is square-like: {is_square}")
        return is_square

    def process_box(self, box_path):
        """Process a single box image and convert it to SVG."""
        try:
            # Check if input file exists
            if not os.path.exists(box_path):
                print(f"Error: Input file not found: {box_path}")
                return

            # Read the image
            image = cv2.imread(str(box_path))
            if image is None:
                print(f"Error: Could not read image {box_path}")
                return

            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(box_path)
            if not os.path.exists(output_dir):
                print(f"Creating output directory: {output_dir}")
                os.makedirs(output_dir, exist_ok=True)

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
                area = cv2.contourArea(contour)
                if area < 30:  # Skip very small contours
                    continue

                print(f"\nAnalyzing contour with area: {area:.1f}")
                
                # Check for square-like shapes first
                if self.is_square_like(contour):
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    points = [(point[0][0], point[0][1]) for point in approx]
                    mask = np.zeros(binary.shape, dtype=np.uint8)
                    cv2.drawContours(mask, [contour], 0, 255, -1)
                    mean_val = cv2.mean(binary, mask=mask)[0]
                    is_filled = mean_val < 127
                    shapes_to_draw.append(('polygon', (points, is_filled)))
                    found_shapes = True
                    print("  Classified as square/rectangle")
                    continue
                
                # For large shapes, check for ellipses
                if area >= 1000:
                    if self.is_elliptical(contour):
                        ellipse = cv2.fitEllipse(contour)
                        center, axes, angle = ellipse
                        mask = np.zeros(binary.shape, dtype=np.uint8)
                        cv2.drawContours(mask, [contour], 0, 255, -1)
                        mean_val = cv2.mean(binary, mask=mask)[0]
                        is_filled = mean_val < 127
                        shapes_to_draw.append(('ellipse', (center, axes, angle, is_filled)))
                        found_shapes = True
                        print("  Large shape classified as ellipse")
                        continue
                
                # For smaller shapes or if large shape wasn't elliptical, try polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                points = [(point[0][0], point[0][1]) for point in approx]
                
                # Check if it's a good polygon (3-8 sides)
                if 3 <= len(points) <= 8:
                    # Calculate polygon metrics
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0
                    
                    # If it's a good polygon (high solidity), use it
                    if solidity > 0.85:
                        mask = np.zeros(binary.shape, dtype=np.uint8)
                        cv2.drawContours(mask, [contour], 0, 255, -1)
                        mean_val = cv2.mean(binary, mask=mask)[0]
                        is_filled = mean_val < 127
                        shapes_to_draw.append(('polygon', (points, is_filled)))
                        found_shapes = True
                        print("  Classified as simple polygon")
                        continue
                
                # If not a good polygon and small shape, check if it's elliptical
                if area < 1000:
                    if self.is_elliptical(contour):
                        ellipse = cv2.fitEllipse(contour)
                        center, axes, angle = ellipse
                        mask = np.zeros(binary.shape, dtype=np.uint8)
                        cv2.drawContours(mask, [contour], 0, 255, -1)
                        mean_val = cv2.mean(binary, mask=mask)[0]
                        is_filled = mean_val < 127
                        shapes_to_draw.append(('ellipse', (center, axes, angle, is_filled)))
                        found_shapes = True
                        print("  Small shape classified as ellipse")
                        continue
                
                # If we get here, try as a complex polygon with more precise approximation
                epsilon = 0.01 * cv2.arcLength(contour, True)  # More precise approximation
                approx = cv2.approxPolyDP(contour, epsilon, True)
                points = [(point[0][0], point[0][1]) for point in approx]
                mask = np.zeros(binary.shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], 0, 255, -1)
                mean_val = cv2.mean(binary, mask=mask)[0]
                is_filled = mean_val < 127
                shapes_to_draw.append(('polygon', (points, is_filled)))
                found_shapes = True
                print("  Classified as complex polygon")

            # Draw all shapes
            if found_shapes:
                existing_ellipses = []
                existing_polygons = []

                for shape_type, params in shapes_to_draw:
                    if shape_type == 'ellipse':
                        center, axes, angle, is_filled = params
                        if not self.is_similar_ellipse(center, axes, angle, existing_ellipses):
                            dwg.add(dwg.ellipse(
                                center=center,
                                r=(axes[0]/2, axes[1]/2),
                                transform=f'rotate({angle}, {center[0]}, {center[1]})',
                                fill='black' if is_filled else 'none',
                                stroke='black',
                                stroke_width=5
                            ))
                            existing_ellipses.append((center, axes, angle))
                    else:  # polygon
                        points, is_filled = params
                        if not self.is_similar_polygon(points, existing_polygons):
                            path_data = 'M ' + ' L '.join([f'{x},{y}' for x, y in points]) + ' Z'
                            dwg.add(dwg.path(
                                d=path_data,
                                fill='black' if is_filled else 'none',
                                stroke='black',
                                stroke_width=5
                            ))
                            existing_polygons.append(points)

            # Save the SVG file
            dwg.save()
            print(f"Created: {os.path.basename(svg_path)}")

        except Exception as e:
            print(f"Error processing {box_path}: {str(e)}")

    def process_all_boxes(self):
        """Process all box images in the input folder."""
        # Check if input folder exists
        if not self.input_folder.exists():
            print(f"Error: Input folder not found: {self.input_folder}")
            return

        # Get all PNG files
        box_files = list(self.input_folder.glob("BP*.png"))

        if not box_files:
            print(f"No PNG files found in {self.input_folder}")
            print("Expected files should be named like: BP1_R1.png, BP1_R2.png, etc.")
            return

        print(f"Found {len(box_files)} box images to process")

        # Sort files by name
        box_files.sort(key=lambda x: x.name)

        # Process each box
        for box_path in box_files:
            try:
                print(f"\nProcessing: {box_path}")
                self.process_box(str(box_path))  # Convert Path to string
            except Exception as e:
                print(f"Error processing {box_path.name}: {e}")
                continue
        print("\nProcessing complete!")


if __name__ == "__main__":
    # Create converter instance
    converter = BoxToSVGConverter()
    
    # Process all boxes
    converter.process_all_boxes() 