from PIL import Image
import potrace
import svgwrite
import numpy as np
import glob
import os
from PIL import ImageDraw

def curve_to_path_data(curve):
    """Convert a curve to SVG path data."""
    segments = []
    # Handle start point as a tuple
    segments.append('M %f,%f' % curve.start_point)
    for segment in curve.segments:
        if segment.is_corner:
            segments.append('L %f,%f' % segment.c)
            segments.append('L %f,%f' % segment.end_point)
        else:
            segments.append('C %f,%f %f,%f %f,%f' % (
                segment.c1[0], segment.c1[1],
                segment.c2[0], segment.c2[1],
                segment.end_point[0], segment.end_point[1]))
    # Don't close the path automatically
    return ' '.join(segments)

def get_path_bounds(curve):
    """Get the bounding box of a path."""
    points = [curve.start_point]
    for segment in curve.segments:
        if segment.is_corner:
            points.append(segment.c)
            points.append(segment.end_point)
        else:
            points.append(segment.c1)
            points.append(segment.c2)
            points.append(segment.end_point)
    
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)

def is_path_nested(curve1, curve2):
    """Check if curve1 is nested inside curve2."""
    x1, y1, x2, y2 = get_path_bounds(curve1)
    X1, Y1, X2, Y2 = get_path_bounds(curve2)
    # Check if curve1's bounds are completely inside curve2's bounds with some margin
    margin = 2  # Allow for some numerical imprecision
    return (x1 >= X1 - margin and x2 <= X2 + margin and 
            y1 >= Y1 - margin and y2 <= Y2 + margin and
            (x1 > X1 or x2 < X2 or y1 > Y1 or y2 < Y2))  # At least one bound must be strictly inside

def is_circle_like(points, area):
    """Check if a path resembles a circle."""
    # Get centroid
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    
    # Calculate average radius and variance
    radii = [(((p[0] - cx) ** 2 + (p[1] - cy) ** 2) ** 0.5) for p in points]
    avg_radius = sum(radii) / len(radii)
    variance = sum((r - avg_radius) ** 2 for r in radii) / len(radii)
    
    # Circle should have low radius variance relative to its size
    return variance / avg_radius < 0.2

def is_triangle_like(points):
    """Check if a path resembles a triangle."""
    if len(points) < 3:
        return False
        
    # Get three points with maximum distance between them
    max_dist = 0
    corners = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = ((points[i][0] - points[j][0]) ** 2 + 
                   (points[i][1] - points[j][1]) ** 2)
            if dist > max_dist:
                max_dist = dist
                corners = [points[i], points[j]]
    
    # Find third point with maximum distance from line
    max_dist = 0
    for p in points:
        if p not in corners:
            # Distance from point to line
            x0, y0 = p
            x1, y1 = corners[0]
            x2, y2 = corners[1]
            dist = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / ((y2-y1)**2 + (x2-x1)**2)**0.5
            if dist > max_dist:
                max_dist = dist
                corners.append(p)
    
    if len(corners) < 3:
        return False
        
    # Check if other points are close to the triangle edges
    for p in points:
        if p in corners:
            continue
        # Calculate minimum distance to any edge
        min_dist = float('inf')
        for i in range(3):
            x1, y1 = corners[i]
            x2, y2 = corners[(i + 1) % 3]
            dist = abs((y2-y1)*p[0] - (x2-x1)*p[1] + x2*y1 - y2*x1) / ((y2-y1)**2 + (x2-x1)**2)**0.5
            min_dist = min(min_dist, dist)
        if min_dist > 5:  # Allow some deviation from perfect triangle
            return False
            
    return True

def should_fill_path(curve, img_array):
    """Check if a path should be filled by analyzing the original image."""
    # Create a mask image
    mask = Image.new('L', (img_array.shape[1], img_array.shape[0]), 0)
    draw = ImageDraw.Draw(mask)
    
    # Convert curve points to polygon
    points = []
    points.append(curve.start_point)
    for segment in curve.segments:
        if segment.is_corner:
            points.append(segment.c)
            points.append(segment.end_point)
        else:
            # Approximate bezier curve with points
            points.append(segment.c1)
            points.append(segment.c2)
            points.append(segment.end_point)
    
    # Draw polygon on mask
    draw.polygon(points, fill=1)
    mask_array = np.array(mask)
    
    # Calculate area and perimeter
    area = np.sum(mask_array)
    perimeter = len(points)  # Approximate perimeter using number of points
    
    # Don't fill if object is too small (likely noise or small details)
    if area < 100:  # Minimum area threshold
        return False
        
    # Don't fill if object is thin (like a line)
    if area / perimeter < 5:  # Increased threshold for better line detection
        return False
    
    # For small objects, check if they're circles or triangles
    if area < 1000:
        # Never fill small circles or triangles
        if is_circle_like(points, area) or is_triangle_like(points):
            return False
    
    # Check if the original image has black pixels inside the path
    inside_pixels = img_array[mask_array > 0]
    black_ratio = np.sum(inside_pixels <= 128) / len(inside_pixels)
    
    # More strict fill criteria for small objects
    if area < 500:  # For small objects
        return black_ratio > 0.9  # Even higher black ratio for small objects
    
    return black_ratio > 0.85  # Threshold for larger objects

def image_to_svg(input_image, output_file):
    img = Image.open(input_image).convert("L")  # Grayscale
    # Convert PIL Image to numpy array and threshold to binary
    img_array = np.array(img)
    # Create binary image using threshold - invert so black (0) becomes 1
    threshold = 128
    img_binary = (img_array <= threshold).astype(np.uint8)
    # Create bitmap from binary array
    bitmap = potrace.Bitmap(img_binary)
    path = bitmap.trace()  # Vectorize
    dwg = svgwrite.Drawing(output_file, size=(img.width, img.height))
    
    # Convert path to list for easier manipulation
    curves = list(path)
    
    # Filter out nested paths
    curves_to_draw = []
    for i, curve in enumerate(curves):
        is_nested = False
        for j, other_curve in enumerate(curves):
            if i != j and is_path_nested(curve, other_curve):
                is_nested = True
                break
        if not is_nested:
            curves_to_draw.append(curve)
    
    # Process each path
    for curve in curves_to_draw:
        path_data = curve_to_path_data(curve)
        # Determine if path should be filled
        if should_fill_path(curve, img_array):
            dwg.add(dwg.path(d=path_data, stroke="black", fill="black", stroke_width="1"))
        else:
            dwg.add(dwg.path(d=path_data, stroke="black", fill="none", stroke_width="5"))
    dwg.save(output_file)

if __name__ == "__main__":
    # Process all PNG files in the boxes directory
    png_files = glob.glob('boxes/*.png')
    for png_file in png_files:
        svg_file = png_file.replace('.png', '.svg')
        print(f"Converting {png_file} to {svg_file}")
        image_to_svg(png_file, svg_file)