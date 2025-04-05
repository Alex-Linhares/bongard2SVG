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
    
    # Check if the path is thin (like a line) by comparing area to perimeter
    area = np.sum(mask_array)
    perimeter = len(points)  # Approximate perimeter using number of points
    if area / perimeter < 3:  # Threshold for thin shapes
        return False
    
    # Check if the original image has black pixels inside the path
    inside_pixels = img_array[mask_array > 0]
    black_ratio = np.sum(inside_pixels <= 128) / len(inside_pixels)
    
    return black_ratio > 0.5  # If more than 50% of pixels are black, fill the path

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