from PIL import Image
import potrace
import svgwrite
import numpy as np
import glob
import os

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
    return ' '.join(segments) + ' z'

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
    # Process each path
    for curve in path:
        path_data = curve_to_path_data(curve)
        # Use stroke for line drawings
        dwg.add(dwg.path(d=path_data, stroke="black", stroke_width="5"))
    dwg.save(output_file)

if __name__ == "__main__":
    # Process all PNG files in the boxes directory
    png_files = glob.glob('boxes/*.png')
    for png_file in png_files:
        svg_file = png_file.replace('.png', '.svg')
        print(f"Converting {png_file} to {svg_file}")
        image_to_svg(png_file, svg_file)