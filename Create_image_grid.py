from PIL import Image
import os
import shutil
import cairosvg
import io

def create_grid_comparison(image_dir, output_dir="comparisons"):
    """Creates a 6x2 grid for each BP problem, with PNGs above and SVGs below."""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each box problem
    for i in range(1, 101):  # Loop through BP1 to BP100
        for side in ['L', 'R']:
            # First check if we have all 6 images for this BP and side
            all_files_exist = True
            images = []
            
            # Get sample image to determine size
            sample_png = f"BP{i}_{side}1.png"
            sample_path = os.path.join(image_dir, sample_png)
            if not os.path.exists(sample_path):
                continue
                
            # Get dimensions from sample image
            with Image.open(sample_path) as img:
                box_width, box_height = img.size
            
            # Create large image for 6x2 grid
            grid_width = box_width * 6  # 6 columns
            grid_height = box_height * 2  # 2 rows
            grid_image = Image.new('RGB', (grid_width, grid_height), 'white')
            
            # Process each position (1-6)
            for j in range(1, 7):
                png_filename = f"BP{i}_{side}{j}.png"
                svg_filename = f"BP{i}_{side}{j}.svg"
                png_filepath = os.path.join(image_dir, png_filename)
                svg_filepath = os.path.join(image_dir, svg_filename)
                
                if os.path.exists(png_filepath) and os.path.exists(svg_filepath):
                    try:
                        # Open PNG image
                        png_img = Image.open(png_filepath)
                        
                        # Convert SVG to PNG
                        svg_png = cairosvg.svg2png(url=svg_filepath,
                                                 output_width=box_width,
                                                 output_height=box_height)
                        svg_img = Image.open(io.BytesIO(svg_png))
                        
                        # Calculate positions
                        png_pos = ((j-1) * box_width, 0)  # Top row
                        svg_pos = ((j-1) * box_width, box_height)  # Bottom row
                        
                        # Paste images
                        grid_image.paste(png_img, png_pos)
                        grid_image.paste(svg_img, svg_pos)
                        
                    except Exception as e:
                        print(f"Error processing BP{i}_{side}{j}: {e}")
                        all_files_exist = False
                else:
                    all_files_exist = False
                    if not os.path.exists(png_filepath):
                        print(f"Warning: PNG file not found: {png_filepath}")
                    if not os.path.exists(svg_filepath):
                        print(f"Warning: SVG file not found: {svg_filepath}")
            
            if all_files_exist:
                # Save the grid image
                output_filename = f"BP{i}_{side}_grid.png"
                grid_image.save(os.path.join(output_dir, output_filename))
                print(f"Created grid for BP{i}_{side}")

if __name__ == "__main__":
    create_grid_comparison("boxes")  # Replace with your image directory