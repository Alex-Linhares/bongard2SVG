from PIL import Image, ImageDraw
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.utils import ImageReader
import io

def create_comparison_pdf(grid_dir="comparisons", output_file="bongard_comparisons.pdf"):
    """Creates a PDF with all grid comparison images from BP1 to BP100, with black squares around each top box and two sets of rows per page."""
    
    # Get all grid images and sort them
    grid_files = []
    for i in range(1, 101):
        for side in ['L', 'R']:
            filename = f"BP{i}_{side}_grid.png"
            filepath = os.path.join(grid_dir, filename)
            if os.path.exists(filepath):
                grid_files.append((i, side, filepath))
    
    if not grid_files:
        print("No grid images found!")
        return
        
    # Sort by BP number and side
    grid_files.sort(key=lambda x: (x[0], x[1]))
    
    # Create PDF
    c = canvas.Canvas(output_file, pagesize=landscape(letter))
    page_width, page_height = landscape(letter)
    
    # Get dimensions of first image to calculate scaling
    with Image.open(grid_files[0][2]) as img:
        img_width, img_height = img.size
    
    # Calculate maximum size that maintains aspect ratio
    margin = 50  # points (1/72 inch)
    max_width = page_width - 2 * margin
    max_height = (page_height - 3 * margin) / 2  # Two sets of rows per page
    scale = min(max_width / img_width, max_height / img_height)
    
    scaled_width = img_width * scale
    scaled_height = img_height * scale
    
    # Calculate centered position for two rows
    x = (page_width - scaled_width) / 2
    y_top = page_height - margin - scaled_height
    y_bottom = y_top - scaled_height - margin
    
    # Add each image to PDF
    for index, (bp_num, side, filepath) in enumerate(grid_files):
        try:
            # Add BP number and side as title
            c.setFont("Helvetica-Bold", 24)
            title = f"Bongard Problem {bp_num}"
            y_title = y_top + scaled_height + margin / 2
            c.drawString(margin, y_title, title)
            
            c.setFont("Helvetica", 14)
            side_title = f"{side} side"
            c.drawString(margin, y_title - 20, side_title)
            
            # Convert PIL Image to format reportlab can use
            with Image.open(filepath) as img:
                # Add black square around each top box
                draw = ImageDraw.Draw(img)
                for j in range(6):
                    top_left = (j * img_width // 6, 0)
                    bottom_right = ((j + 1) * img_width // 6, img_height // 2)
                    draw.rectangle([top_left, bottom_right], outline="black", width=3)
                
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    
                # Save to bytes buffer
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                # Draw the image
                y_position = y_top if index % 2 == 0 else y_bottom
                c.drawImage(ImageReader(img_buffer),
                          x, y_position,
                          width=scaled_width,
                          height=scaled_height)
            
            # Start new page after every two images
            if index % 2 == 1:
                c.showPage()
            print(f"Added {filepath} to PDF")
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    # Save PDF
    try:
        c.save()
        print(f"\nPDF saved as {output_file}")
    except Exception as e:
        print(f"Error saving PDF: {e}")

if __name__ == "__main__":
    create_comparison_pdf() 