import cv2
import numpy as np

def get_white_pixel_percentage_in_contour(image_path, contour, threshold=127):
    """
    Calculates the percentage of pixels within a contour in an image that are considered "white"
    (greater than or equal to the specified threshold).

    Args:
        image_path (str): The path to the PNG image file.
        contour (numpy.ndarray): The contour (from cv2.findContours).
        threshold (int, optional): The minimum pixel value to be considered "white" (0-255).
                                 Defaults to 127.

    Returns:
        float: The percentage of white pixels (0.0 to 1.0) within the contour.
               Returns 0.0 if the contour is empty or an error occurs.
    """

    try:
        # Read the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return 0.0

        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1)  # Create a mask of the contour

        # Calculate the number of pixels within the contour
        total_pixels = np.sum(mask == 255)

        if total_pixels == 0:
            print("Warning: Contour has no pixels.")
            return 0.0

        # Calculate the number of "white" pixels within the contour
        white_pixels = np.sum(image[mask == 255] >= threshold)

        return float(white_pixels) / total_pixels

    except Exception as e:
        print(f"Error processing image or contour: {e}")
        return 0.0


# --- Example Usage (Illustrative) ---
# Replace with your actual image path and contour data

# Create a sample image (replace with your image loading)
# This part is just for demonstration, you'll load your own image.
image_path = "boxes/BP1_R5.png"  # Replace with the actual path to your PNG file


# Find contours (replace with your contour detection)
# This part is also illustrative. In your application you already have the contour.
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour = contours[0]  # Assuming the star is the first contour


# Calculate white pixel percentage
white_percentage = get_white_pixel_percentage_in_contour(image_path, contour, threshold=200)
print(f"White pixel percentage: {white_percentage:.2f}")