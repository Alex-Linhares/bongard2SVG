import cv2
import numpy as np

class ImagePreprocessor:
    def __init__(self):
        self.binary_image = None
        
    def load_and_binarize(self, image_path, threshold=127):
        """
        Loads an image and converts it to binary using a threshold.
        
        Args:
            image_path (str): Path to the image file
            threshold (int): Threshold value for binarization (0-255)
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Load the image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Could not open or find the image at {image_path}")
            return False
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Convert to binary using Otsu's method for automatic thresholding
        _, self.binary_image = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Optional: Show the binary image for debugging
        cv2.imshow("Binary Image", self.binary_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return True

def detect_shapes(image_path):
    """Detects circles and polygons in a binary image."""
    # Create preprocessor instance
    preprocessor = ImagePreprocessor()
    if not preprocessor.load_and_binarize(image_path):
        return

    # Use the binary image for shape detection
    img = preprocessor.binary_image.copy()
    binary_img = preprocessor.binary_image

    # Detect circles
    circles = cv2.HoughCircles(binary_img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)  # Draw the circle

    # Detect polygons
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        if len(approx) == 3:
            cv2.drawContours(img, [approx], 0, (0, 0, 255), 2)  # Triangle
        elif len(approx) == 4:
            cv2.drawContours(img, [approx], 0, (255, 0, 0), 2)  # Quadrilateral
        elif len(approx) == 5:
            cv2.drawContours(img, [approx], 0, (255, 255, 0), 2)  # pentagon
        else:
            cv2.drawContours(img, [approx], 0, (255, 0, 255), 2)  # Other polygons or circles(if hough fails)

    cv2.imshow("Detected Shapes", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage:
if __name__ == "__main__":
    image_path = "data/Bp79-81.BMP"  # Updated path to the data folder
    detect_shapes(image_path)