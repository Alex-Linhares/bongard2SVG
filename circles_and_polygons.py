import cv2
import numpy as np

class ImagePreprocessor:
    def __init__(self):
        self.binary_image = None
        
    def load_and_binarize(self, image_path):
        """
        Loads a binary PNG image.
        
        Args:
            image_path (str): Path to the PNG file
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Load the image - since it's already binary, we just need to load it
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Could not open or find the image at {image_path}")
            return False
            
        self.binary_image = img
        
        # Optional: Show the loaded image for debugging
        cv2.imshow("Loaded Binary Image", self.binary_image)
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
    img = cv2.cvtColor(preprocessor.binary_image.copy(), cv2.COLOR_GRAY2BGR)  # Convert to BGR for colored drawings
    binary_img = preprocessor.binary_image

    # Detect circles with adjusted parameters for binary images
    circles = cv2.HoughCircles(
        binary_img, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=30,  # Minimum distance between circles
        param1=50,   # Upper threshold for edge detection
        param2=30,   # Threshold for center detection
        minRadius=10,
        maxRadius=100
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)  # Draw the circle

    # Detect polygons
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Filter out very small contours
        if cv2.contourArea(contour) < 100:  # Adjust this threshold as needed
            continue
            
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        if len(approx) == 3:
            cv2.drawContours(img, [approx], 0, (0, 0, 255), 2)  # Triangle
        elif len(approx) == 4:
            cv2.drawContours(img, [approx], 0, (255, 0, 0), 2)  # Quadrilateral
        elif len(approx) == 5:
            cv2.drawContours(img, [approx], 0, (255, 255, 0), 2)  # pentagon
        else:
            cv2.drawContours(img, [approx], 0, (255, 0, 255), 2)  # Other polygons or circles(if hough fails)

    # Show results
    cv2.imshow("Detected Shapes", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage:
if __name__ == "__main__":
    image_path = "data/png/binary_Bp79-81.png"  # Updated path to use PNG files
    detect_shapes(image_path)