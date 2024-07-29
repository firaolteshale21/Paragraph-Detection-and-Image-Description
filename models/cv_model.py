import cv2
import numpy as np
from matplotlib import pyplot as plt

def read_and_display_image(image_path):
    image = cv2.imread(image_path)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()  # Display the image
    return image

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def detect_edges_and_contours(blurred):
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_bounding_boxes_and_crop(image, contours, color=(0, 255, 0), thickness=2):
    cropped_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness)
        cropped_image = image[y:y+h, x:x+w]
        cropped_images.append(cropped_image)
    return image, cropped_images

def enhance_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image

if __name__ == "__main__":
    image_path = 'path_to_your_sample_image.jpg'
    image = read_and_display_image(image_path)
    
    blurred_image = preprocess_image(image)
    
    contours = detect_edges_and_contours(blurred_image)
    
    color = (0, 0, 255)  # Red color for bounding boxes
    thickness = 3  # Thickness of the bounding box
    boxed_image, cropped_images = draw_bounding_boxes_and_crop(image, contours, color, thickness)
    
    enhanced_image = enhance_image(boxed_image)
    
    # Display the final processed image with bounding boxes
    plt.imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
    plt.show()
    
    # Save the final processed image
    cv2.imwrite('processed_image_with_boxes.jpg', enhanced_image)
    
    # Optionally, display and save each cropped image
    for idx, cropped_image in enumerate(cropped_images):
        plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        plt.show()
        cv2.imwrite(f'cropped_image_{idx}.jpg', cropped_image)
