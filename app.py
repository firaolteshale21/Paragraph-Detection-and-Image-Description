import cv2
from models.cv_model import read_and_display_image, preprocess_image, detect_edges_and_contours, draw_bounding_boxes_and_crop, enhance_image
from models.nlp_model import generate_caption
import os

def process_image(image_path):
    image = read_and_display_image(image_path)
    blurred = preprocess_image(image)
    contours = detect_edges_and_contours(blurred)
    
    color = (0, 0, 255)  # Red color for bounding boxes
    thickness = 1  # Thickness of the bounding box
    boxed_image, cropped_images = draw_bounding_boxes_and_crop(image, contours, color, thickness)
    
    # Save the processed image with bounding boxes
    cv2.imwrite('data/images/processed/boxed/boxed_image.jpg', boxed_image)
    
    # Debug: Print the number of cropped images
    print(f"Number of cropped images: {len(cropped_images)}")
    
    # Save and process each cropped image
    for idx, cropped in enumerate(cropped_images):
        # Debug: Print the index of the current cropped image
        print(f"Processing cropped image {idx}")
        
        enhanced = enhance_image(cropped)
        cropped_image_path = f'data/images/processed/cropped_images/cropped_image_{idx}.jpg'
        cv2.imwrite(cropped_image_path, enhanced)
        caption = generate_caption(cropped_image_path)
        print(f"Caption for image {idx}: {caption}")

if __name__ == "__main__":
    image_path = 'data/images/sample_image.jpg'
    if os.path.exists(image_path):
        process_image(image_path)
    else:
        print(f"Image not found: {image_path}")
