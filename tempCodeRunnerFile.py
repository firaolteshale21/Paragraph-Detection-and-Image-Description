def process_image(image_path):
    image = read_and_display_image(image_path)
    blurred = preprocess_image(image)
    contours = detect_edges_and_contours(blurred)
    boxed_image, cropped_images = draw_bounding_boxes_and_crop(image, contours)
    
    # Save the processed image with bounding boxes
    cv2.imwrite('data/images/boxed_image.jpg', boxed_image)
    
    # Save and process each cropped image
    for idx, cropped in enumerate(cropped_images):
        enhanced = enhance_image(cropped)
        cropped_image_path = f'data/images/cropped_image_{idx}.jpg'
        cv2.imwrite(cropped_image_path, enhanced)
        caption = generate_caption(cropped_image_path)
        print(f"Caption for image {idx}: {caption}")