import cv2
import numpy as np
import os
import random
from scipy.ndimage import rotate

def resize_with_aspect_ratio(image, size=(224, 224)):
    """
    Resize image while maintaining aspect ratio and padding to fit the target size.
    """
    h, w = image.shape[:2]
    target_h, target_w = size

    # Calculate scaling factor and resize
    scale = min(target_w / w, target_h / h)
    resized_w, resized_h = int(w * scale), int(h * scale)
    resized_image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    # Create padded image
    delta_w = target_w - resized_w
    delta_h = target_h - resized_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_image

def augment_image(image, output_dir, base_filename):
    """
    Generate augmented images by rotating the input image with random step sizes between 15 and 30 degrees.
    """
    angle = 0
    while angle < 360:
        rotated_image = rotate(image, angle, reshape=False, order=0, mode='nearest')  # Nearest neighbor interpolation
        resized_image = resize_with_aspect_ratio(rotated_image, size=(224, 224))
        filename = os.path.join(output_dir, f'{base_filename}_angle_{angle}.jpg')
        cv2.imwrite(filename, resized_image)
        angle += random.randint(15, 30)

def process_folder(input_dir, output_dir):
    """
    Process all images in a folder and perform augmentation.
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(input_dir, filename)
            image = cv2.imread(filepath)
            if image is None:
                print(f"Error reading {filename}, skipping.")
                continue

            base_filename = os.path.splitext(filename)[0]

            # Perform augmentation
            augment_image(image, output_dir, base_filename)
            print(f"Processed and augmented {filename}")


if __name__ == "__main__":
    input_dir = 'dents'  # Replace with your input folder path
    output_dir = 'dents_augmented'  # Replace with your output folder path

    process_folder(input_dir, output_dir)
