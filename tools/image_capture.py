import cv2
import numpy as np


def center_crop(frame, size=224):
    """
    Center crops the frame to the specified size.
    """
    height, width, _ = frame.shape
    
    # Calculate cropping box
    start_x = (width - size) // 2
    start_y = (height - size) // 2
    
    # Crop the frame
    cropped_frame = frame[start_y:start_y + size, start_x:start_x + size]
    return cropped_frame


def capture_images():
    """
    Capture and save multiple center-cropped 224x224 images from webcam when Enter is pressed.
    """
    cap = cv2.VideoCapture(0)  # Open the default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    count = 0  # Image counter

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Perform center crop
            cropped_frame = center_crop(frame, size=224)

            # Display the cropped frame
            cv2.imshow('Center Cropped Frame', cropped_frame)

            # Wait for key press
            key = cv2.waitKey(1)

            # Capture image on Enter key (key code 13)
            if key == 13: 
                filename = f'captured_image_{count}.jpg'
                cv2.imwrite(filename, cropped_frame)
                print(f"Image {count} captured and saved as '{filename}'")
                count += 1

            # Exit on 'q' key
            elif key & 0xFF == ord('q'):
                break

    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_images()
