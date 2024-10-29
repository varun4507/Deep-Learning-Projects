import numpy as np
import cv2
import onnxruntime as ort
from picamera2 import Picamera2
import matplotlib.pyplot as plt
from PIL import Image
import time



def decode_segmap(labels, nc=21):
    """
    Decodes the segmentation map into an RGB image.
    Args:
        labels (np.array): 2D array of class indices.
        nc (int): Number of classes.
    Returns:
        np.array: RGB image.
    """
    label_colors = np.array([
        (0, 0, 0),(128, 0, 0),(0, 128, 0),(128, 128, 0),(0, 0, 128),(128, 0, 128),(0, 128, 128),
        (128, 128, 128),(64, 0, 0),(192, 0, 0),(64, 128, 0),(192, 128, 0),(64, 0, 128),(192, 0, 128),
        (64, 128, 128),(192, 128, 128),(0, 64, 0),(128, 64, 0),(0, 192, 0),(128, 192, 0),(0, 64, 128)])

    # Ensure labels are within range
    if np.max(labels) >= nc:
        raise ValueError(f"Class index out of bounds. Labels contain indices >= {nc}.")

    # Create an RGB image using advanced indexing
    rgb = label_colors[labels]

    return rgb

# Initialize Picamera2
picam2 = Picamera2()
camera_config = picam2.create_still_configuration(main={"size": (512, 512)})
picam2.configure(camera_config)
picam2.start()
time.sleep(2)  # Allow camera to warm up

# Load the ONNX model
onnx_model_path = "/home/rspi4/Downloads/onnx_models/deeplabv3_mobilenet_v3_large.onnx"
session = ort.InferenceSession(onnx_model_path)

# Preprocessing function
def preprocess_image(image):
    # Convert image to float32 and normalize
    image = image.astype(np.float32) / 255.0
    # Transpose from (H, W, C) to (C, H, W)
    image = np.transpose(image, (2, 0, 1))
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

# Apply softmax to logits
def apply_softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=0, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=0, keepdims=True)

# Main loop
while True:
    try:
        # Capture an image using Picamera2
        print("Press 's' to capture image or 'q' to quit.")
        key = input()  # Use input() to simulate waiting for a keypress

        if key == 's':
            # Capture and process the frame
            frame = picam2.capture_array()
            frame=cv2.flip(frame, -1)

            # Save and display the captured image
            filename = 'captured_image.jpg'
            cv2.imwrite(filename, frame)
            print("Image captured and saved as", filename)

            # Load the captured image
            img = Image.open(filename)
            plt.imshow(img)
            plt.show()

            # Apply transformations
            inp = preprocess_image(np.array(img))

            # Perform inference with ONNX Runtime
            ort_inputs = {session.get_inputs()[0].name: inp}
            ort_outs = session.run(None, ort_inputs)

            # Process the output
            out = ort_outs[0][0]  # [0] for the first output and [0] for the batch dimension
            print("Model output shape:", out.shape)

            # Apply softmax to get probabilities
            probs = apply_softmax(out)
            predicted = np.argmax(probs, axis=0)  # Remove the batch dimension

            # Debugging: Check unique class indices
            unique_indices = np.unique(predicted)
            print("Unique class indices:", unique_indices)

            # Decode the segmentation map
            rgb = decode_segmap(predicted)
            plt.imshow(rgb)
            plt.show()
            
            class_labels = [
                "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
                "bus", "car", "cat", "chair", "cow", "dining table", "dog", "horse",
                "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv/monitor"
            ]

            predicted_classes = [class_labels[idx] for idx in np.unique(predicted)]
            print("Predicted classes:", predicted_classes)

        elif key == 'q':
            # Quit the program
            print("Exiting the program.")
            picam2.stop()
            break

    except KeyboardInterrupt:
        print("Interrupted by user.")
        picam2.stop()
        break

    except Exception as e:
        print(f"An error occurred: {e}")
