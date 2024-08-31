# Import the necessary libraries
import cv2  # OpenCV library for video capture and image processing
import pixellib  # PixelLib library for image segmentation
from pixellib.instance import instance_segmentation  # Import the instance segmentation class

# Create an instance of the segmentation model
segment_image = instance_segmentation()

# Load the pre-trained model (Mask R-CNN) for object detection and segmentation
segment_image.load_model("mask_rcnn_coco.h5")

# Open the camera for video capture (0 indicates the default camera)
camera = cv2.VideoCapture(0)

# Start a loop to capture frames from the camera continuously
while camera.isOpened():
    # Read the frame from the camera
    res, frame = camera.read()
    
    # Check if the frame was captured successfully
    if not res:
        break
    
    # Apply instance segmentation on the captured frame
    # 'segmentFrame' returns a dictionary with segmentation details and the processed image
    result = segment_image.segmentFrame(frame, show_bboxes=True)
    
    # Extract the segmented image from the result
    image = result[1]
    
    # Display the segmented image in a window named 'Image Segmentation'
    cv2.imshow('Image Segmentation', image)

    # Wait for 10 milliseconds for a key press
    # If the 'q' key is pressed, exit the loop
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the camera resource once the loop ends
camera.release()

# Close all OpenCV windows that were opened
cv2.destroyAllWindows()
