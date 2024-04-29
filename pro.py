import cv2
import numpy as np

# Define HSV ranges for colors
color_ranges = {
    'Violet': ([125, 50, 50], [150, 255, 255]),
    'Indigo': ([100, 100, 50], [125, 255, 255]),
    'Blue': ([90, 100, 50], [120, 255, 255]),
    'Green': ([50, 100, 50], [80, 255, 255]),
    'Yellow': ([20, 100, 100], [40, 255, 255]),
    'Orange': ([10, 100, 100], [20, 255, 255]),
    'Red': ([0, 100, 100], [10, 255, 255]),
    'Black': ([0, 0, 0], [180, 255, 30])  # Added range for black color
}

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Create a kernel for morphological operations
kernel = np.ones((5,5), np.uint8)

while True:
    ret, frame = cap.read()  # Read a frame from the video stream

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Iterate through color ranges
    for color_name, (lower, upper) in color_ranges.items():
        # Create mask for the current color range
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

        # Apply morphological operations to the mask
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Iterate through detected contours
        for contour in contours:
            # Calculate area of contour
            area = cv2.contourArea(contour)
            # Filter out small contours (noise)
            if area > 100:
                # Get the coordinates and dimensions of a bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                # Draw a rectangle around the detected object
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Get the centroid (center) of the bounding rectangle
                centroid_x = x + w // 2
                centroid_y = y + h // 2
                # Draw a circle at the centroid
                cv2.circle(frame, (centroid_x, centroid_y), 4, (255, 0, 0), -1)
                # Display color name
                cv2.putText(frame, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the original frame with contours drawn
    cv2.imshow('Color Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
