import cv2
import numpy as np
# Function to detect squares in the image
def detect_squares(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Perform edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize a list to store square contours
    square_contours = []
    
    # Loop over the contours
    for contour in contours:
        # Approximate the contour to simplify it
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        # If the contour has 4 vertices, it's likely a square
        if len(approx) == 4:
            square_contours.append(approx)
    
    return square_contours

# Read the video clip
cap = cv2.VideoCapture('F:\\Phela Chilla\\Conveyor Belt\\AniBox.mp4')

while cap.isOpened():
    # Read frame from the video clip
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect squares in the frame
    squares = detect_squares(frame)
    
    # Draw bounding boxes around the detected squares
    for square in squares:
        cv2.drawContours(frame, [square], -1, (0, 255, 0), 3)
    
    # Display the frame
    cv2.imshow('Square Detection', frame)
    
    # Check for key press, break loop if 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    # Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()