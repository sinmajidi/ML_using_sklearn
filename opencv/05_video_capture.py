import cv2

# Open the default webcam (usually, ID=0)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Read video frames in a loop
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # If frame is read correctly ret is True
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    # Display the resulting frame
    cv2.imshow('Webcam Video', frame)
    
    # Wait for 1 ms, if the user presses 'q', break out of the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()
