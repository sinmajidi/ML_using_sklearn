import cv2
from cvzone.HandTrackingModule import HandDetector

def main():
    # Initialize the webcam to capture video
    cap = cv2.VideoCapture(0)

    # Initialize the HandDetector class with the given parameters
    detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

    # Continuously get frames from the webcam
    while True:
        # Capture each frame from the webcam
        success, img = cap.read()
        
        # If frame capture was not successful, break the loop
        if not success:
            print("Failed to capture image from camera.")
            break

        # Find hands in the current frame
        hands, img = detector.findHands(img, draw=True, flipType=True)

        # Check if any hands are detected
        if hands:
            # Iterate over detected hands
            for i, hand in enumerate(hands):
                # Count the number of fingers up for the detected hand
                fingers = detector.fingersUp(hand)
                print(f'Hand {i + 1} Fingers Up: {fingers.count(1)}')  # Print the count of fingers that are up

        # Display the image in a window
        cv2.imshow("Hand Tracking", img)

        # Wait for 1 millisecond and check if the user pressed 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
