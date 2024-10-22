import cv2
from cvzone.PoseModule import PoseDetector

def main():
    # Initialize the webcam (change index if needed)
    cap = cv2.VideoCapture(0)  # Use 0 for built-in webcam, or another index for external cameras
    cap.set(3, 1280)  # Set width
    cap.set(4, 720)   # Set heigh
    # Initialize the PoseDetector from cvzone
    detector = PoseDetector(staticMode=False,
                            modelComplexity=1,
                            smoothLandmarks=True,
                            enableSegmentation=False,
                            smoothSegmentation=True,
                            detectionCon=0.5,
                            trackCon=0.5)

    # Continuously read from the webcam feed
    while True:
        success, img = cap.read()  # Read the current frame from the webcam

        if not success:
            print("Error: Could not retrieve frame from webcam.")
            break

        # Find pose landmarks and draw them on the image
        img = detector.findPose(img, draw=True)

        # Get landmark positions (this line is not strictly necessary unless you need the positions)
        lmList, bboxInfo = detector.findPosition(img, draw=True, bboxWithHands=False)

        # Display the processed image with landmarks
        cv2.imshow("Pose Detection", img)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
