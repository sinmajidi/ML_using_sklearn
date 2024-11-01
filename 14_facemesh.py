import cv2
from cvzone.FaceMeshModule import FaceMeshDetector

def main():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)  # Set width
    cap.set(4, 720)   # Set heigh

    # Initialize the FaceMeshDetector with the desired parameters
    detector = FaceMeshDetector(staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5)

    # Start the loop to continually get frames from the webcam
    while True:
        # Read the current frame from the webcam
        success, img = cap.read()

        # If the frame was not captured successfully, break the loop
        if not success:
            print("Failed to capture image from camera.")
            break

        # Find face mesh in the image
        img, faces = detector.findFaceMesh(img, draw=True)

        # Check if any faces are detected
        if faces:
            # Loop through each detected face
            for face in faces:  
                # print(face)
                pass

        # Display the image in a window named 'Face Mesh'
        cv2.imshow("Face Mesh", img)

        # Wait for 1 millisecond to check for any user input; if 'q' is pressed, break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
