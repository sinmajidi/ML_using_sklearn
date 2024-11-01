import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# Function to draw the keyboard and return key rectangles
def draw_keyboard(img):
    # Define keyboard keys and their positions (10 keys)
    keys = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    key_rects = []

    # Draw the 10 keys as rectangles on the bottom of the screen
    for i, key in enumerate(keys):
        x = 50 + i * 100
        y = 0
        w = 80
        h = 80
        key_rects.append((x, y, x + w, y + h))
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), cv2.FILLED)  # Filled blue keys
        cv2.putText(img, key, (x + 20, y + 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)  # White text

    return img, key_rects, keys

def main():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)  # Set width
    cap.set(4, 720)   # Set heigh

    # Initialize the hand detector
    detector = HandDetector(staticMode=False, maxHands=2, detectionCon=0.7, minTrackCon=0.7)

    # List to store pressed keys
    pressed_keys = []

    while True:
        # Capture each frame from the webcam
        success, img = cap.read()

        if not success:
            print("Failed to capture image from camera.")
            break

        # Find hands in the current frame
        hands, img = detector.findHands(img, draw=True, flipType=True)

        # Draw the virtual keyboard
        img, key_rects, keys = draw_keyboard(img)

        # Check if any hands are detected
        if hands:
            # We are only interested in the right hand, so find it
            for hand in hands:
                if hand["type"] == "Right":
                    lmList = hand['lmList']  # List of 21 landmarks for the hand

                    # The index finger tip landmark is number 8
                    index_finger_tip = lmList[8]

                    # Get the coordinates of the index finger
                    ix, iy = index_finger_tip[0], index_finger_tip[1]

                    # Draw a circle on the tip of the index finger
                    cv2.circle(img, (ix, iy), 15, (0, 255, 0), cv2.FILLED)

                    # Check if the index finger is touching any of the keys
                    for rect, key in zip(key_rects, keys):
                        x1, y1, x2, y2 = rect

                        # If the finger is within the key rectangle
                        if x1 < ix < x2 and y1 < iy < y2:
                            # Add key to pressed_keys if not already added (to avoid repeated prints)
                            if key not in pressed_keys:
                                print(f"Key Pressed: {key}")
                                pressed_keys.append(key)

                            # Change the key color to indicate it was pressed
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), cv2.FILLED)
                            cv2.putText(img, key, (x1 + 20, y1 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)

        # Display the image
        cv2.imshow("Virtual Keyboard", img)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
