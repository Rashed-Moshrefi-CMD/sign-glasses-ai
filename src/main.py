import cv2
import os
from datetime import datetime
from hand_tracking import HandTracker


# This script tests the camera functionality by opening the webcam 
# feed and allowing the user to save snapshots.
def main():
    # Check if OpenCV is installed
    print("Starting camera...")
    # Try to open the default camera (index 0)
    cap = cv2.VideoCapture(0)
    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Camera opened successfully.")
    print("Press 'q' to quit.")
    print("Press 's' to save a snapshot.")

    # Initialize the hand tracker
    tracker = HandTracker()

    # Create a directory for screenshots if it doesn't exist
    screenshot_dir = os.path.join("data", "screenshots")
    os.makedirs(screenshot_dir, exist_ok=True)
    # Start the camera feed loop
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        # Check if the frame was read successfully
        if not ret:
            print("Error: Could not read frame.")
            break

        frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a mirror effect

        results = tracker.process_frame(frame)
        frame = tracker.draw_landmarks(frame, results)

        landmarks = tracker.extract_landmarks(results)

        if landmarks:
            print("Hand landmarks detected:")
            print(landmarks[0][0])
        # Display the frame in a window
        cv2.imshow("Sign Glasses AI - Camera Test", frame)
        # Wait for a key press and check if it's 'q' or 's'
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Quitting...")
            break

        elif key == ord("s"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_{timestamp}.png"
            filepath = os.path.join(screenshot_dir, filename)

            success = cv2.imwrite(filepath, frame)

            if success:
                print(f"Snapshot saved in: {filepath}")
            else:
                print("Error: Snapshot could not be saved.")
    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Run the main function when the script is executed
if __name__ == "__main__":
    main()