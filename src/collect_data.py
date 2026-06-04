"""
Dataset collection module for gesture recognition.

Responsibilities:
- Capture hand landmark data
- Associate landmarks with gesture labels
- Save structured data into CSV format
- Manage dataset generation workflow

Current Dataset Format:
- label
- timestamp
- hand index
- landmark coordinates

This module creates the training dataset
for gesture classification models.

Future Improvements:
- Automatic labeling modes
- Sequence recording
- Video recording support
- Dataset balancing tools
"""

import csv
import os
from datetime import datetime

import cv2
from hand_tracking import HandTracker

CSV_PATH = os.path.join("data", "landmarks", "hand_landmarks.csv")

def create_csv_if_needed():
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

    if os.path.exists(CSV_PATH):
        return
    
    header = ["label", "timestamp", "hand_index"]

    for i in range(21):
        header.extend([
            f"x{i}",
            f"y{i}",
            f"z{i}",
        ])
    
    with open(CSV_PATH, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)

def save_landmarks(label, landmarks, hand_index):
    timestamp = datetime.now().isoformat(timespec="seconds")

    row = [label, timestamp, hand_index]

    for landmark in landmarks:
        row.extend([
            landmark[0],  # x
            landmark[1],  # y
            landmark[2],  # z
        ])
    
    with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(row)

def main():
    print("Starting data collection...")
    print("Press keys to save samples:")
    print("h = hello")
    print("y = yes")
    print("n = no")
    print("t = thank you")
    print("s = stop")
    print("q = quit")

    create_csv_if_needed()

    tracker = HandTracker()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    key_to_label = {
        ord("h"): "hello",
        ord("y"): "yes",
        ord("n"): "no",
        ord("t"): "thank you",
        ord("s"): "stop",
    }

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        frame = cv2.flip(frame, 1)

        result = tracker.process_frame(frame)
        frame = tracker.draw_landmarks(frame, result)

        cv2.imshow("Data Collection - Signe Glasses AI", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Quitting data collection...")
            break
        if key in key_to_label:
            label = key_to_label[key]
            all_hands = tracker.extract_landmarks(result)

            if not all_hands:
                print(f"No hands detected for label '{label}'. Sample not saved.")
                continue

            for hand_index, hand_landmarks in enumerate(all_hands):
                save_landmarks(label, hand_landmarks, hand_index)

            print(f"Save sample: {label}")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()