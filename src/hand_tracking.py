from pathlib import Path
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandTracker:
    def __init__(self):
        project_root = Path(__file__).resolve().parent.parent
        model_path = project_root / "models" / "mediapipe" / "hand_landmarker.task"

        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe hand model not found: {model_path}\n"
                "Download the HandLandmarker .task model and place it at this path."
            )

        model_bytes = model_path.read_bytes()
        if not model_bytes:
            raise ValueError(
                f"MediaPipe hand model is empty: {model_path}\n"
                "Replace it with the real hand_landmarker.task file."
            )

        base_options = python.BaseOptions(model_asset_buffer=model_bytes)

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.detector = vision.HandLandmarker.create_from_options(options)

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )
        return self.detector.detect(mp_image)

    def draw_landmarks(self, frame, result):
        height, width, _ = frame.shape

        if not result.hand_landmarks:
            return frame

        for hand_landmarks in result.hand_landmarks:
            for landmark in hand_landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        return frame
    
    def extract_landmarks(self, result):
        """
        Extract hand landmarks as numeric data.
        Returns:
            A list of hands.
            Each hand is a list of 21 landmarks.
            Each landmark is [x,y,z]
        """
        all_hands = []

        if not result.hand_landmarks:
            return all_hands
        
        for hand_landmarks in result.hand_landmarks:
            single_hand =[]

            for landmark in hand_landmarks:

                single_hand.append([landmark.x, landmark.y, landmark.z])

            all_hands.append(single_hand)
        return all_hands
