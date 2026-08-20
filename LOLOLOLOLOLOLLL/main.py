"""LoL my First Repo: webcam expression and hand-gesture controlled meme viewer.

Install the pinned dependencies from requirements.txt with Python 3.12, then run:
    python main.py
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

if not hasattr(mp, "solutions"):
    raise RuntimeError(
        "LoL my First Repo requires the legacy MediaPipe Solutions API. "
        "Activate the Python 3.12 virtual environment and install "
        "mediapipe==0.10.21 from requirements.txt."
    )

PROJECT_DIR = Path(__file__).resolve().parent
ASSET_DIR = PROJECT_DIR / "assets" / "new"

GESTURE_LOVE = "love"
GESTURE_GTFO = "gtfo"
GESTURE_SIXTY_SEVEN = "sixty_seven"
GESTURE_BLEP = "blep"
GESTURE_SMILE = "smile"
GESTURE_THINKING = "thinking"
GESTURE_THUMBS_UP = "thumbs_up"
GESTURE_TIMEOUT = "timeout"
GESTURE_DEFAULT = "default"

MEME_FILES = {
    GESTURE_LOVE: "love.jpg",
    GESTURE_GTFO: "gtfo.jpg",
    GESTURE_SIXTY_SEVEN: "sixty_seven.jpg",
    GESTURE_BLEP: "blep.jpg",
    GESTURE_SMILE: "109fb257daabe2f3db63bd7bc1944934.jpg",
    GESTURE_THINKING: "maxresdefault.jpg",
    GESTURE_THUMBS_UP: "7dc6efb0fe7548ae00dd6143e739f630.jpg",
    GESTURE_TIMEOUT: "bc3d38ffc8a2e9a574bb54d3bffa5445.jpg",
    GESTURE_DEFAULT: "maxresdefault.jpg",
}

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def point(landmarks, index: int) -> np.ndarray:
    landmark = landmarks[index]
    return np.array([landmark.x, landmark.y, landmark.z], dtype=np.float32)


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


def face_scale(landmarks) -> float:
    return max(distance(point(landmarks, 33), point(landmarks, 263)), 1e-5)


def detect_big_smile(face_landmarks) -> bool:
    landmarks = face_landmarks.landmark
    scale = face_scale(landmarks)
    mouth_width = distance(point(landmarks, 61), point(landmarks, 291)) / scale
    mouth_open = distance(point(landmarks, 13), point(landmarks, 14)) / scale
    return mouth_width > 0.55 and mouth_open > 0.16


def detect_looking_up(face_landmarks) -> bool:
    landmarks = face_landmarks.landmark
    scale = max(distance(point(landmarks, 10), point(landmarks, 152)), 1e-5)
    nose = point(landmarks, 1)
    eye_center = (point(landmarks, 159) + point(landmarks, 386)) / 2.0
    nose_to_eye = (nose[1] - eye_center[1]) / scale
    return nose_to_eye < 0.105


def detect_blep(face_landmarks) -> bool:
    """Approximate blep with an open mouth and strongly squinted eyes.

    FaceMesh does not provide a dedicated tongue segmentation mask, so the
    visible-tongue cue is approximated from the mouth opening and eye squint.
    """
    landmarks = face_landmarks.landmark
    scale = face_scale(landmarks)
    mouth_open = distance(point(landmarks, 13), point(landmarks, 14)) / scale
    mouth_width = distance(point(landmarks, 61), point(landmarks, 291)) / scale
    left_eye_open = distance(point(landmarks, 159), point(landmarks, 145)) / max(distance(point(landmarks, 33), point(landmarks, 133)), 1e-5)
    right_eye_open = distance(point(landmarks, 386), point(landmarks, 374)) / max(distance(point(landmarks, 362), point(landmarks, 263)), 1e-5)
    squinted = left_eye_open < 0.24 and right_eye_open < 0.24
    return mouth_open > 0.23 and mouth_width > 0.45 and squinted


def finger_is_up(hand, tip_index: int, pip_index: int) -> bool:
    return hand.landmark[tip_index].y < hand.landmark[pip_index].y


def thumb_is_up(hand) -> bool:
    landmarks = hand.landmark
    return landmarks[4].y < landmarks[3].y and distance(point(landmarks, 4), point(landmarks, 5)) > distance(point(landmarks, 3), point(landmarks, 5))


def hand_finger_state(hand) -> tuple[bool, bool, bool, bool, bool]:
    return (
        thumb_is_up(hand),
        finger_is_up(hand, 8, 6),
        finger_is_up(hand, 12, 10),
        finger_is_up(hand, 16, 14),
        finger_is_up(hand, 20, 18),
    )


def detect_peace_sign(hand) -> bool:
    thumb, index, middle, ring, pinky = hand_finger_state(hand)
    return index and middle and not ring and not pinky and not thumb


def detect_thumbs_up(hand) -> bool:
    thumb, index, middle, ring, pinky = hand_finger_state(hand)
    return thumb and not index and not middle and not ring and not pinky


def detect_middle_finger(hand) -> bool:
    thumb, index, middle, ring, pinky = hand_finger_state(hand)
    return middle and not index and not ring and not pinky and not thumb


def detect_heart_gesture(hands: list) -> bool:
    """Detect two hands making a heart outline with index fingers and thumbs."""
    if len(hands) != 2:
        return False
    states = [hand_finger_state(hand) for hand in hands]
    for thumb_a, index_a, middle_a, ring_a, pinky_a in states:
        if not index_a or not thumb_a or middle_a or ring_a or pinky_a:
            return False
    first, second = hands
    index_tips_close = distance(point(first.landmark, 8), point(second.landmark, 8)) < 0.16
    thumb_tips_close = distance(point(first.landmark, 4), point(second.landmark, 4)) < 0.16
    return index_tips_close and thumb_tips_close


def hand_is_vertical(hand) -> bool:
    wrist = point(hand.landmark, 0)
    middle_mcp = point(hand.landmark, 9)
    return abs(middle_mcp[1] - wrist[1]) > abs(middle_mcp[0] - wrist[0]) * 1.25


def hand_is_horizontal(hand) -> bool:
    wrist = point(hand.landmark, 0)
    middle_mcp = point(hand.landmark, 9)
    return abs(middle_mcp[0] - wrist[0]) > abs(middle_mcp[1] - wrist[1]) * 1.25


def detect_timeout_gesture(hands: list) -> bool:
    if len(hands) != 2:
        return False
    for stem in hands:
        other = hands[1] if stem is hands[0] else hands[0]
        if not hand_is_vertical(stem) or not hand_is_horizontal(other):
            continue
        stem_tip = point(stem.landmark, 8)
        palm_center = (point(other.landmark, 0) + point(other.landmark, 5) + point(other.landmark, 17)) / 3.0
        if distance(stem_tip, palm_center) < 0.18:
            return True
    return False


def detect_two_hands_up(hands: list) -> bool:
    """Detect both hands raised with several fingers extended."""
    if len(hands) != 2:
        return False
    for hand in hands:
        _, index, middle, ring, pinky = hand_finger_state(hand)
        extended_fingers = sum((index, middle, ring, pinky))
        wrist_is_raised = hand.landmark[0].y < 0.72
        if extended_fingers < 3 or not wrist_is_raised:
            return False
    return True


def detect_sixty_seven_motion(hands: list, motion_history: deque[float]) -> bool:
    """Legacy helper retained for compatibility; 67 now uses two raised hands."""
    if len(hands) != 1:
        motion_history.clear()
        return False
    hand = hands[0]
    _, index, middle, ring, pinky = hand_finger_state(hand)
    # An open/relaxed palm is more likely than a closed fist. Allow the thumb
    # to be either state because webcam orientation can change its y ordering.
    if not (index or middle or ring or pinky):
        motion_history.clear()
        return False
    wrist_y = float(hand.landmark[0].y)
    motion_history.append(wrist_y)
    if len(motion_history) < 12:
        return False
    values = list(motion_history)
    deltas = np.diff(values)
    significant = [float(delta) for delta in deltas if abs(delta) > 0.008]
    if len(significant) < 5:
        return False
    signs = [1 if delta > 0 else -1 for delta in significant]
    changes = sum(signs[i] != signs[i - 1] for i in range(1, len(signs)))
    return changes >= 3 and (max(values) - min(values)) > 0.07


def classify_frame(face_results, hand_results, motion_history: deque[float]) -> str:
    hands = list(hand_results.multi_hand_landmarks or [])
    if detect_timeout_gesture(hands):
        return GESTURE_TIMEOUT
    if detect_middle_finger(hands[0]) if len(hands) == 1 else any(detect_middle_finger(hand) for hand in hands):
        return GESTURE_GTFO
    if detect_two_hands_up(hands):
        motion_history.clear()
        return GESTURE_SIXTY_SEVEN
    if any(detect_peace_sign(hand) for hand in hands):
        return GESTURE_LOVE
    if any(detect_thumbs_up(hand) for hand in hands):
        return GESTURE_THUMBS_UP
    if face_results.multi_face_landmarks:
        face = face_results.multi_face_landmarks[0]
        if detect_blep(face):
            return GESTURE_BLEP
        if detect_big_smile(face):
            return GESTURE_SMILE
        if detect_looking_up(face):
            return GESTURE_THINKING
    motion_history.clear()
    return GESTURE_DEFAULT


def load_meme_images() -> dict[str, np.ndarray]:
    images: dict[str, np.ndarray] = {}
    for gesture, filename in MEME_FILES.items():
        image = cv2.imread(str(ASSET_DIR / filename))
        if image is not None:
            images[gesture] = image
    return images


def make_fallback_meme(gesture: str, size: tuple[int, int] = (640, 480)) -> np.ndarray:
    """Return a completely blank frame when a meme image is unavailable."""
    width, height = size
    return np.zeros((height, width, 3), dtype=np.uint8)


def prepare_meme(image: Optional[np.ndarray], gesture: str) -> np.ndarray:
    return image.copy() if image is not None else make_fallback_meme(gesture)


def draw_face_landmarks(frame: np.ndarray, face_results) -> None:
    if not face_results.multi_face_landmarks:
        return
    height, width = frame.shape[:2]
    for face in face_results.multi_face_landmarks:
        for landmark in face.landmark:
            x = min(max(int(landmark.x * width), 0), width - 1)
            y = min(max(int(landmark.y * height), 0), height - 1)
            cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)


def draw_hand_landmarks(frame: np.ndarray, hand_results) -> None:
    for hand in hand_results.multi_hand_landmarks or []:
        mp_drawing.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 105, 180), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(255, 105, 180), thickness=2),
        )


def main() -> None:
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open the webcam. Please run LoL my First Repo from a Windows terminal, not WSL2, and check camera permissions.")
        return

    meme_images = load_meme_images()
    history: deque[str] = deque(maxlen=7)
    motion_history: deque[float] = deque(maxlen=24)
    current_gesture = GESTURE_DEFAULT

    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            success, frame = camera.read()
            if not success:
                print("Error: Failed to read a frame from the webcam.")
                break
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = face_mesh.process(rgb_frame)
            hand_results = hands.process(rgb_frame)
            draw_face_landmarks(frame, face_results)
            draw_hand_landmarks(frame, hand_results)

            detected = classify_frame(face_results, hand_results, motion_history)
            history.append(detected)
            if len(history) == history.maxlen and len(set(history)) == 1:
                current_gesture = history[0]

            cv2.putText(frame, f"Reaction: {current_gesture}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "ESC: quit | Hold a reaction for 7 frames", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("LoL my First Repo - Webcam", frame)
            cv2.imshow("LoL my First Repo - Matched Meme", prepare_meme(meme_images.get(current_gesture), current_gesture))
            if cv2.waitKey(1) & 0xFF == 27:
                break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
