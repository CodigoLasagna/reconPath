from gesture_detection import HandGestureDetector

if __name__ == "__main__":
    detector = HandGestureDetector(max_num_hands=2, min_detection_confidence=0.9)
    detector.detect_gestures()
