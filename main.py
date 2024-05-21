from gesture_detection import HandGestureDetector
from gestKnn_module import HandGestureClassifierKnn

if __name__ == "__main__":
    classifier = HandGestureClassifierKnn('hand_gesture_dataset/labels.csv', 'model.pkl', 'model_2.pkl')
    detector = HandGestureDetector(max_num_hands=2, min_detection_confidence=0.7, auto_word='', classifier=classifier)

    detector.detect_gestures()
