#from gesture_detection import HandGestureDetector
from gestKnn_module import HandGestureClassifier

#if __name__ == "__main__":
#    detector = HandGestureDetector(max_num_hands=2, min_detection_confidence=0.9, auto_word='continua')
#    detector.detect_gestures()

if __name__ == "__main__":
    classifier = HandGestureClassifier('hand_gesture_dataset/labels.csv')
    classifier.train()
    classifier.evaluate()
