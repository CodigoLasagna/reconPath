from gesture_detection import HandGestureDetector
from gestKnn_module import HandGestureClassifierKnn
import os

def check_file_exists(filepath):
    return os.path.isfile(filepath)

def initialize_classifier(dataset_path, model_path_to_use, model_path_to_train):
    classifier = HandGestureClassifierKnn(dataset_path, model_path_to_use, model_path_to_train)
    return classifier

if __name__ == "__main__":
    dataset_path = 'hand_gesture_dataset/labels.csv'
    model_use_path = 'trained_models/model_01.pkl'
    model_train_path = 'trained_models/model_01.pkl'
    classifier = initialize_classifier(dataset_path, model_use_path, model_train_path)
    detector = HandGestureDetector(max_num_hands=2, min_detection_confidence=0.7, auto_word='', classifier=classifier)

    detector.detect_gestures()
