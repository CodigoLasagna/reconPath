#from gesture_detection import HandGestureDetector
#
#if __name__ == "__main__":
#    detector = HandGestureDetector(max_num_hands=2, min_detection_confidence=0.9, auto_word='continua')
#    detector.detect_gestures()

import pandas as pd
from ast import literal_eval
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

df = pd.read_csv('hand_gesture_dataset/labels.csv')

df['keypoints'] = df['keypoints'].apply(literal_eval)

def extract_features(row):
    keypoints = row['keypoints']
    hand_type = row['hand_type']
    
    hand_type_numeric = 1 if hand_type == 'right' else 0
    
    x_coords = [point[0] for point in keypoints]
    y_coords = [point[1] for point in keypoints]
    mean_x = np.mean(x_coords)
    mean_y = np.mean(y_coords)
    
    return [mean_x, mean_y, hand_type_numeric]

df['features'] = df.apply(extract_features, axis=1)

X = np.array(df['features'].tolist())
y = df['gesture_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=18)

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)

print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred))

print("Matriz de Confusión:")
conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)
