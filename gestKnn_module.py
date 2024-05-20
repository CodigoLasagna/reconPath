import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

class HandGestureClassifier:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.df = pd.read_csv(dataset_path)
        self.df['keypoints'] = self.df['keypoints'].apply(literal_eval)
        
    def extract_features(self, row):
        keypoints = row['keypoints']
        hand_type = row['hand_type']
        hand_type_numeric = 1 if hand_type == 'right' else 0
        
        x_coords = [point[0] for point in keypoints]
        y_coords = [point[1] for point in keypoints]
        mean_x = np.mean(x_coords)
        mean_y = np.mean(y_coords)
        
        return [mean_x, mean_y, hand_type_numeric]
    
    def train(self):
        self.df['features'] = self.df.apply(self.extract_features, axis=1)
        X = np.array(self.df['features'].tolist())
        y = self.df['gesture_label']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=18)
        
        self.knn_model = KNeighborsClassifier(n_neighbors=3)
        self.knn_model.fit(self.X_train, self.y_train)
        
    def evaluate(self):
        y_pred = self.knn_model.predict(self.X_test)
        
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))

## Uso del clasificador
#if __name__ == "__main__":
#    classifier = HandGestureClassifier('hand_gesture_dataset/labels.csv')
#    classifier.train()
#    classifier.evaluate()
