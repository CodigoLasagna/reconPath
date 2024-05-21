import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

import seaborn as sns
import matplotlib.pyplot as plt

class HandGestureClassifierKnn:
    def __init__(self, dataset_path, model_path_to_use, model_path_to_train):
        self.dataset_path = dataset_path
        self.model_to_save = model_path_to_train
        self.df = pd.read_csv(dataset_path)
        self.df['keypoints'] = self.df['keypoints'].apply(literal_eval)
        self.knn_model = joblib.load(model_path_to_use)
        
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
        report = classification_report(self.y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        plt.figure(figsize=(10, 7))
        sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='Blues')
        plt.title('Reporte de clasificación')
        plt.show()
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        
        gesture_labels = sorted(self.df['gesture_label'].unique())
        
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=gesture_labels, yticklabels=gesture_labels)
        plt.xlabel('Predicho')
        plt.ylabel('Verdadero')
        plt.title('Matriz de confusión')
        plt.show()
        
    def save_model(self):
        joblib.dump(self.knn_model, self.model_to_save)
