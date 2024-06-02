import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

class HandGestureClassifierKnn:
    def __init__(self, dataset_path, model_path_to_use, model_path_to_train):
        self.dataset_path = dataset_path
        self.model_to_save = model_path_to_train
        self.model_path_to_use = model_path_to_use

        self.load_cvs_data()

        if os.path.isfile(self.model_path_to_use):
            self.load_model()
        else:
            self.knn_model = None
            print("No se encontró el modelo. Es necesario entrenarlo.")
    def load_model(self):
        self.knn_model = joblib.load(self.model_path_to_use)

    def load_cvs_data(self):
        if os.path.isfile(self.dataset_path):
            self.df = pd.read_csv(self.dataset_path)
            self.df['keypoints_left'] = self.df['keypoints_left'].apply(literal_eval)
            self.df['keypoints_right'] = self.df['keypoints_right'].apply(literal_eval)
        else:
            self.df = None
            print("No se encontró la base de datos. No se podrá entrenar el modelo.")
            

    def extract_combined_features(self, keypoints_combined):
        features = []
        
        if 'Left' in keypoints_combined:
            keypoints_left = keypoints_combined['Left']
            x_coords_left = [point[0] for point in keypoints_left]
            y_coords_left = [point[1] for point in keypoints_left]
            mean_x_left = np.mean(x_coords_left) if x_coords_left else 0
            mean_y_left = np.mean(y_coords_left) if y_coords_left else 0
            features.extend([mean_x_left, mean_y_left])
        else:
            features.extend([0, 0])  # Si no hay datos de la mano izquierda, se añaden valores por defecto
    
        if 'Right' in keypoints_combined:
            keypoints_right = keypoints_combined['Right']
            x_coords_right = [point[0] for point in keypoints_right]
            y_coords_right = [point[1] for point in keypoints_right]
            mean_x_right = np.mean(x_coords_right) if x_coords_right else 0
            mean_y_right = np.mean(y_coords_right) if y_coords_right else 0
            features.extend([mean_x_right, mean_y_right])
        else:
            features.extend([0, 0])  # Si no hay datos de la mano derecha, se añaden valores por defecto
    
        return features

    def train(self):
        if self.df is None:
            print("No hay datos disponibles para entrenar el modelo.")
            return

        self.df['features'] = self.df.apply(lambda row: self.extract_combined_features({'Left': row['keypoints_left'], 'Right': row['keypoints_right']}), axis=1)
        X = np.array(self.df['features'].tolist())
        y = self.df['gesture_label']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=18)

        self.knn_model = KNeighborsClassifier(n_neighbors=13)
        self.knn_model.fit(self.X_train, self.y_train)

    def evaluate(self):
        if self.knn_model is None:
            print("No hay modelo entrenado para evaluar.")
            return
    
        y_pred = self.knn_model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        cr = classification_report(self.y_test, y_pred, output_dict=True)
        
        # Mostrar reporte de clasificación
        #print(classification_report(self.y_test, y_pred))
    
        # Preparar datos para el gráfico de barras
        metrics = cr.keys()
        precision = [cr[label]['precision'] for label in metrics if label != 'accuracy']
        recall = [cr[label]['recall'] for label in metrics if label != 'accuracy']
        f1score = [cr[label]['f1-score'] for label in metrics if label != 'accuracy']
        categories = [label for label in metrics if label != 'accuracy']
    
        # Configurar el gráfico de barras
        plt.figure(figsize=(10, 5), facecolor='#293942')  # Ajuste el color de fondo de la figura
        barWidth = 0.25
        ax = plt.axes()
        ax.set_facecolor('#1B232B')
        r1 = np.arange(len(categories))
        r2 = [x + barWidth for x in r1]
        r3 = [x + barWidth for x in r2]
        
        # Usar los colores de la paleta YlGnBu para las barras
        palette = sns.color_palette('YlGnBu', len(categories))
        
        plt.bar(r1, precision, color=palette[0], width=barWidth, label='Precision')
        plt.bar(r2, recall, color=palette[1], width=barWidth, label='Recall')
        plt.bar(r3, f1score, color=palette[2], width=barWidth, label='F1-score')
        
        plt.xlabel('Metricas', fontweight='bold', color='white')
        plt.ylabel('Score', color='white')  # Agregar etiqueta para el eje Y
        plt.xticks([r + barWidth for r in range(len(categories))], categories, color='white')
        plt.yticks(color='white')  # Ajustar color de los números en el eje Y
        plt.title('Reporte de clasificación', color='white')
        plt.legend()
        
        plt.savefig('figures/clas_rep.png')
        #plt.show()
        
        # Mostrar matriz de confusión
        plt.figure(figsize=(10, 7), facecolor='#293942')
        heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=np.unique(self.y_test), yticklabels=np.unique(self.y_test))
        heatmap.set_yticklabels(heatmap.get_yticklabels(), color='white')
        heatmap.set_xticklabels(heatmap.get_xticklabels(), color='white')

        cbar = heatmap.collections[0].colorbar
        cbar.set_label('Colorbar Label', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.ax.yaxis.set_ticklabels(cbar.ax.yaxis.get_ticklabels(), color='white')

        plt.xlabel('Predicho', color='white')
        plt.ylabel('Verdadero', color='white')
        plt.title('Matriz de confusión', color='white')
        plt.savefig('figures/conf_mat.png')
        #plt.show()


    def save_model(self):
        if self.knn_model is None:
            print("No hay modelo entrenado para guardar.")
            return

        joblib.dump(self.knn_model, self.model_to_save)
        print(f'Modelo guardado en {self.model_to_save}')
