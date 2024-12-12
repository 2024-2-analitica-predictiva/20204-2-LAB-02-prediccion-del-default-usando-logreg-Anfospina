# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".

# Importacion de librerias
import pandas as pd
import os
import json
import gzip
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)


def clean_data(path):
    d_frame=pd.read_csv(path,index_col=False,compression='zip')
    
    d_frame.rename(columns={'default payment next month':'default'},inplace=True)
    d_frame.drop(columns='ID',inplace=True)
    d_frame=d_frame.loc[d_frame[(d_frame['EDUCATION']!=0) & (d_frame['MARRIAGE']!=0)].index]
    d_frame['EDUCATION']=d_frame['EDUCATION'].apply(lambda x: 4 if x>4 else x)
    return d_frame

train_data=clean_data('files/input/train_data.csv.zip')
test_data=clean_data('files/input/test_data.csv.zip')

#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.

x_train,y_train=train_data.drop(columns='default'),train_data['default']
x_test,y_test=test_data.drop(columns='default'),test_data['default']

#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
categorical_columns=['EDUCATION', 'MARRIAGE', 'SEX']
numerical_columns=[col for col in x_train.columns if col not in categorical_columns]

steps=[('preprocessor', ColumnTransformer(
        transformers=[  
            ('cat', OneHotEncoder(), categorical_columns),
            ('num',MinMaxScaler(),numerical_columns)
            ],
        remainder='passthrough'  # Mantener el resto de las columnas sin cambios
    )),
       ('feature_selection',SelectKBest(score_func=f_classif)),
       ('Logistic_Regression', LogisticRegression(random_state=42))
    ]

pipeline = Pipeline(steps)

#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.

param_grid = {
    'feature_selection__k': (1,11),
    'Logistic_Regression__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'Logistic_Regression__solver': ['liblinear'],
    'Logistic_Regression__penalty': ['l1', 'l2'],
    'Logistic_Regression__max_iter': [100, 200]
}

model = GridSearchCV(
    pipeline,  # Pipeline definido anteriormente
    param_grid=param_grid,  # Espacio de búsqueda
    cv=10,  # Número de divisiones para validación cruzada
    scoring='balanced_accuracy',  # Métrica de evaluación
    n_jobs=-1,  # Paralelizar para aprovechar múltiples núcleos
)

model.fit(x_train, y_train)

#
# Paso 5.
# Salve el modelo como "files/models/model.pkl".

models_dir = 'files/models'
os.makedirs(models_dir, exist_ok=True)

model_path = "files/models/model.pkl.gz"
with gzip.open(model_path, "wb") as file:
    pickle.dump(model, file)  

#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}

def metrics_calculate(model, x_train, x_test, y_train, y_test):
    
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    metrics = [
        {
            'type': 'metrics',
            'dataset': 'train',
            'precision': precision_score(y_train, y_train_pred, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y_train, y_train_pred),
            'recall': recall_score(y_train, y_train_pred, zero_division=0),
            'f1_score': f1_score(y_train, y_train_pred, zero_division=0)
        },
        {
            'type': 'metrics',
            'dataset': 'test',
            'precision': precision_score(y_test, y_test_pred, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
            'recall': recall_score(y_test, y_test_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_test_pred, zero_division=0)
        }
    ]

    os.makedirs("files/output", exist_ok=True)
    with open("files/output/metrics.json", "w") as f:
        for metric in metrics:
            f.write(json.dumps(metric) + '\n')

#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
def calculate_confusion_matrices(model, x_train, x_test, y_train, y_test):
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    matrices = [
        {
            'type': 'cm_matrix',
            'dataset': 'train',
            'true_0': {'predicted_0': int(cm_train[0, 0]), 'predicted_1': int(cm_train[0, 1])},
            'true_1': {'predicted_0': int(cm_train[1, 0]), 'predicted_1': int(cm_train[1, 1])}
        },
        {
            'type': 'cm_matrix',
            'dataset': 'test',
            'true_0': {'predicted_0': int(cm_test[0, 0]), 'predicted_1': int(cm_test[0, 1])},
            'true_1': {'predicted_0': int(cm_test[1, 0]), 'predicted_1': int(cm_test[1, 1])}
        }
    ]

    with open("files/output/metrics.json", "a") as f:
        for matrix in matrices:
            f.write(json.dumps(matrix) + '\n')
            
# Ejecutar cálculo de métricas y matrices
metrics_calculate(model, x_train, x_test, y_train, y_test)
calculate_confusion_matrices(model, x_train, x_test, y_train, y_test)
