from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, STATUS_FAIL
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import traceback
import os

# Definir el espacio de búsqueda
space = {
    'epochs': hp.quniform('epochs', 25, 500, 5),
    'batch_size': hp.choice('batch_size', [8,16]),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.01)),
    'warmup_epochs': hp.choice('warmup_epochs', [0, 1, 2, 3]),
    'momentum': hp.uniform('momentum', 0.0, 0.95),
    'weight_decay': hp.loguniform('weight_decay', np.log(1e-5), np.log(1e-2)),
    'box': hp.uniform('box', 0.02, 0.2),
    'cls': hp.uniform('cls', 0.2, 0.8),
    'dfl': hp.uniform('dfl', 0.5, 1.5),
    'optimizer': hp.choice('optimizer', ['SGD', 'Adam', 'AdamW'])
}

# Definir el conjunto de datos
data = '/home/atdcctval/Downloads/wood/data.yaml'

# Función para verificar las anotaciones
def check_annotations(yolo_labels_folder):
    for filename in os.listdir(yolo_labels_folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(yolo_labels_folder, filename)
            with open(filepath, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:  # Debe haber al menos class_id, x_center, y_center, width, height
                        print(f"Error en archivo {filename}: {line}")

# Crear la función objetivo
def objective(params):
    model = YOLO('yolov8n-seg.pt')  # Cargar el modelo preentrenado
    
    print(f"Entrenando con params: {params}")
    
    try:
        # Verificar las anotaciones
        data_folder = os.path.dirname(data)
        check_annotations(data_folder)
        
        # Definir el optimizador según el seleccionado
        optimizer = params['optimizer']
        seed = int(params.get('seed', 42))  # Usar un valor predeterminado si no se proporciona
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Entrenar el modelo con los hiperparámetros actuales
        results = model.train(
            data=data,
            epochs=int(params['epochs']),
            batch=int(params['batch_size']),
            lr0=params['learning_rate'],
            warmup_epochs=int(params['warmup_epochs']),
            momentum=params['momentum'],
            weight_decay=params['weight_decay'],
            box=params['box'],
            cls=params['cls'],
            dfl=params['dfl'],
            optimizer=optimizer  # Asegúrate de que este parámetro es válido
        )
        
        # Evaluar el rendimiento en el conjunto de validación
        metrics = model.val(data=data)
        results_dict = metrics.results_dict
        
        # Obtener las métricas
        val_mAP50 = results_dict.get('metrics/mAP50(B)', np.inf)
        val_precision = results_dict.get('metrics/precision(B)', np.inf)
        val_recall = results_dict.get('metrics/recall(B)', np.inf)
        val_mAP50_95 = results_dict.get('metrics/mAP50-95(B)', np.inf)
        val_fitness = results_dict.get('fitness', np.inf)
        
        if np.isinf(val_mAP50):
            print("Valor de mAP50 es infinito, posible problema en la evaluación.")
            return {'loss': np.inf, 'status': STATUS_FAIL}
        
        # Puedes elegir la métrica que desees para la optimización. Aquí usamos mAP50 como ejemplo.
        print(f"mAP50: {val_mAP50:.4f}")
        print(f"Precision: {val_precision:.4f}")
        print(f"Recall: {val_recall:.4f}")
        print(f"mAP50-95: {val_mAP50_95:.4f}")
        print(f"Fitness: {val_fitness:.4f}")
        
        # Retornar la métrica para minimizar (en este caso, el mAP50-95 negativo)
        return {'loss': -val_mAP50_95, 'status': STATUS_OK}
    
    except Exception as e:
        print(f"Error durante el entrenamiento o la evaluación: {e}")
        traceback.print_exc()  # Imprime el traceback completo del error
        return {'loss': np.inf, 'status': STATUS_FAIL}

# Configurar Hyperopt para buscar los mejores hiperparámetros
trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=35,
    trials=trials
)

print("Mejores hiperparámetros encontrados:", best)






