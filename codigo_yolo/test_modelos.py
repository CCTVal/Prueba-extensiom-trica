#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 15:43:28 2024

@author: atdcctval
"""
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import random
import os

# Semilla para la reproducibilidad
seed = 42  # Establecer el valor fijo para la semilla

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # Asegurar reproducibilidad
    torch.backends.cudnn.benchmark = False


# Definir el archivo de datos
data = '/home/atdcctval/Downloads/wood/data.yaml'



# Inicializar y entrenar el modelo
model = YOLO('yolov8n-seg.pt')
results = model.train(
    data=data,
    epochs=380,
    batch=8,
    lr0=0.0015331415405656806,
    warmup_epochs=0,
    momentum=0.8838880773354888,
    weight_decay=2.6268632815839626e-05,
    box=0.1212650217269146,
    cls=0.20242680326104873,
    dfl=0.5723326549184705,
    optimizer='AdamW',
)

test_results = model.val(split='test')
test_results_dict = test_results.results_dict
print(f"Test mAP50: {test_results_dict.get('metrics/mAP50(B)', 'No mAP50 metric'):.4f}")
print(f"Test Precision: {test_results_dict.get('metrics/precision(B)', 'No Precision metric'):.4f}")
print(f"Test Recall: {test_results_dict.get('metrics/recall(B)', 'No Recall metric'):.4f}")
print(f"Test mAP50-95: {test_results_dict.get('metrics/mAP50-95(B)', 'No mAP50-95 metric'):.4f}")
print(f"Test Fitness: {test_results_dict.get('fitness', 'No Fitness metric'):.4f}")

model.save('/home/atdcctval/Downloads/wood3.pt')



