import json
import os
from PIL import Image
import numpy as np
import shutil

def process_dataset(coco_json_path, images_folder, output_folder, set_name):
    # Crear carpeta de salida para las anotaciones YOLO
    labels_folder = os.path.join(output_folder, set_name, 'labels')
    os.makedirs(labels_folder, exist_ok=True)
    
    # Carpeta para las imágenes
    images_output_folder = os.path.join(output_folder, set_name, 'images')
    os.makedirs(images_output_folder, exist_ok=True)

    # Cargar el archivo JSON de COCO
    with open(coco_json_path) as f:
        coco_data = json.load(f)

    # Diccionario para guardar las categorías
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # Función para normalizar coordenadas
    def normalize_coordinates(coords, width, height):
        return [coord / width if i % 2 == 0 else coord / height for i, coord in enumerate(coords)]

    # Procesar cada imagen
    for image_info in coco_data['images']:
        image_id = image_info['id']
        image_filename = image_info['file_name']
        image_path = os.path.join(images_folder, image_filename)
        
        # Copiar la imagen al directorio de salida
        shutil.copy(image_path, images_output_folder)
        
        # Obtener dimensiones de la imagen
        image = Image.open(image_path)
        image_width, image_height = image.size
        
        # Crear archivo de anotaciones YOLO para cada imagen
        yolo_filename = os.path.splitext(image_filename)[0] + '.txt'
        yolo_filepath = os.path.join(labels_folder, yolo_filename)
        
        # Obtener todas las anotaciones para esta imagen
        image_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
        
        with open(yolo_filepath, 'w') as f:
            for annotation in image_annotations:
                category_id = annotation['category_id']
                segmentation = annotation['segmentation'][0]  # Tomamos el primer polígono
                
                # Normalizar coordenadas de segmentación
                normalized_segmentation = normalize_coordinates(segmentation, image_width, image_height)
                
                # Escribir en formato YOLOv8 segmentation
                f.write(f"{category_id - 1} {' '.join(map(str, normalized_segmentation))}\n")

    print(f"Conversión completada para {set_name}. Archivos guardados en: {os.path.join(output_folder, set_name)}")
    return categories

# Rutas para los conjuntos de datos
datasets = {
    'train': {
        'coco_json': '/media/atdcctval/ESD-USB/train/_annotations.coco.json',
        'images_folder': '/media/atdcctval/ESD-USB/train'
    },
    'valid': {
        'coco_json': '/media/atdcctval/ESD-USB/valid/_annotations.coco.json',
        'images_folder': '/media/atdcctval/ESD-USB/valid'
    },
    'test': {
        'coco_json': '/media/atdcctval/ESD-USB/test/_annotations.coco.json',
        'images_folder': '/media/atdcctval/ESD-USB/test'
    }
}

output_base_folder = '/home/atdcctval/Downloads/wood'

# Procesar cada conjunto de datos
all_categories = {}
for set_name, paths in datasets.items():
    categories = process_dataset(paths['coco_json'], paths['images_folder'], output_base_folder, set_name)
    all_categories.update(categories)

# Crear archivo data.yaml
data_yaml_content = f"""
path: {output_base_folder}
train: train/images
val: valid/images
test: test/images

nc: {len(all_categories)}
names: {list(all_categories.values())}
"""

with open(os.path.join(output_base_folder, 'data.yaml'), 'w') as f:
    f.write(data_yaml_content)

print("Archivo data.yaml creado.")
