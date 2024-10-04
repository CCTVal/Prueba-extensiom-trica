import cv2
import numpy as np
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import time

# Cargar el modelo YOLOv8 con segmentación preentrenado
model = YOLO('wood3.pt')  # Asegúrate de que sea un modelo con segmentación

# GLOBAL VARIABLES
LENGTH_PX = 3840
WIDTH_PX = 2160
pixel_mm_ratio = 0.06  # Ajusta según tu configuración (mm/pixel)
fps = 30  # Cuadros por segundo del video
mediciones_superior = []  # Arreglo para almacenar las mediciones de la curva superior
mediciones_inferior = []  # Arreglo para almacenar las mediciones de la curva inferior
posicion_inicial_superior = None  # Variable para almacenar la posición inicial superior
posicion_inicial_inferior = None  # Variable para almacenar la posición inicial inferior

# Cambia la ruta del archivo a la ruta correcta en tu sistema
capture = cv2.VideoCapture('/home/atdcctval/Desktop/Registros_FlexCam/25_03_2024/DSC_0102.MOV')

# Verifica si el video se ha abierto correctamente
if not capture.isOpened():
    print("Error: No se puede abrir el video.")
    exit()

print("Video abierto correctamente.")

# Crear el objeto VideoWriter para guardar el video procesado
output_path = '/home/atdcctval/Desktop/output_video.avi'  # Ruta donde se guardará el video
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec para el archivo de video
output_video = cv2.VideoWriter(output_path, fourcc, fps, (LENGTH_PX, WIDTH_PX))

cv2.namedWindow("Video Original", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Video Original", 1920, 1080)

frame_count = 0  # Contador de cuadros

while capture.isOpened():
    val_returned, frame = capture.read()
    if val_returned:
        frame = cv2.resize(frame, (LENGTH_PX, WIDTH_PX))
        
        # Detectar objetos usando YOLOv8 con segmentación
        results = model(frame)
        
        # Iterar sobre las detecciones
        for result in results:
            boxes = result.boxes
            masks = result.masks  # Extraer las máscaras de segmentación
            for i, box in enumerate(boxes):
                # Filtrar para encontrar solo la madera (clase 'wood') y confianza >= 70%
                if model.names[int(box.cls[0])] == 'wood' and box.conf[0] >= 0.70:
                    # Obtener las coordenadas de la caja
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                    # Obtener y aplicar la máscara de segmentación correspondiente
                    mask = masks.data[i].cpu().numpy()  # Convertir la máscara a un array de NumPy
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))  # Redimensionar la máscara al tamaño del frame
                    mask = (mask > 0.5).astype(np.uint8)  # Binarizar la máscara
                    
                    # Limpiar la máscara utilizando operaciones morfológicas
                    kernel = np.ones((7, 7), np.uint8)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Limpiar ruido adicional
                    
                    # Aplicar dilatación y erosión para reducir el ruido
                    mask = cv2.dilate(mask, kernel, iterations=2)
                    mask = cv2.erode(mask, kernel, iterations=2)

                    # Crear una imagen de color para segmentar el área de la madera
                    colored_mask = np.zeros_like(frame)  # Crear una máscara de color del mismo tamaño que el frame original
                    colored_mask[mask == 1] = (70, 268, 180)  # Verde pastel para la segmentación de la madera

                    # Aplicar la máscara coloreada sobre el frame original
                    frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)  # Superponer la máscara coloreada con el frame original

                    # Encontrar los contornos en la máscara
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        # Filtrar por área mínima para evitar contornos pequeños (ruido)
                        min_contour_area = 1000  # Ajusta este valor según sea necesario
                        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
                        
                        if filtered_contours:
                            # Encontrar el contorno más grande (suponemos que es la madera)
                            largest_contour = max(filtered_contours, key=cv2.contourArea)
                            
                            # Encontrar los puntos más altos y más bajos de la madera en el contorno
                            topmost_point = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
                            bottommost_point = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])
                            
                            # Calcular la deformación entre la curva superior e inferior
                            deformation_px = bottommost_point[1] - topmost_point[1]
                            deformation_mm = deformation_px * pixel_mm_ratio
                            
                            # Establecer la posición inicial si no está definida
                            if posicion_inicial_superior is None and posicion_inicial_inferior is None:
                                posicion_inicial_superior = topmost_point[1] * pixel_mm_ratio
                                posicion_inicial_inferior = bottommost_point[1] * pixel_mm_ratio
                            
                            # Guardar las mediciones relativas a la posición inicial
                            medicion_superior_relativa = (topmost_point[1] * pixel_mm_ratio) - posicion_inicial_superior
                            medicion_inferior_relativa = (bottommost_point[1] * pixel_mm_ratio) - posicion_inicial_inferior
                            mediciones_superior.append(medicion_superior_relativa)
                            mediciones_inferior.append(medicion_inferior_relativa)
                            
                            # Dibujar la caja y el contorno en la imagen
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.drawContours(frame, [largest_contour], -1, (255, 0, 0), 2)  # Dibuja el contorno
                            
                            # Mostrar la deformación calculada en milímetros en la imagen
                            text_size = cv2.getTextSize(f"Deformacion: {deformation_mm:.2f} mm", 
                                                        cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                            text_x = x1 + (x2 - x1 - text_size[0]) // 2
                            cv2.putText(frame, f"Deformacion: {deformation_mm:.2f} mm", (text_x, y1 - 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Texto en rojo para resaltar
                            cv2.circle(frame, topmost_point, 5, (0, 0, 255), -1)  # Punto más alto (curva superior)
                            cv2.circle(frame, bottommost_point, 5, (0, 255, 0), -1)  # Punto más bajo (curva inferior)

        # Escribir el cuadro procesado en el archivo de salida
        output_video.write(frame)
        
        # Mostrar la imagen en la ventana de visualización con una resolución de 1080p
        cv2.imshow("Video Original", cv2.resize(frame, (1920, 1080)))
        
        # Incrementar el contador de cuadros
        frame_count += 1
        
        # Detener la ejecución al presionar la tecla 'q'
        tecla = cv2.waitKey(1)
        if tecla == ord('q'):
            break
    else:
        break

# Liberar la captura de video y cerrar todas las ventanas
capture.release()
output_video.release()  # Liberar el VideoWriter
cv2.destroyAllWindows()

# Imprimir las mediciones de la curva superior e inferior almacenadas en cada segundo
print("Mediciones de la curva superior (en milímetros):")
print(mediciones_superior)
print("Mediciones de la curva inferior (en milímetros):")
print(mediciones_inferior)

# Crear gráfico
plt.figure(figsize=(10, 6))
plt.plot(mediciones_superior, marker='o', color='r', label='Curva Superior (mm)')
plt.plot(mediciones_inferior, marker='o', color='b', label='Curva Inferior (mm)')
plt.title('Mediciones Relativas de la Deformación de la Madera')
plt.xlabel('Tiempo (segundos)')
plt.ylabel('Deformación Relativa (mm)')
plt.legend(loc='best')
plt.grid(True)
plt.show()












