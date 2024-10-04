

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import os
from sklearn.metrics import r2_score

# GLOBAL VARIABLES
LENGTH_PX = 3840 
WIDTH_PX = 2160 
pixel_traking = []
pixel_mm_ratio = 0.06

# Cambiar nombres a estas variables??
redBajo1 = np.array([0, 100, 20], np.uint8)
redAlto1 = np.array([8, 255, 255], np.uint8)

redBajo2 = np.array([175, 100, 20], np.uint8)
redAlto2 = np.array([179, 255, 255], np.uint8)

# STYLE VARIABLES
font = cv2.FONT_HERSHEY_COMPLEX

# Abrir video
capture = cv2.VideoCapture(
    r'C:\Users\CCTVAL-USM\Desktop\Registros_FlexCam\25_03_2024\DSC_0103.MOV')

if not capture.isOpened():
  print("Cannot open camera")
  exit()

frame_count = 0
cv2.namedWindow("Video Original", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Video Original", 1920, 1080)

while (capture.isOpened()):
    # Capture frame-by-frame
    val_returned, frame = capture.read()
    if val_returned == True:
        frame = cv2.resize(frame, (LENGTH_PX, WIDTH_PX))

        # Gaussian para suavisar
        frame = cv2.bilateralFilter(frame, 5, 150, 150)

        # Obtener perfil HSV
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Crear mascara y aplicar máscara de color rojo
        maskRed = cv2.inRange(frame_hsv, redBajo1, redAlto1)
        maskRedvis = cv2.bitwise_and(frame, frame, mask=maskRed)

        # Obtener imagen en escala de grises, y aplicar treshold y así queda en ByN nada más
        frame_gray = cv2.cvtColor(maskRedvis, cv2.COLOR_BGR2GRAY)
        frame_binary = cv2.threshold(
            frame_gray, 100, 255, cv2.THRESH_BINARY)[1]

        # Encontrar contornos de la imagen binaria
        frame_contours, _ = cv2.findContours(
            frame_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Recorrer contornos encontrados, obtener centro de masa y dibujarlo.
        for contour in frame_contours:
          contour_area = cv2.contourArea(contour)
          if contour_area > 500:
            contour_moment = cv2.moments(contour)
            if (contour_moment["m00"] == 0):
              contour_moment["m00"] == 1
            contour_moment_x = int(
                contour_moment['m10'] / contour_moment['m00'])
            contour_moment_y = int(
                contour_moment['m01'] / contour_moment['m00'])
            cv2.circle(frame, (contour_moment_x, contour_moment_y),
                       5, (0, 255, 0), -1)
            new_contour = cv2.convexHull(contour)
            cv2.drawContours(frame, [new_contour], 0, (255, 0, 0), 2)

        # Imprimir el centro de masa
        print(f"Centro de Masa: ({contour_moment_x}, {contour_moment_y})")

        # Agregar pixel del centro de masa y a lista
        pixel_traking.append(contour_moment_y)

        # Mostrar la imagen en la ventana de visualización con una resolución de 1080p
        cv2.imshow("Video Original", cv2.resize(frame, (1920, 1080)))

        tecla = cv2.waitKey(1)
        if tecla == ord('q'):
            break
        frame_count += 1
    else:
      break

capture.release()
cv2.destroyAllWindows()

# Obtención de distancia en mm:
pixel_inicial = pixel_traking[0]
posicion_seguimiento = []

for pixel in pixel_traking:
  posicion_seguimiento.append((pixel-pixel_inicial)*pixel_mm_ratio)

deformacion = []
contador_aumento_posicion = 0
posicion_maxima = 0.0000
posicion_actual = 0.0000
posicion_anterior = 0.0000
for index, val in enumerate(posicion_seguimiento):
  if index == 0:
    deformacion.append(val)
  if index > 0:
    posicion_actual = val
    posicion_anterior = posicion_seguimiento[index-1]

    if posicion_actual == posicion_anterior:
      contador_aumento_posicion += 1
    else:
      contador_aumento_posicion = 0

    if (contador_aumento_posicion >= 10) & (posicion_actual >= posicion_maxima):
      posicion_maxima = posicion_actual

    deformacion.append(posicion_maxima)
print(" \n Longitud de arreglo de posición seguimiento:", len(deformacion))

indices_deformacion = np.linspace(
    1, len(deformacion), len(deformacion))  # [0, 1, ... , 14540]
print(indices_deformacion)

indices_deformacion_interpolados = np.linspace(
    1, 7500, 7500)  # [0, 1, ... , 7500]
print(indices_deformacion_interpolados)
deformacion_interpolada = np.interp(
    indices_deformacion_interpolados, indices_deformacion, deformacion)

plt.plot(indices_deformacion_interpolados, deformacion_interpolada)
plt.show()

# Exportar deformaciones a excel
data = {'Deformación': deformacion}
df = pd.DataFrame(data)


deformacion = df['Deformación']

cantidad_total = len(df['Deformación'])
print(cantidad_total)

# Calcular la media
media = deformacion.mean()

# Calcular la desviación estándar
desviacion_estandar = deformacion.std()

# Calcular el valor máximo
valor_maximo = deformacion.max()



# Imprimir los resultados
print(f"Media: {media:.2f}")
print(f"Desviación Estándar: {desviacion_estandar:.2f}")
print(f"Valor Máximo: {valor_maximo:.2f}")

ruta = 'C:\\Users\\CCTVAL-USM\\Desktop\\Registros_FlexCam\\25_03_2024'
nombre_archivo ='Ensayos Repetidos - 25-03-2024.xlsx'
# Une la ruta y el nombre del archivo
archivo = os.path.join(ruta, nombre_archivo)

# Define la página que deseas seleccionar
pagina = '2CUL100'  # reemplaza con el nombre de la página que deseas seleccionar
# Carga el archivo Excel y selecciona la página
df2 = pd.read_excel(archivo, sheet_name=pagina)

columnas = ['2CUL100 A', '2CUL100 B', '2CUL100 C']
df_seleccionado = df2[columnas]

# Elimina las primeras dos filas
df_seleccionado = df_seleccionado.iloc[2:, :]

# Asigna cada columna a una variable separada
var1 = df_seleccionado['2CUL100 A']
var2 = df_seleccionado['2CUL100 B']
var3 = df_seleccionado['2CUL100 C']

# Elimina los valores NaN de cada variable
var1 = var1.dropna()
var2 = var2.dropna()
var3 = var3.dropna()

def smape_asimetrico(y_true, y_pred):
    # Calcula el denominador
    denominador = np.abs(y_true) + np.abs(y_pred)
    
    # Asegúrate de que el denominador no sea cero
    denominador = np.where(denominador == 0, np.nan, denominador)  # Reemplaza ceros con NaN para evitar la división por cero
    
    # Calcula el resultado
    resultado = np.abs(y_true - y_pred) / denominador
    
    # Calcula la media, ignorando los NaN
    resultado_media = np.nanmean(resultado) * 100
    
    return resultado_media

def calcular_r2(y, y_pred):
    # Truncar los datos de predicción a la misma longitud que y
    y_pred_truncated = y_pred[:len(y)]
    
    # Calcular el Coeficiente de Determinación (R²)
    r2 = r2_score(y, y_pred_truncated)
    
    return r2


# Ejemplo de uso
sMAPE_var1 = smape_asimetrico(var1, deformacion)
sMAPE_var2 = smape_asimetrico(var2, deformacion)
sMAPE_var3 = smape_asimetrico(var3, deformacion)

# Muestra los resultados
print("sMAPE Var1:", sMAPE_var1)
print("sMAPE Var2:", sMAPE_var2)
print("sMAPE Var3:", sMAPE_var3)



# Gráfico de var1 vs deformacion
plt.figure(figsize=(10, 6))
plt.plot(var1, deformacion[:len(var1)], marker='o', color='blue')
plt.title('Relación entre var1 y deformación')
plt.xlabel('var1')
plt.ylabel('Deformación')
plt.grid(True)
plt.show()

# Gráfico de var2 vs deformacion
plt.figure(figsize=(10, 6))
plt.plot(var2, deformacion[:len(var2)], marker='o', color='green')
plt.title('Relación entre var2 y deformación')
plt.xlabel('var2')
plt.ylabel('Deformación')
plt.grid(True)
plt.show()

# Gráfico de var3 vs deformacion
plt.figure(figsize=(10, 6))
plt.plot(var3, deformacion[:len(var3)], marker='o', color='red')
plt.title('Relación entre var3 y deformación')
plt.xlabel('var3')
plt.ylabel('Deformación')
plt.grid(True)
plt.show()
