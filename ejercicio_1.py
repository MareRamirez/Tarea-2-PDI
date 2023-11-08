import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from skimage import filters

#Parte 1.A
original_img = cv2.imread('numbers.jpg') # Se lee la imagen
alto, ancho = original_img.shape[:2] # Se obtiene el alto y ancho de la imagen
mitad_vertical = alto // 2 # Se obtiene la mitad de la altura de la imagen
img_superior = original_img[:mitad_vertical, :] # Se obtiene la imagen superior

# Se inverte la parte superior para el efecto espejo
img_espejo_superior = cv2.flip(img_superior, 0)  # 0 para espejo vertical
resultado = np.vstack((img_superior, img_espejo_superior)) # Se apilan las imagenes

# Mostrar la img con el efecto de espejo en una sola figura
plt.figure() # Se crea una figura
plt.subplot(1, 2, 1) # Se crea una subfigura
plt.title('img original') # Se agrega un titulo a la subfigura
plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)) # Se muestra la img original

plt.subplot(1, 2, 2) # Se crea una subfigura
plt.title('img numeros original y espejo') # Se agrega un titulo a la subfigura
plt.imshow(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB)) # Se muestra la img con efecto de espejo

plt.show() # Se muestra la figura


#Parte 1.B 
# Definición de la región a suavisar
x1, y1 = 600, 700  # Esquina superior izquierda
x2, y2 = 750, 750 # Esquina inferior derecha
region_to_smooth = original_img[y1:y2, x1:x2] 

# Definir el tamaño de la ventana del filtro de media
window_size = (7, 7)  # Ajusta el tamaño según tu preferencia

# Crear el filtro de media
filtro_media = np.ones(window_size, dtype=np.float32) / (window_size[0] * window_size[1])

# Aplicar el filtro de media a la región que deseas suavizar
region_suavizada = cv2.filter2D(region_to_smooth, -1, filtro_media)

# Combinar la región suavizada con el resto de la imagen original
imagen_resultante = np.copy(original_img)
imagen_resultante[y1:y2, x1:x2] = region_suavizada

# Mostrar la imagen original y la imagen con la región suavizada en una sola figura
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
plt.title('Imagen original')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(imagen_resultante, cv2.COLOR_BGR2RGB))
plt.title('Imagen numeros con marca de agua suavizada')

plt.show()


#Parte 1.C

#Parte 1.D
imagen_original = cv2.imread('numbers.jpg', cv2.IMREAD_GRAYSCALE)
sob = filters.sobel(imagen_original) # Aplicar el filtro de Sobel
umbral = sob.max() * 0.5 # Calcular el umbral para binarizar la imagen
sob_b = (sob <= umbral).astype(np.uint8) * 255 # Binarizar la imagen

plt.figure(figsize=(15, 15))
plt.subplot(131)
plt.imshow(cv2.cvtColor(imagen_original, cv2.COLOR_BGR2RGB))
plt.title('Original')
plt.axis('off')

plt.subplot(132)
plt.imshow(sob, cmap='Greys_r')
plt.title('Sobel')
plt.axis('off')

plt.subplot(133)
plt.imshow(sob_b, cmap='Greys_r')
plt.title('Sobel Binarizada')
plt.axis('off')

plt.show()