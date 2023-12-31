import cv2
import numpy as np
import matplotlib.pyplot as plt

#########################################################################################################

# En el ejercicio c) ocupé la imagen de la tortuga con el filtro de la mediana ya que a mi parecer era 
# el que mejor se veía, pero al compararla con la imagen referencial de la tortuga de colores no se veía
# igual, por qué pasa esto?

# PD: en c) Y d) imprimí más tortugas de las que debería porque quería compararlas

#########################################################################################################


# a) Carga y despliegue de imagenes en escala de grises.

# Carga de imagenes
img = cv2.imread('tortugas1.jpg',0)
img2 = cv2.imread('tortugas2.jpeg',0)
# Despliegue de imagenes
cv2.imshow('Tortuga 1',img)
cv2.imshow('Tortuga 2',img2)
cv2.waitKey(0)

# b) Aplique los filtros de media, mediana y gaussiano vistos en clases (con el menor suavizado
# posible para reducir el ruido), a cada imagen. Despliegue los resultados de cada imagen en
# escala de grises y muestre los resultados en subplots con los nombre correspondientes. Elija
# el que considere mejor para cada imagen y explique la selección de cada filtro.

# Aplica los filtros
img_media = cv2.blur(img, (3,3))
img_mediana = cv2.medianBlur(img, 3)
img_gauss = cv2.GaussianBlur(img, (3,3), 0)

img2_media = cv2.blur(img2, (3,3))
img2_mediana = cv2.medianBlur(img2, 3)
img2_gauss = cv2.GaussianBlur(img2, (3,3), 0)

# Configuración de la figura de Matplotlib para mostrar las imágenes
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Mostrar imágenes de la primera tortuga
axs[0, 0].imshow(img_media, cmap='gray')
axs[0, 0].set_title('Media Tortuga 1')
axs[0, 1].imshow(img_mediana, cmap='gray')
axs[0, 1].set_title('Mediana Tortuga 1')
axs[0, 2].imshow(img_gauss, cmap='gray')
axs[0, 2].set_title('Gaussiano Tortuga 1')

# Mostrar imágenes de la segunda tortuga
axs[1, 0].imshow(img2_media, cmap='gray')
axs[1, 0].set_title('Media Tortuga 2')
axs[1, 1].imshow(img2_mediana, cmap='gray')
axs[1, 1].set_title('Mediana Tortuga 2')
axs[1, 2].imshow(img2_gauss, cmap='gray')
axs[1, 2].set_title('Gaussiano Tortuga 2')

# Ocultar ejes
for ax in axs.flat:
    ax.axis('off')

# Mostrar las tortugas
plt.show()

# c) Cree dos matrices de unos, con dimensiones de 3x3 y 5x5, respectivamente. Luego, mediante
# convolución, aplíquelas por separado sobre las imágenes originales en escala de grises. Mues-
# tre los resultados en subplots. Comente los resultados obtenidos. 

# Aplica los filtros de media, mediana y gaussiano
img_media = cv2.blur(img, (3,3))
img_mediana = cv2.medianBlur(img, 3)
img_gauss = cv2.GaussianBlur(img, (3,3), 0)

img2_media = cv2.blur(img2, (3,3))
img2_mediana = cv2.medianBlur(img2, 3)
img2_gauss = cv2.GaussianBlur(img2, (3,3), 0)

# Crea las matrices de convolución
kernel_3 = np.ones((3,3),np.float32) / 9  # Normaliza el kernel para mantener el rango de la imagen
kernel_5 = np.ones((5,5),np.float32) / 25  # Normaliza el kernel

# Aplica las convoluciones
img_conv_3 = cv2.filter2D(img, -1, kernel_3)
img_conv_5 = cv2.filter2D(img, -1, kernel_5)

img2_conv_3 = cv2.filter2D(img2, -1, kernel_3)
img2_conv_5 = cv2.filter2D(img2, -1, kernel_5)

# Configura la figura de Matplotlib para mostrar las imágenes
fig, axs = plt.subplots(2, 5, figsize=(20, 8))  # Ajusta la figura a dos filas y cinco columnas

# Mostrar imágenes procesadas con los filtros para la primera tortuga
axs[0, 0].imshow(img_media, cmap='gray')
axs[0, 0].set_title('Media 3x3 Tortuga 1')
axs[0, 1].imshow(img_mediana, cmap='gray')
axs[0, 1].set_title('Mediana 3x3 Tortuga 1')
axs[0, 2].imshow(img_gauss, cmap='gray')
axs[0, 2].set_title('Gaussiano 3x3 Tortuga 1')
axs[0, 3].imshow(img_conv_3, cmap='gray')
axs[0, 3].set_title('Convolución 3x3 Tortuga 1')
axs[0, 4].imshow(img_conv_5, cmap='gray')
axs[0, 4].set_title('Convolución 5x5 Tortuga 1')

# Mostrar imágenes procesadas con los filtros para la segunda tortuga
axs[1, 0].imshow(img2_media, cmap='gray')
axs[1, 0].set_title('Media 3x3 Tortuga 2')
axs[1, 1].imshow(img2_mediana, cmap='gray')
axs[1, 1].set_title('Mediana 3x3 Tortuga 2')
axs[1, 2].imshow(img2_gauss, cmap='gray')
axs[1, 2].set_title('Gaussiano 3x3 Tortuga 2')
axs[1, 3].imshow(img2_conv_3, cmap='gray')
axs[1, 3].set_title('Convolución 3x3 Tortuga 2')
axs[1, 4].imshow(img2_conv_5, cmap='gray')
axs[1, 4].set_title('Convolución 5x5 Tortuga 2')

# Ocultar los ejes 
for ax in axs.flat:
    ax.axis('off')
# Para ajustar el espacio entre las imagenes
plt.tight_layout()
# Mostrar todas las tortugas
plt.show()

# ¿Qué efecto realizaron las máscaras de convolución sobre el ruido de las imágenes?
# Las máscaras de convolución suavizan la imagen, eliminando el ruido de la misma. 

# d) Cuantifique las imagenes , obtenidas de los incisos b) o c), con 16
# niveles de gris y despliegue el resultado en una figura.

# Cuantificación de imágenes con 16 niveles de gris
img_media_16 = np.uint8(img_media / 16) * 16
img_mediana_16 = np.uint8(img_mediana / 16) * 16
img_gauss_16 = np.uint8(img_gauss / 16) * 16
img_conv_3_16 = np.uint8(img_conv_3 / 16) * 16
img_conv_5_16 = np.uint8(img_conv_5 / 16) * 16

img2_media_16 = np.uint8(img2_media / 16) * 16
img2_mediana_16 = np.uint8(img2_mediana / 16) * 16
img2_gauss_16 = np.uint8(img2_gauss / 16) * 16
img2_conv_3_16 = np.uint8(img2_conv_3 / 16) * 16
img2_conv_5_16 = np.uint8(img2_conv_5 / 16) * 16

# Configura la figura de Matplotlib para mostrar las imágenes en subplots
fig, axs = plt.subplots(2, 5, figsize=(20, 8))  # Ajusta la figura a dos filas y cinco columnas

# Mostrar imágenes cuantificadas de la primera tortuga
axs[0, 0].imshow(img_media_16, cmap='gray')
axs[0, 0].set_title('Media 16 Tortuga 1')
axs[0, 1].imshow(img_mediana_16, cmap='gray')
axs[0, 1].set_title('Mediana 16 Tortuga 1')
axs[0, 2].imshow(img_gauss_16, cmap='gray')
axs[0, 2].set_title('Gaussiano 16 Tortuga 1')
axs[0, 3].imshow(img_conv_3_16, cmap='gray')
axs[0, 3].set_title('Convolución 3x3 16 Tortuga 1')
axs[0, 4].imshow(img_conv_5_16, cmap='gray')
axs[0, 4].set_title('Convolución 5x5 16 Tortuga 1')

# Mostrar imágenes cuantificadas de la segunda tortuga
axs[1, 0].imshow(img2_media_16, cmap='gray')
axs[1, 0].set_title('Media 16 Tortuga 2')
axs[1, 1].imshow(img2_mediana_16, cmap='gray')
axs[1, 1].set_title('Mediana 16 Tortuga 2')
axs[1, 2].imshow(img2_gauss_16, cmap='gray')
axs[1, 2].set_title('Gaussiano 16 Tortuga 2')
axs[1, 3].imshow(img2_conv_3_16, cmap='gray')
axs[1, 3].set_title('Convolución 3x3 16 Tortuga 2')
axs[1, 4].imshow(img2_conv_5_16, cmap='gray')
axs[1, 4].set_title('Convolución 5x5 16 Tortuga 2')

# Ocultar los ejes 
for ax in axs.flat:
    ax.axis('off')
# Para ajustar el espacio entre las imagenes
plt.tight_layout()
# Mostrar la ventana con todos los subplots
plt.show()

# La imagen mejor filtrada es la imagen filtrada con el filtro de mediana, ya que es la que mejor
# conserva los bordes de la imagen.

# e) Cree y despliegue una imagen de 3 canales a partir de la imagen cuantificada en el inciso
# anterior con los colores del mapa de colores ‘cool’ de la biblioteca matplotlib. Para obtener el
# mapa de colores utilice la función get cmap disponible en la biblioteca matplotlib. Asigne los
# valores RGB del mapa a los píxeles que comparten el mismo nivel de cuantificación.

# Mapa de colores de Matplotlib
cool_map = plt.get_cmap('cool')

# Función para aplicar los colores a la imagen
def apply_cool_colormap(image):
    normalized_img = image / image.max()
    cool_image = cool_map(normalized_img)
    return (cool_image[:, :, :3] * 255).astype('uint8')  # Convierte a RGB

# Aplica el color a cada imagen
img_media_16_color = apply_cool_colormap(img_media_16)
img_mediana_16_color = apply_cool_colormap(img_mediana_16)
img_gauss_16_color = apply_cool_colormap(img_gauss_16)

# Establece el espacio para mostrar las imágenes
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Muestra las imágenes en sus espacios correspondientes
axes[0].imshow(img_media_16_color)
axes[0].set_title('Media 16 Tortuga 1')
axes[0].axis('off')  # Oculta los ejes
axes[1].imshow(img_mediana_16_color)
axes[1].set_title('Mediana 16 Tortuga 1')
axes[1].axis('off')  
axes[2].imshow(img_gauss_16_color)
axes[2].set_title('Gaussiano 16 Tortuga 1')
axes[2].axis('off')  

# Muestra todas las tortugas de colores
plt.show()





