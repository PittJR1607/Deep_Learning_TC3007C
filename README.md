# 🎨 Art Style Transfer with Neural Networks

Este proyecto implementa una red de transferencia de estilo neuronal para aplicar estilos artísticos específicos a imágenes de contenido. Utilizando redes neuronales convolucionales y técnicas avanzadas de procesamiento de imágenes, el modelo es capaz de aprender características visuales de diferentes estilos de arte y aplicarlos a nuevas imágenes. A continuación, se detallan los archivos clave del proyecto, la estructura del dataset y los pasos para entrenar y probar el modelo.

PARA EL REPROTE COMPLETO DEL MODELO, ABRIR Módulo_2_Reporte_Modelo_Deep_Learning .pdf

---

## 📂 Estructura del Proyecto

### Archivos principales

- **Art_Generator_STN.ipynb**: Notebook que contiene el modelo completo y el flujo de entrenamiento. En este archivo se entrena el modelo para capturar los estilos artísticos y generar las imágenes estilizadas.
- **Art_Generator_Tester.py**: Script de Python para probar el modelo entrenado. Permite aplicar estilos a nuevas imágenes y visualizar los resultados estilizados.
- **StyleTransferNetwork.py**: Archivo que define la arquitectura de la red neuronal StyleTransferNetwork (STN), utilizada para capturar y transferir el estilo visual de las imágenes de referencia a las imágenes de contenido.

---

## 🖼️ Estructura del Dataset

Link al Dataset: https://www.kaggle.com/datasets/steubk/wikiart/data

El dataset utilizado es **WikiArt**, que contiene imágenes organizadas en carpetas, donde cada carpeta representa un estilo artístico diferente. Para el desarrollo de este proyecto, se seleccionaron los siguientes estilos:

1. **Barroco**: 4,241 imágenes
2. **Arte Moderno**: 4,335 imágenes
3. **Cubismo**: 2,236 imágenes
4. **Impresionismo**: 13,061 imágenes

Las imágenes dentro de cada carpeta son representativas de su estilo, incluyendo pinturas, esculturas y otras obras de arte. Este dataset sirve como base para que el modelo aprenda patrones y texturas características de cada estilo, generando una matriz de Gram promedio que se utiliza para estilizar nuevas imágenes de contenido.

---

## 🚀 Instrucciones de Uso

### 1. Entrenamiento del Modelo

**NOTA:EN CASO DE CORRER EN OTRO DISPOSITIVO, CAMBIAR EL ENTRENAMIENTO DE GPU, CAMBIAR DE MPS A CUDA**

Para entrenar el modelo, abre y ejecuta el archivo `Art_Generator_STN.ipynb` en Jupyter Notebook o Google Colab. Este archivo contiene el flujo completo de entrenamiento:

1. **Carga de imágenes**: Se carga el dataset WikiArt con los estilos deseados.
2. **Extracción de características**: Se utiliza un modelo VGG-19 preentrenado para extraer características de cada estilo artístico.
3. **Cálculo de la matriz de Gram**: Calcula la matriz de Gram promedio para capturar la esencia de cada estilo.
4. **Entrenamiento de la red StyleTransferNetwork**: La red neuronal convolucional STN se entrena para capturar y replicar los estilos visuales específicos.

#### Parámetros de entrenamiento
- **Tamaño de imagen**: 512x512 píxeles
- **Tasa de aprendizaje**: 1e-3
- **Épocas**: 500

### 2. Prueba del Modelo

Para aplicar los estilos aprendidos a una imagen nueva, utiliza el script `Art_Generator_Tester.py`. Este script te permite seleccionar una imagen de contenido y aplicar cualquiera de los estilos entrenados.
