# Detección temprana de cambios de régimen de mercado y anomalías financieras mediante Deep Learning no supervisado

## Descripción del Proyecto
El objetivo principal de este repositorio es actuar como un "Semáforo para Inversores". A diferencia de los modelos tradicionales de predicción de precios, este sistema utiliza un enfoque de aprendizaje no supervisado para clasificar el estado de salud del mercado financiero y detectar transiciones hacia regímenes de alta volatilidad o pánico.

## Innovación Tecnológica

El sistema implementa una arquitectura híbrida avanzada para superar las limitaciones de los indicadores técnicos clásicos (como el retraso temporal y la suposición de estacionariedad):

- **Autoencoders Convolucionales (CNN-AE)**: Para el aprendizaje de la representación latente de condiciones de mercado "normales" y extracción de características espaciales.

- **Temporal Convolutional Networks (TCN):** Para la captura de dependencias temporales y tendencias de largo plazo.

- **Detección por Error de Reconstrucción:** El sistema identifica anomalías cuando los datos entrantes difieren significativamente del patrón de estabilidad aprendido por la red.

## Stack Tecnológico

**Lenguaje:** Python 3.x

**Frameworks de IA:** [TensorFlow / PyTorch] (especificar el elegido)

**Tratamiento de Datos:** Pandas, NumPy, Scikit-learn

**Visualización:** Matplotlib, Seaborn

## Organización del Grupo

Este es un proyecto grupal desarrollado por:

- **Cristian José Pérez García -** Ingeniería y Pipeline de Datos

- **Carlos Seguí Martínez -** Diseño e Implementación del Modelo CNN-AE

- **Alexander Jestaedt Maldonado -** Diseño e Implementación del Modelo TCN

Datos: Series temporales de activos financieros (API de Yahoo Finance)
