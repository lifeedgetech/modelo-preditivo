import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall
from tensorflow.keras.losses import BinaryCrossentropy 
import cv2
from collections import Counter

# Defina o caminho para o diretório das imagens
data_dir = 'imagens/parkinsons_dataset'
img_size = (128, 128)  # Redimensione as imagens para um tamanho fixo

# Função para carregar imagens
def load_images(data_dir, img_size):
    images = []
    labels = []
    for label in ['normal', 'parkinson']:
        class_dir = os.path.join(data_dir, label)
        for img in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img)
            img_array = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
            img_array = tf.keras.preprocessing.image.img_to_array(img_array)
            img_array /= 255.0  # Normalizar as imagens
            images.append(img_array)
            labels.append(label)
    return np.array(images), np.array(labels)

def balance_dataset(images, labels, img_size):
    label_counts = Counter(labels)
    max_count = max(label_counts.values())
    balanced_images = []
    balanced_labels = []

    for label in label_counts:
        class_images = images[labels == label]
        class_count = len(class_images)
        
        if class_count < max_count:
            augmented_images = augment_images(class_images, max_count - class_count, img_size)
            balanced_images.extend(augmented_images)
            balanced_labels.extend([label] * (max_count - class_count))
        
        balanced_images.extend(class_images)
        balanced_labels.extend([label] * class_count)

    return np.array(balanced_images), np.array(balanced_labels)

def augment_images(class_images, max_count, img_size):
    augmented_images = []
    for _ in range(max_count):
        idx = np.random.randint(0, len(class_images))
        img = class_images[idx].copy()
        
        # Apply random augmentations
        if np.random.rand() < 0.5:
            img = cv2.GaussianBlur(img, (15, 15), 0)
        
        if np.random.rand() < 0.5:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        if np.random.rand() < 0.5:
            _, img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 128, 255, cv2.THRESH_BINARY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        
        if np.random.rand() < 0.5:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # Converte a imagem para o tipo CV_8UC1, se necessário
            gray_img = cv2.convertScaleAbs(gray_img)
            contours, _ = cv2.findContours(gray_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
            img = cv2.bitwise_and(img, img, mask=mask)
        
        img = cv2.resize(img, img_size)
        augmented_images.append(img)
    
    return augmented_images

# Load and balance the dataset
images, labels = load_images(data_dir, img_size)

# images, labels = balance_dataset(images, labels, img_size)

# Codificar os rótulos
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = tf.keras.utils.to_categorical(labels)

# Dividir o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss=BinaryCrossentropy(), metrics=[Recall()])
model.summary()

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

datagen.fit(X_train)

history = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=50, validation_data=(X_test, y_test))

# Avaliar o modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# Plotar a acurácia e perda
# plt.figure(figsize=(12, 4))
print(history)
# plt.subplot(1, 2, 1)
# plt.plot(history.history['Recall'], label='Acurácia de Treinamento')
# plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
# plt.xlabel('Época')
# plt.ylabel('Acurácia')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'], label='Perda de Treinamento')
# plt.plot(history.history['val_loss'], label='Perda de Validação')
# plt.xlabel('Época')
# plt.ylabel('Perda')
# plt.legend()

# plt.show()

model.save('parkinson_mri_cnn_model_3.h5')
