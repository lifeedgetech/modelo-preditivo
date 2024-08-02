import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.preprocessing import LabelBinarizer

# Supomos que suas etiquetas são categóricas e precisamos binarizá-las
def binarize_labels(y):
    lb = LabelBinarizer()
    return lb.fit_transform(y)

# Substitua estas linhas pelos seus dados de treino e validação
# Estes são apenas exemplos e devem ser substituídos pelos seus dados reais
X_train = np.random.rand(100, 128, 128, 3)  # seus dados de treino
y_train = np.random.randint(0, 2, 100)     # suas etiquetas de treino, binárias
X_test = np.random.rand(20, 128, 128, 3)   # seus dados de validação
y_test = np.random.randint(0, 2, 20)       # suas etiquetas de validação, binárias

# Binarize as etiquetas se necessário
y_train = binarize_labels(y_train)
y_test = binarize_labels(y_test)

# Definir o modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Saída binária
])

# Compilar o modelo com binary_crossentropy
model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['accuracy'])

# Gerador de dados de imagem
datagen = ImageDataGenerator(rescale=1./255)

# Ajustar o gerador de dados
train_generator = datagen.flow(X_train, y_train, batch_size=32)
validation_generator = datagen.flow(X_test, y_test, batch_size=32)

# Treinar o modelo
history = model.fit(train_generator, epochs=50, validation_data=validation_generator)

# Salvar o modelo
model.save('parkinson_mri_cnn_model.h5')
