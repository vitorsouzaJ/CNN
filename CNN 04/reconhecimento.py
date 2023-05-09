import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from PIL import Image

# Diretório de dados
train_dir = './data/train'
validation_dir = './data/validation'
test_dir = './data/test'

# Dimensões da imagem
img_width = 224
img_height = 224

# Cria gerador de dados para aumento da base de treinamento
datagen = ImageDataGenerator(
    rotation_range=90,  # rotação nos 4 sentidos
    rescale=1./255,     # normalização dos pixels
    shear_range=0.2,    # distorção de cisalhamento
    zoom_range=0.2,     # distorção de zoom
    horizontal_flip=True,  # inversão horizontal
    fill_mode='nearest'     # modo de preenchimento
)

# Cria gerador de dados para treinamento
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',
    classes=['Oncidium', 'Zygopetalum', 'Vanda', 'Angraecum']
)

# Cria gerador de dados para validação
validation_generator = datagen.flow_from_directory(
    validation_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',
    classes=['Oncidium', 'Zygopetalum', 'Vanda', 'Angraecum']

)


test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    classes=['Oncidium', 'Zygopetalum', 'Vanda', 'Angraecum']

)
# Cria modelo da rede neural
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(4, activation='softmax')
])

# Compila modelo
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Treina modelo
history = model.fit(train_generator,
                    epochs=10,
                    validation_data=validation_generator)

# Avalia modelo
test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical'
)

test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)
