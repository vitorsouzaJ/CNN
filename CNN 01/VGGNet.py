import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Definir os caminhos das pastas de treinamento, validação e teste
train_dir = './data_set_orquideas_normal_70_30_30/train'
val_dir = './data_set_orquideas_normal_70_30_30/validation'
test_dir = './data_set_orquideas_normal_70_30_30/test'

# Definir o tamanho das imagens de entrada
input_shape = (224, 224, 3)  # Tamanho utilizado pela VGGNet

# Definir o número de classes
num_classes = 4

# Criar geradores de dados para pré-processamento e aumento de dados
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=input_shape[:2],
    batch_size=32,
    class_mode='categorical',
    classes=['Oncidium', 'Zygopetalum', 'Vanda', 'Angraecum']
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=input_shape[:2],
    batch_size=32,
    class_mode='categorical',
    classes=['Oncidium', 'Zygopetalum', 'Vanda', 'Angraecum']
)

## Construir o modelo VGGNet
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), strides=(2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), strides=(2, 2)),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), strides=(2, 2)),
    Flatten(),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compilar o modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Treinar o modelo
history = model.fit(train_generator, epochs=30, validation_data=validation_generator)

# Avaliar o modelo no conjunto de teste
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=input_shape[:2],
    batch_size=1,  # Alterado para batch_size=1 para obter previsões por amostra
    class_mode='categorical',
    shuffle=False,  # Desativar o embaralhamento para garantir correspondência entre previsões e rótulos
    classes=['Oncidium', 'Zygopetalum', 'Vanda', 'Angraecum']
)

test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print('Test accuracy:', test_acc)

# Fazer previsões nos dados de teste
predictions = model.predict(test_generator)
predicted_labels = np.argmax(predictions, axis=1)

# Obter rótulos reais dos dados de teste
true_labels = test_generator.classes

# Calcular métricas
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=1)
recall = recall_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')

print('Acurácia:', accuracy)
print('Precisão:', precision)
print('Recall:', recall)
print('F1-score:', f1)