import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

train_datagen = ImageDataGenerator(rescale=1./255, 
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    './data_set_orquideas_normal_70_30_30/train',
    target_size=(180, 180),  # Adjusted target size for AlexNet
    batch_size=32,
    class_mode='categorical',
    classes=['Oncidium', 'Zygopetalum', 'Vanda', 'Angraecum'])

validation_generator = val_datagen.flow_from_directory(
    './data_set_orquideas_normal_70_30_30/validation',
    target_size=(180, 180),  # Adjusted target size for AlexNet
    batch_size=32,
    class_mode='categorical',
    classes=['Oncidium', 'Zygopetalum', 'Vanda', 'Angraecum'])

test_generator = val_datagen.flow_from_directory(
    './data_set_orquideas_normal_70_30_30/test',
    target_size=(180, 180),  # Adjusted target size for AlexNet
    batch_size=32,
    class_mode='categorical',
    classes=['Oncidium', 'Zygopetalum', 'Vanda', 'Angraecum'])

model = keras.Sequential([
    keras.layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu'),
    keras.layers.MaxPooling2D((3, 3), strides=(2, 2)),
    keras.layers.Conv2D(256, (5, 5), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((3, 3), strides=(2, 2)),
    keras.layers.Conv2D(384, (3, 3), activation='relu', padding='same'),
    keras.layers.Conv2D(384, (3, 3), activation='relu', padding='same'),
    keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((3, 3), strides=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_generator, epochs=10,
                    validation_data=validation_generator)

test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print('Test accuracy:', test_acc)

# Make predictions on test data
predictions = model.predict(test_generator)
predicted_labels = np.argmax(predictions, axis=1)

# Get true labels of the test data
true_labels = test_generator.classes

# Calculate metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=1)
recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=1)
f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=1)

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1-score:', f1)