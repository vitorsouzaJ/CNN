import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, 
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    './data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    classes=['Oncidium', 'Zygopetalum', 'Vanda', 'Angraecum'])

validation_generator = val_datagen.flow_from_directory(
    './data/validation',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    classes=['Oncidium', 'Zygopetalum', 'Vanda', 'Angraecum'])

test_generator = val_datagen.flow_from_directory(
    './data/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    classes=['Oncidium', 'Zygopetalum', 'Vanda', 'Angraecum'])

model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(128, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_generator, epochs=10,
                    validation_data=validation_generator)

test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print('Test accuracy:', test_acc)