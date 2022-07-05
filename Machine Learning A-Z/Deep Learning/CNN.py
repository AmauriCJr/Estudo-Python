import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

tf.__version__

#Preprocessing the Training set

#Image Augmentation
train_datagen = ImageDataGenerator(
    rescale = 1./255, #Feature Scaling
    shear_range = 0.2,  #Transformações que serão utilizadas para evitar overfitting
    zoom_range = 0.2,
    horizontal_flip = True)
training_set = train_datagen.flow_from_directory(
    'D:/Backup/Documents/Machine Learning A-Z (Codes and Datasets)/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/dataset/training_set',
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'binary')

# Preprocessing the Test set

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory(
    'D:/Backup/Documents/Machine Learning A-Z (Codes and Datasets)/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/dataset/test_set',
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'binary')


# Initialising the CNN

cnn = tf.keras.models.Sequential()

#Convolution

cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu', input_shape = [64, 64, 3]))
#filters - número de filtros
#kernel_size - tamanho da matriz do filtro
#activation - tipo de filtro aplicado ao pesos das sinapses, no caso retificador
#input_shape - formato do dado de entrada, só é colocodao na input layer

#Pooling

cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))
#pool_size - tamanho da matriz que forma o pooled feature map
#strides - valor que a matriz que forma o pooled feature map anda para o lado

#Adding a second convolutional layer

cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))

#Flattening

#Transforma as matrizes em vetores
cnn.add(tf.keras.layers.Flatten())

#Full Connection

cnn.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))
#units - número de neurônios

#Output Layer

cnn.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN

cnn.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set

cnn.fit(x = training_set, validation_data = test_set, epochs = 25)

#Making a single prediction

import numpy as np
test_image = tf.keras.preprocessing.image.load_img('D:/Backup/Documents/Machine Learning A-Z (Codes and Datasets)/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/dataset/single_prediction/cat_or_dog_2.jpg', target_size = (64, 64))
test_image = tf.keras.preprocessing.image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
print("Índices = ",training_set.class_indices)
if result[0][0] == 1:
    prediction = 'dog'
if result[0][0] == 0:
    prediction = 'cat'

print(prediction)
