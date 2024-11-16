from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
# Inisialisasi CNN
MesinKlasifikasi = Sequential()

# Langkah 1 - Convolution
MesinKlasifikasi.add(Input(shape=(128, 128, 3)))
MesinKlasifikasi.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))

# Langkah 2 - Pooling
MesinKlasifikasi.add(MaxPooling2D(pool_size=(2, 2)))

# Menambah convolutional layer
MesinKlasifikasi.add(Conv2D(32, (3, 3), activation='relu'))
MesinKlasifikasi.add(MaxPooling2D(pool_size=(2, 2)))

# Langkah 3 - Flattening
MesinKlasifikasi.add(Flatten())

# Langkah 4 - Full connection
MesinKlasifikasi.add(Dense(units=128, activation='relu'))
MesinKlasifikasi.add(Dense(units=1, activation='sigmoid'))

# Menjalankan CNN
MesinKlasifikasi.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Menjalankan CNN ke training dan test set
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('./dataset/training_set',
                                                 target_size=(128, 128),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('./dataset/test_set',
                                            target_size=(128, 128),
                                            batch_size=32,
                                            class_mode='binary')

# Mengganti fit_generator dengan fit
MesinKlasifikasi.fit(
    training_set,
    steps_per_epoch=8000 // 32,
    epochs=50,
    validation_data=test_set,
    validation_steps=2000 // 32
)



count_dog = 0
count_cat = 0
for i in range(4001, 5001): 
    test_image = image.load_img('dataset/test_set/dogs/dog.' + str(i) + '.jpg', target_size = (128, 128))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = MesinKlasifikasi.predict(test_image)
    training_set.class_indices
    if result[0][0] == 0:
        prediction = 'cat'
        count_cat = count_cat + 1
    else:
        prediction = 'dog'
        count_dog = count_dog + 1

# Mencetak hasil prediksinya agar bisa dibaca
print("count_dog:" + str(count_dog))    
print("count_cat:" + str(count_cat))