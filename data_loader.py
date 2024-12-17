import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(train_dir, test_dir, img_height=48, img_width=48, batch_size=32):
    """
    Load training and test data from directories.
    """
    datagen = ImageDataGenerator(rescale=1./255)

    train_data = datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode='grayscale',  # Grayscale images
        class_mode='categorical'
    )

    test_data = datagen.flow_from_directory(
        test_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode='grayscale',  # Grayscale images
        class_mode='categorical'
    )

    return train_data, test_data
