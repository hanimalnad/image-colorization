import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, UpSampling2D, InputLayer
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence

class NumpyDataGenerator(Sequence):
    def __init__(self, gray_path, ab_paths, batch_size=32):
        self.gray_path = gray_path
        self.ab_paths = ab_paths
        self.batch_size = batch_size

        # Load just shapes (not full data) to get sample count
        self.gray_data = np.load(gray_path, mmap_mode='r')
        self.ab_data = [np.load(p, mmap_mode='r') for p in ab_paths]

        self.num_samples = self.gray_data.shape[0]
        self.indices = np.arange(self.num_samples)

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        gray_batch = self.gray_data[batch_idx].astype('float32') / 255.0

        # Find out which AB file the batch belongs to
        ab_batch = []
        offset = 0
        for ab_array in self.ab_data:
            count = ab_array.shape[0]
            local_idx = batch_idx[(batch_idx >= offset) & (batch_idx < offset + count)] - offset
            if local_idx.size > 0:
                ab_batch.append(ab_array[local_idx])
            offset += count

        ab_batch = np.concatenate(ab_batch, axis=0).astype('float32') / 128.0

        return gray_batch, ab_batch
# Constants
IMAGE_SIZE = 128  # Resize all images to this size

# Function to load grayscale and AB channel .npy files
def load_images_from_npy(data_dir):
    grayscale_images = []
    ab_images = []

    # Print contents for verification
    print("Files in grayscale directory:", os.listdir(os.path.join(data_dir, 'i')))
    print("Files in AB directory:", os.listdir(os.path.join(data_dir, 'ab', 'ab')))

    # Load grayscale
    gray_file_path = os.path.join(data_dir, 'i', 'gray_scale.npy')
    if os.path.exists(gray_file_path):
        gray_data = np.load(gray_file_path)
        grayscale_images.append(gray_data)
    else:
        print(f"Warning: {gray_file_path} not found!")

    # Load all ab files
    ab_dir = os.path.join(data_dir, 'ab', 'ab')
    ab_files = [f for f in os.listdir(ab_dir) if f.endswith('.npy')]
    for file in ab_files:
        ab_file_path = os.path.join(ab_dir, file)
        ab_data = np.load(ab_file_path)
        ab_images.append(ab_data)

    # Combine arrays
    grayscale_images = np.concatenate(grayscale_images, axis=0)
    ab_images = np.concatenate(ab_images, axis=0)

    print(f"Loaded {grayscale_images.shape[0]} grayscale images and {ab_images.shape[0]} AB images.")
    return grayscale_images, ab_images


# Preprocess grayscale and AB images
def preprocess_data(gray_images, ab_images):
    gray_images = gray_images.astype('float32') / 255.0
    ab_images = ab_images.astype('float32') / 128.0
    return gray_images, ab_images


# Build the CNN model for image colorization
def build_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)))  # Grayscale input (1 channel)
    
    # CNN layers for colorization
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))  # 2 channels (AB)

    model.compile(optimizer=Adam(), loss='mse')
    model.summary()
    return model

# Main function to train the model
def main():
    gray_path = 'data/train/i/gray_scale.npy'
    ab_paths = [
        'data/train/ab/ab/ab1.npy',
        'data/train/ab/ab/ab2.npy',
        'data/train/ab/ab/ab3.npy'
    ]

    if not os.path.exists(gray_path):
        print(f"Error: Grayscale file not found at {gray_path}")
        return

    for path in ab_paths:
        if not os.path.exists(path):
            print(f"Error: AB file not found at {path}")
            return

    # Initialize generator
    batch_size = 32
    data_gen = NumpyDataGenerator(gray_path, ab_paths, batch_size=batch_size)

    # Build the model
    model = build_model()

    # Train the model
    model.fit(data_gen, epochs=20)

    # Save the trained model
    model.save("colorization_model.h5")
    print("Model saved as 'colorization_model.h5'.")


if __name__ == "__main__":
    main()