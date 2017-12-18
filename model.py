from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from sklearn.model_selection import train_test_split
import random
import sklearn
import cv2
import numpy as np
import csv

dataset_dir = "../p3/t1data/"
img_dir = dataset_dir + "IMG/"

# Offsets for camera images: center left right
img_adjust = [0.0, +0.25, -0.25]

nb_epochs = 10
model_name = "model.h5"

# Referencing Nvidia Model Pipeline from https://arxiv.org/pdf/1604.07316v1.pdf
def Nvidia_Model():
    model = Sequential()

    # Normalize
    model.add(Lambda(lambda x: x / 127.5 - 1, 
        input_shape=(160, 320, 3), output_shape=(160,320,3)))
    # Crop
    model.add(Cropping2D(cropping=((70, 25), (0,0))))

    # Conv Layers
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Flatten())

    # Dropout
    model.add(Dropout(0.5))

    # Fully Connected Layers
    model.add(Dense(100, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1))

    return model

def load_image_ref():
    lines = []
    with open(dataset_dir + "driving_log.csv") as log_file:
        reader = csv.reader(log_file)
        for line in reader:
            lines.append(line)
    return lines[1:]

def augment_img_flip(img, angle):
    perform_flip = np.random.randint(2)
    if perform_flip == 0:
        return img, angle
    return cv2.flip(img, 1), -angle

def augment_img_shift(img, angle):
    rand_hshift = np.random.uniform(low=-60, high=60)
    rand_vshift = np.random.uniform(low=-20, high=20)
    M = np.float32([[1, 0, rand_hshift], [0, 1, rand_vshift]])
    rows, cols, depth = img.shape
    image = cv2.warpAffine(img, M, (cols, rows))
    angle += rand_hshift * 0.3 / 60
    return image, angle

def augment_img_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = v.astype(np.float64)
    v *= np.random.uniform(low=0.5, high=1.5)
    v[v > 255] = 255
    v = v.astype(np.uint8)
    hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def train_generator(samples, batch_size):
    num_samples = len(samples)
    while True: 
        sklearn.utils.shuffle(samples)
        for i in range(0, num_samples, batch_size):
            batch_samples = samples[i:i+batch_size]

            images = []
            angles = []

            for batch_sample in batch_samples:
                img_selection = np.random.randint(3)

                img_path = batch_sample[img_selection]
                img_name = img_dir + img_path.split('/')[-1]
                image = cv2.imread(img_name)
                center_angle = float(batch_sample[3])
                angle = center_angle + img_adjust[img_selection]

                image, angle = augment_img_flip(image, angle)
                image, angle = augment_img_shift(image, angle)
                image = augment_img_brightness(image)

                images.append(image)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def valid_generator(samples, batch_size):
    num_samples = len(samples)
    while True: 
        sklearn.utils.shuffle(samples)

        for i in range(0, num_samples, batch_size):
            batch_samples = samples[i:i+batch_size]

            images = []
            angles = []

            for batch_sample in batch_samples:
                img_selection = np.random.randint(3)
                img_path = batch_sample[img_selection]
                img_name = img_dir + img_path.split('/')[-1]
                image = cv2.imread(img_name)
                center_angle = float(batch_sample[3])
                angle = center_angle + img_adjust[img_selection]

                images.append(image)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train, y_train)


if __name__ == "__main__":
    samples = load_image_ref()
    print("Sample Size: {}".format(len(samples)))

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    train_generator = train_generator(train_samples, batch_size=256)
    validation_generator = valid_generator(validation_samples, batch_size=32)
    
    samples_per_epoch = len(train_samples)
    nb_val_samples = len(validation_samples)

    model = Nvidia_Model()

    # Adam Optimizer 
    model.compile(loss="mse", optimizer="adam")

    print("Sample Split: train_samples={} validation_samples={}".format(
        samples_per_epoch,nb_val_samples))

    model.fit_generator( train_generator, 
        samples_per_epoch = samples_per_epoch, 
        validation_data = validation_generator,
        nb_val_samples = nb_val_samples, 
        nb_epoch = nb_epochs)

    print("Saving model: {}".format(model_name))
    model.save(model_name)

    model.summary()

    print("Done.")