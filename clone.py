import csv
import cv2
import os
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split

EPOCHS = 20

def read_samples(dirs = ['data']):
    samples = []
    for dir_ in dirs:
        with open(os.path.join(dir_, 'driving_log.csv')) as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)
            for line in reader:
                for i in range(3):
                    cur_sample = {}
                    filename = line[i].split('/')[-1]
                    cur_sample['filename'] = os.path.join(dir_, 'IMG', filename)
                    correction_d = 0.2
                    correction = 0
                    if i == 1: correction += correction_d
                    if i == 2: correction -= correction_d
                    cur_sample['angle'] = float(line[3]) + correction
                    cur_sample['flipped'] = False
                    samples.append(cur_sample)
    return samples

def augment_samples(samples):
    augmented_samples = list(samples)
    for sample in samples:
        cur_sample = {}
        cur_sample['filename'] = sample['filename']
        cur_sample['angle'] = sample['angle'] * -1.0
        cur_sample['flipped'] = True
        augmented_samples.append(cur_sample)
    return augmented_samples

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while True:
        np.random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                filename = batch_sample['filename']
                image = cv2.imread(filename)
                if batch_sample['flipped']:
                    image = cv2.flip(image, 1)
                angle = batch_sample['angle']
                images.append(image)
                angles.append(angle)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

samples = read_samples(['data', 'data2', 'data3'])
samples = augment_samples(samples)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D, MaxPooling2D, Dropout
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

model = Sequential()
model.add(Lambda(lambda x: x / 255. - .5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Dropout(.2))
model.add(Flatten())
model.add(Dense(200, activation="relu"))
model.add(Dense(20, activation="relu"))
model.add(Dense(5, activation="relu"))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
checkpointer = ModelCheckpoint(filepath='model.h5', verbose=1, \
        save_best_only=True)
# Use to continue learning from checkpointed or final model.
#model = load_model('model.h5')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), \
        validation_data=validation_generator, \
        nb_val_samples=len(validation_samples), nb_epoch=EPOCHS, \
        callbacks=[checkpointer])

model.save('final.h5')
