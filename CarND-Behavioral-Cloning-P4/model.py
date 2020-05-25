### Borrowed heavily from the course
import argparse
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, AveragePooling2D, Conv2D, Dense, Dropout
from tensorflow.keras.layers import Flatten, Lambda, ReLU, GlobalAveragePooling2D, Cropping2D
from tensorflow.keras.callbacks import ModelCheckpoint
import cv2
import csv
import numpy as np
import os
from sklearn.model_selection import train_test_split
import glob
from random import uniform, Random
from sklearn.utils import shuffle
from math import ceil


def load_data(di, steering_correction=0.25, training_size=0.8, validation_size=0.1,
              test_size=0.1, seed=None):
    """ Function that finds all image filenames and corresponding driving data
    from the CSV files generated from the simulator given the directory of where
    the data was extracted to

    Args;
        di (str): Directory where the CSV files are located
        steering_correction (float): Steering correction to add for left and
                                     right camera images
        training_size (float): Fraction of the data to allocate to the training
                               dataset
        validation_size (float): Fraction of the data to allocate to the validation
                                 dataset
        test_size (float): Fraction of the data to allocate to the test dataset
        seed (int): Specify random seed for reproducibility.  Set to None to
                    perform no randomisation.

    Returns:
        A tuple of two lists containing the paths to the image data and
        the relevant vehicle data.  The first list is the training data
        and the second list is the validation data.
    """
    if training_size <= 0 or training_size >= 1:
        raise Exception("Training size must be between 0 and 1")
    if validation_size <= 0 or validation_size >= 1:
        raise Exception("Validation size must be between 0 and 1")
    if test_size <= 0 or test_size >= 1:
        raise Exception("Test size must be between 0 and 1")

    sum_size = training_size + validation_size + test_size
    training_size /= sum_size
    validation_size /= sum_size
    test_size /= sum_size

    csvfiles = glob.glob(di)

    # Contains every driving sample over all possible files
    samples = []

    # For each CSV filename...
    for fi in csvfiles:
        # Skip over the OLD files
        if fi.find("OLD") != -1:
            continue
        # Open up the CSV file
        with open(fi, 'r') as csvfile:
            # Get the directory where the CSV file is stored
            main_dir = os.path.split(fi)[0]

            # Iterate through every row of the CSV file...
            reader = csv.reader(csvfile)
            for line in reader:
                # If blank line, ignore
                if len(line) == 0:
                    continue
                # If the CSV file contains a header, ignore
                if line[0] == 'center':
                    continue

                # For the image paths, prepend with the full
                # path to where we can find it
                # Doing this for the centre, left and right images
                for i in range(3):
                    line[i] = os.path.join(main_dir, line[i].strip())

                # Directly add the image path and steering
                steering = float(line[3])
                # For the centre, left and right images, add the
                # corresponding corrective factor to ensure we gravitate
                # to the centre
                for j, val in enumerate([0, steering_correction, -steering_correction]):
                    samples.append((line[j], steering + val))

    # Get the test dataset first
    split_point = int((training_size + validation_size) * len(samples))
    test_samples = samples[split_point:]

    # Split the data up into train and validation
    if seed is not None:
        train_samples, validation_samples = train_test_split(samples[:split_point],
                                                            random_state=seed,
                                                            test_size=validation_size)
    else:
        split_point_train = int(training_size * len(samples))
        train_samples = samples[:split_point_train]
        validation_samples = samples[split_point_train:split_point]

    return train_samples, validation_samples, test_samples


def generator(samples, batch_size=32, prob_flip=0.3, seed=None):
    """ Generator to produce a batch of images for Keras

    Args:
        samples (list): List of samples (training or validation) from load_data
        batch_size (int): Desired batch size
        prob_flip (float): Probability of horizontally flipping an image
                           during training.  Specify None to skip.
        seed (int): Random seed for reproducibility.  Specify None for
                     default behaviour.
    """
    samples_copy = samples[:]
    num_samples = len(samples_copy)
    if seed is not None:
        r = Random(seed)
    while True: # Loop forever so the generator never terminates
        # Randomly shuffle the samples
        if seed is not None:
            r.shuffle(samples_copy)

        # For each batch...
        for i in range(0, num_samples, batch_size):
            # To send to the output
            X = []
            y = []

            # Get the batch
            batch = samples_copy[i : i + batch_size]

            # For each element in the batch...
            for filename, steering in batch:
                # Obtain the filename and read the image in
                img = cv2.imread(filename)
                # Reverse channels so they're in RGB format
                img = img[...,::-1]

                # Data augmentation - flip horizontally at random
                # If we flip horizontally, the steering also needs to be
                # corrected too
                if prob_flip is not None:
                    un = r.uniform(0.0, 1.0) if seed is not None else uniform(0.0, 1.0)
                    if un <= prob_flip:
                        img = np.fliplr(img)
                        steering = -steering

                # Add to the list
                X.append(img)
                y.append(steering)

            # Convert the lists to numpy arrays
            X = np.array(X)
            y = np.array(y)

            # Randomly shuffle this batch to be really sure then
            # send it off
            if seed is not None:
                X, y = shuffle(X, y, random_state=r.randint(0, 2**32 - 1))
            yield X, y


def define_model():
    """Define model architecture for inferring the steering angle given
    a front-facing camera image

    Returns:
        The Tensorflow model compiled using Adam with the default parameters
        and MSE as the loss
    """
    model = Sequential()
    model.add(Input(shape=(160, 320, 3), dtype='float32', name='input'))
    model.add(Lambda(lambda x: (x - 127.5) / 127.5, name='normalisation'))
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), name='cropping'))
    model.add(AveragePooling2D(pool_size=(2, 2), name='downsample'))
    model.add(Conv2D(16, kernel_size=(8, 8), strides=(4, 4), padding='same',
              name='conv1'))
    model.add(ReLU(name='relu1'))
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(2, 2), padding='same',
              name='conv2'))
    model.add(ReLU(name='relu2'))
    model.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding="same",
              name='conv3'))
    model.add(ReLU(name='relu3'))
    model.add(Flatten(name='flatten'))
    model.add(Dropout(0.5, name='dropout1'))
    model.add(Dense(512, name='fc1'))
    model.add(ReLU(name='relu4'))
    model.add(Dropout(0.5, name='dropout2'))
    model.add(Dense(1, name='output'))

    model.compile(optimizer="adam", loss="mse")

    return model


def main(data_dir, model_checkpoint_dir='./models', model_output_dir='./checkpoint',
         training_size=0.8, validation_size=0.1, test_size=0.1, steering_correction=0.25,
         prob_flip=0.3, batch_size=128, num_epochs=50, plot_loss=True, seed=42):
    """ Main function to run for the training

    Args:
        data_dir (str): Directory where driving data is stored
        model_checkpoint_dir (str): Directory where to save model checkpoints
        model_output_dir (str); Directory where to save final output model
        training_size (float): Fraction of the data to allocate to the training
                               dataset
        validation_size (float): Fraction of the data to allocate to the validation
                                 dataset
        test_size (float): Fraction of the data to allocate to the test dataset
        steering_correction (float): Steering correction to add for left and
                                     right camera images
        prob_flip (float): Probability for horizontally flipping an image (augmentation)
        batch_size (int): Batch size
        num_epochs (int): Total number of epochs
        plot_loss (bool): Plot the training and validation losses
        seed (int): Random seed for reproducibility

    Returns:
        A tuple of two NumPy arrays that record the training and validation loss
        at each epoch.  As a side-effect, the final model gets saved to
        model_output_dir in a file called steering.hdf5 for use with Tensorflow
        later
    """

    # Set to true if you want to see the loss plots
    if plot_loss:
        import matplotlib.pyplot as plt
        plt.rcParams['figure.figsize'] = (16, 12)

    # Get the training and validation data
    dir_path = os.path.join(data_dir, '**', '*.csv')
    train_samples, validation_samples, test_samples = load_data(dir_path,
                                        steering_correction=steering_correction,
                                        test_size=test_size, seed=seed)

    # Create generator objects
    train_generator = generator(train_samples, batch_size=batch_size,
                                prob_flip=prob_flip, seed=seed)
    # We must ensure that the probability of flipping for the validation
    # is disabled as we need a base of comparison for the loss.  Introducing
    # randomness loses our base of comparison.  We also don't need to
    # shuffle anything
    validation_generator = generator(validation_samples, batch_size=batch_size,
                                     prob_flip=None, seed=None)

    # Same situation applies with the test set
    test_generator = generator(test_samples, batch_size=batch_size,
                               prob_flip=None, seed=None)


    # Get the model
    model = define_model()

    # Model checkpoint - save the model with the lowest validation loss
    try:
        os.makedirs(model_checkpoint_dir)
    except OSError:
        pass
    filepath = os.path.join(model_checkpoint_dir, 'best_weights.hdf5')
    model_checkpoint_callback = ModelCheckpoint(filepath=filepath,
                                                save_weights_only=False,
                                                monitor='val_loss',
                                                mode='min',
                                                save_best_only=True)

    # Perform the training
    history = model.fit(x=train_generator, y=None,
                steps_per_epoch=ceil(len(train_samples) / batch_size),
                validation_data=validation_generator,
                validation_steps=ceil(len(validation_samples) / batch_size),
                epochs=num_epochs, verbose=1, callbacks=[model_checkpoint_callback])

    # Find the best model and save the full model
    model.load_weights(filepath)
    # Save the best model
    try:
        os.makedirs(model_output_dir)
    except OSError:
        pass
    model.save(os.path.join(model_output_dir, "steering.hdf5"))

    # Also evaluate the test dataset
    test_loss = model.evaluate(x=test_generator, y=None, batch_size=batch_size,
                               verbose=1, steps=ceil(len(test_samples) / batch_size))

    # Plot the training and validation loss over the epochs
    if plot_loss:
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model MSE loss')
        plt.ylabel('MSE Loss')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='upper right')
        plt.show()

    return history.history['loss'], history.history['val_loss'], test_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Behavioural Cloning Training')
    parser.add_argument("--data-dir", type=str, help="Directory where driving data is stored")
    parser.add_argument("--model-checkpoint-dir", type=str, default="./checkpoint", help="Directory where to save model checkpoints")
    parser.add_argument("--model-output-dir", type=str, default="./models", help="Directory where to save final output model")
    parser.add_argument("--training-size", type=float, default=0.8, help="Fraction of the data to allocate to the training set")
    parser.add_argument("--validation-size", type=float, default=0.1, help="Fraction of the data to allocate to the validation set")
    parser.add_argument("--test-size", type=float, default=0.1, help="Fraction of the data to allocate to the test set")
    parser.add_argument("--steering_correction", type=float, default=0.25, help="Steering correction to add for left and right camera images")
    parser.add_argument("--prob-flip", type=float, default=0.3, help="Probability for horizontally flipping an image (augmentation)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--plot-loss", action="store_true", help="Plot the training and validation losses")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    main(**args)