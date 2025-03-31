import os
import pickle

import tensorflow as tf

from GAN.model import GAN
from utils import config, DataGenerator

from keras.applications.resnet import ResNet50, preprocess_input


def load_and_resize(image_filepath):
    img = tf.io.read_file(image_filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)

    img = tf.image.resize(img, size=config.IMAGE_SIZE, method="nearest")

    # applying resnet preprocess
    img = preprocess_input(img)

    return img


def print_gap():
    print()
    print()


def main():
    print("CREATING DATASET:\n")
    images_dir = os.path.join(config.DATASET_DIR, "images")

    # the test set is not usable
    # but still we don't want to take the whole dataset due to  computational expense 
    # so to reduce the dataset size the test_size parameter is adjusted
    data_gen = DataGenerator.DataGenerator(images_dir, map_default_process=False, test_size=0.5)

    train_dataset = data_gen.train_dataset.map(load_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
    train_batches = train_dataset.cache().shuffle(config.BUFFER_SIZE).batch(config.BATCH_SIZE)
    train_batches = train_batches.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    print_gap()

    print("CREATING MODEL\n")
    gen = GAN.build_generator()

    pretrained_resnet = ResNet50(input_shape=config.IMAGE_SHAPE, include_top=False, weights='imagenet')
    dis = GAN.build_transfer_learning_discriminator(pretrained_model=pretrained_resnet, input_shape=config.IMAGE_SIZE)
    # dis = GAN.build_discriminator(input_shape=config.IMAGE_SHAPE)

    gan = GAN(gen, dis)

    # gen_opt = Adam(learning_rate=0.0001)
    # gen_loss = BinaryCrossentropy()

    # dis_opt = Adam(learning_rate=0.000005)
    # dis_loss = BinaryCrossentropy()
    # gan.compile(gen_opt, dis_opt, gen_loss, dis_loss)
    gan.compile(gen_lr=0.001, dis_lr=0.0000001)
    # gan.build()

    if config.PRINT_GAN_SUMMARY:
        print("GENERATOR:")
        print(gen.summary())
        print_gap()

        print("DISCRIMINATOR:")
        print(dis.summary())
        print_gap()

    print("TRAINING THE MODEL\n")
    training = gan.fit(train_batches, epochs=8)
    print_gap()

    print("EVALUATING AND SAVING THE MODEL AND LOGS")
    try:
        model_filepath = f"trained_model/first_gen.h5"
        gan.generator.save(model_filepath)
        gan.discriminator.save(f"trained_model/first_dis.h5")

        print(f"Model successfully saved to file: {model_filepath}")
    except TypeError:
        print("Didn't manage to save the model due to an exception: ", TypeError)

    try:
        history_filepath = f"trained_model/training-history-first-gan.pkl"
        with open(history_filepath, 'wb') as file:
            pickle.dump(training.history, file)
        print(f"Training successfully saved to file: {history_filepath}")
    except FileNotFoundError:
        print("Didn't manage to training history of the model due to an exception: ", FileNotFoundError)


if __name__ == "__main__":
    main()


