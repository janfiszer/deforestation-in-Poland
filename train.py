import os
import pickle

import tensorflow as tf

from utils import DataGenerator, config
from model import UNet

# def intersection_over_union(y_true, y_pred):
#     tp = metrics.TruePositives()
#     fp = metrics.FalsePositives()
#     fn = metrics.FalseNegatives()
#
#     tp_count = tp.update_state(y_true, y_pred).result()
#     fp_count = fp.update_state(y_true, y_pred).result()
#     fn_count = fn.update_state(y_true, y_pred).result()
#
#     return tp_count.numpy()/(tp_count.numpy() + fp_count.numpy(), fn_count.numpy())


def create_data_gen():
    masks_dir = os.path.join(config.DATASET_DIR, "masks")
    images_dir = os.path.join(config.DATASET_DIR, "images")

    data_gen = DataGenerator.DataGenerator(images_dir, masks_dir)

    return data_gen


def create_train_test_sets(data_gen):
    train_batches = data_gen.train_dataset.cache().shuffle(config.BUFFER_SIZE).batch(config.BATCH_SIZE)
    train_batches = train_batches.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    test_batches = data_gen.test_dataset.cache().shuffle(config.BUFFER_SIZE).batch(config.BATCH_SIZE)
    test_batches = test_batches.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_batches, test_batches


def train_model(model, train_batches, test_batches):
    # defining loss function and metrics
    loss = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    intersection_over_union = tf.keras.metrics.BinaryIoU()

    # compiling model
    model.compile(loss=loss, optimizer=optimizer, metrics=["acc", intersection_over_union])
    # training model
    training = model.fit(train_batches, validation_data=test_batches, epochs=config.EPOCHS)

    return model, training


def save_model_and_logs(model, training, test_batches):
    eval_results = model.evaluate(test_batches)
    print()

    try:
        model_filepath = f"{config.TRAINED_MODEL_DIR}/trained-iou{int(eval_results[2] * 100)}.h5"
        model.save(model_filepath)
        print(f"Model successfully saved to file: {model_filepath}")
    except TypeError:
        print("Didn't manage to save the model due to an exception: ", TypeError)

    try:
        history_filepath = f"{config.TRAINED_MODEL_DIR}/training-history-iou{int(eval_results[2] * 100)}.pkl"
        with open(history_filepath, 'wb') as file:
            pickle.dump(training, file)
        print(f"Training successfully saved to file: {history_filepath}")
    except FileNotFoundError:
        print("Didn't manage to training history of the model due to an exception: ", FileNotFoundError)


def print_gap():
    print()
    print()


def main():
    # loading data and creating train and test set
    print("CREATING DATASET:\n")
    data_gen = create_data_gen()
    train_batches, test_batches = create_train_test_sets(data_gen)
    print_gap()

    print("CREATING MODEL\n")
    # creating model
    model = UNet()
    model.summary()
    print_gap()

    print("TRAINING THE MODEL\n")
    model, training = train_model(model, train_batches, test_batches)
    print_gap()

    print("EVALUATING AND SAVING THE MODEL AND LOGS")
    save_model_and_logs(model, training, test_batches)


if __name__ == "__main__":
    main()
