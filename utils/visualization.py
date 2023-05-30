import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def display_learning_curves(history, figsize=(12,6)):
    acc = history.history["acc"]
    val_acc = history.history["val_acc"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    iou = history.history["binary_io_u"]
    val_iou = history.history["val_binary_io_u"]

    epochs_range = range(len(acc))

    fig = plt.figure(figsize=figsize)

    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, acc, label="train accuracy")
    plt.plot(epochs_range, val_acc, label="validataion accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")

    plt.subplot(1,3,2)
    plt.plot(epochs_range, loss, label="train loss")
    plt.plot(epochs_range, val_loss, label="validataion loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")

    plt.subplot(1,3,3)
    plt.plot(epochs_range, iou, label="train IoU")
    plt.plot(epochs_range, val_iou, label="validataion IoU")
    plt.title("Binary intersection over union")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.legend(loc="upper right")

    fig.tight_layout()
    plt.show()


def display(display_list):
    plt.figure(figsize=(15, 15))

    alpha = 1.0

    title = ["Input Image", "True Mask", "Predicted Mask", "Predicted Probabilities"]

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])

        plt.imshow(tf.keras.utils.array_to_img(display_list[i]), alpha=alpha)
        plt.axis("off")

    plt.show()


def count_values(array):
    counter = {}

    for value in np.unique(array):
        count = np.count_nonzero(array == value)
        counter[value] = count

    return counter


def white_balancing(image, percentile_value=99.9, MAX=255, min_counts=10, n_channels=3):
    balanced_image = np.zeros(image.shape)

    for channel_index in range(n_channels):
        channel = image[:, :, channel_index]

        # shifting towards zero
        value_counts = {value: np.count_nonzero(channel == value) for value in np.unique(channel)}

        min_value = 0
        for min_value, counts in value_counts.items():
            if counts > min_counts:
                break

        shifted_channel = channel - min_value
        
        # scaling to 255
        channel_max = np.percentile(shifted_channel, percentile_value)
        balanced_image[:, :, channel_index] = shifted_channel * (MAX/channel_max) 

    # values greater that MAX (usually 255) may occur after multiplying by (MAX/channel_max)
    saturated = np.where(balanced_image < MAX, balanced_image, MAX)

    # zero values may occur after - min_value, they are change for 0
    saturated = np.where(saturated > 0, saturated, 0)

    return saturated


def show_histograms_rgb(image, figsize=(15, 4)):
    fig,  axs= plt.subplots(1, 3, figsize=figsize)

    axs[0].hist(image[:, :, 0].ravel(), bins=256)
    axs[1].hist(image[:, :, 1].ravel(), bins=256)
    axs[2].hist(image[:, :, 2].ravel(), bins=256)

    axs[0].set_title("red")
    axs[1].set_title("green")
    axs[2].set_title("blue")
