import os

import tensorflow as tf

from utils import config


class DataGenerator:
    def __init__(self, images_dir: str, mask_dir: str, test_size=0.2):
        # getting all filenames from given paths
        images_files = os.listdir(images_dir)
        mask_files = os.listdir(mask_dir)

        # concatenating with to get the full path
        images_fullpath = [os.path.join(images_dir, image_filepath) for image_filepath in images_files]
        masks_fullpath = [os.path.join(mask_dir, mask_filepath) for mask_filepath in mask_files]

        self.dataset_size = len(images_files)

        # splitting the dataset into
        train_size = 1.0 - test_size

        train_samples = int(train_size*self.dataset_size)
        self.images_train_paths = images_fullpath[:train_samples]
        self.masks_train_paths = masks_fullpath[:train_samples]

        self.images_test_paths = images_fullpath[train_samples:]
        self.masks_test_paths = masks_fullpath[train_samples:]

        # creating tensorflow Datasets
        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.images_train_paths, self.masks_train_paths))
        self.test_dataset = tf.data.Dataset.from_tensor_slices((self.images_test_paths, self.masks_test_paths))

        print("Found {} images.\n{} of them are in train set, and the rest {} are in the test set.".format(self.dataset_size,
                                            train_samples,
                                            self.dataset_size-train_samples))

    def map_preprocess(self):
        self.train_dataset = self.train_dataset.map(self.load_and_process, num_parallel_calls=tf.data.AUTOTUNE)
        self.test_dataset = self.test_dataset.map(self.load_and_process, num_parallel_calls=tf.data.AUTOTUNE)

    def load_and_process(self, image_filepath, mask_filepath):
        image, mask = self.load_image(image_filepath, mask_filepath)
        image, mask = self.resize(image, mask)

        return image, mask

    @staticmethod
    def resize(image, mask):
        image = tf.image.resize(image, size=config.IMAGE_SIZE, method="nearest")
        mask = tf.image.resize(mask, size=config.IMAGE_SIZE, method="nearest")

        return image, mask

    @staticmethod
    def augment(image):
        pass

    @staticmethod
    def load_image(image_filepath: str, mask_filepath:str):
        img = tf.io.read_file(image_filepath)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)

        mask = tf.io.read_file(mask_filepath)
        mask = tf.image.decode_jpeg(mask, channels=1)
        mask = tf.image.convert_image_dtype(mask, dtype=tf.float32)

        # WARNING: after this operation mask is not BINARY!!
        # I think jpg compression causes a few boundary pixels and
        # they are instead of 1.0, 0.93
        # I gonna leave if for now,
        # if it will cause problems TODO: make a 0.5 threshold
        return img, mask


