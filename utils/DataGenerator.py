import os

import tensorflow as tf

from utils import config


class DataGenerator:
    def __init__(self,
                 images_dir: str,
                 mask_dir: str, 
                 test_size=0.2,
                 image_size=config.IMAGE_SIZE,
                 map_default_process=True):
        """
        Class to generate dataset. It stores file paths to images and corresponding masks and they are loaded by a mapped
        function called DataGenerator.load_image()
        (see: https://www.tensorflow.org/api_docs/python/tf/data/Dataset).
        :param images_dir: Directory of images.
        :param mask_dir: Directory of corresponding mask.
        :param test_size: A value between 0.0 and 1.0, which represents percentage of the dataset sample
        used in the test set.
        :param map_default_process: A flag which maps loading images from the file paths and resizing them. If *False*
        one needs to apply their own, in order to use the dataset in training.
        """
        self.image_size = image_size

        if test_size > 1.0 or test_size < 0.0:
            raise ValueError

        # getting all filenames from given paths
        images_files = os.listdir(images_dir)
        mask_files = os.listdir(mask_dir)

        # concatenating with to get the full path
        images_fullpath = [os.path.join(images_dir, image_filepath) for image_filepath in images_files]
        masks_fullpath = [os.path.join(mask_dir, mask_filepath) for mask_filepath in mask_files]

        self.dataset_size = len(images_files)

        # splitting the dataset into
        train_size = 1.0 - test_size

        train_samples = int(train_size * self.dataset_size)
        self.images_train_paths = images_fullpath[:train_samples]
        self.masks_train_paths = masks_fullpath[:train_samples]

        self.images_test_paths = images_fullpath[train_samples:]
        self.masks_test_paths = masks_fullpath[train_samples:]

        # creating tensorflow Datasets
        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.images_train_paths, self.masks_train_paths))
        self.test_dataset = tf.data.Dataset.from_tensor_slices((self.images_test_paths, self.masks_test_paths))

        print("Found {} images.\n{} of them are in train set, and the rest {} are in the test set.".format(
            self.dataset_size,
            train_samples,
            self.dataset_size - train_samples))

        print("Default preprocessing mapped (resizing, loading)")
        if map_default_process:
            self._map_preprocess()

    def _map_preprocess(self):
        self.train_dataset = self.train_dataset.map(self.load_and_process, num_parallel_calls=tf.data.AUTOTUNE)
        self.test_dataset = self.test_dataset.map(self.load_and_process, num_parallel_calls=tf.data.AUTOTUNE)

    def load_and_process(self, image_filepath, mask_filepath):
        image, mask = self.load_image(image_filepath, mask_filepath)
        image, mask = self.resize(image, mask, self.image_size)

        return image, mask

    @staticmethod
    def resize(image, mask, size):
        """
        Resize and image to the desired size:
        :param image: Image to resize.
        :param mask: Mask to resize.
        :param size: A tuple with desired shape.
        :return:
        """
        image = tf.image.resize(image, size=size, method="nearest")
        mask = tf.image.resize(mask, size=size, method="nearest")

        return image, mask

    @staticmethod
    def augment(image):
        pass

    @staticmethod
    def load_image(image_filepath: str, mask_filepath: str):
        img = tf.io.read_file(image_filepath)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)

        mask = tf.io.read_file(mask_filepath)
        mask = tf.image.decode_jpeg(mask, channels=1)
        mask = tf.image.convert_image_dtype(mask, dtype=tf.float32)

        # WARNING: after this operation mask doesn't have to be BINARY!!
        # I think jpg compression causes a few boundary pixels and
        # they are instead of 1.0, 0.93
        return img, mask
