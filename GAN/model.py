from abc import ABCMeta

import tensorflow as tf
from keras.layers import Dense, Conv2D, UpSampling2D, Input, Reshape, Flatten, Dropout
from keras.models import Model, Sequential


from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy


class GAN(Model):
    def __init__(self, generator, discriminator, noise_vector_size=256, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.generator = generator
        self.discriminator = discriminator

        self.gen_loss = None
        self.dis_loss = None
        self.gen_opt = None
        self.dis_opt = None

        self.noise_vector_size = noise_vector_size

    def compile(self, gen_lr=0.001, dis_lr=0.000001, *args, **kwargs):
        super().compile(*args, **kwargs)

        self.gen_opt = Adam(learning_rate=gen_lr)
        self.gen_loss = BinaryCrossentropy()

        self.dis_opt = Adam(learning_rate=dis_lr)
        self.dis_loss = BinaryCrossentropy()

    def train_step(self, batch):
        size = 16
        real_images = batch
        fake_images = self.generator(tf.random.normal((size, self.noise_vector_size, 1)), training=False)

        with tf.GradientTape() as dis_tape:
            img_real = self.discriminator(real_images, training=False)
            img_fake = self.discriminator(fake_images, training=False)
            imgs_real_fake = tf.concat([img_real, img_fake], axis=0)

            # creating corresponding labels
            y_real_fake = tf.concat([tf.zeros_like(img_real), tf.ones_like(img_fake)], axis=0)

            # adding noise
            noise_real = 0.15*tf.random.uniform(tf.shape(img_real))
            noise_fake = -0.15*tf.random.uniform(tf.shape(img_fake))
            y_real_fake += tf.concat([noise_real, noise_fake], axis=0)

            # computing loss
            total_dis_loss = self.dis_loss(y_real_fake, imgs_real_fake)

        dis_grad = dis_tape.gradient(total_dis_loss, self.discriminator.trainable_variables)
        self.dis_opt.apply_gradients(zip(dis_grad, self.discriminator.trainable_variables))

        with tf.GradientTape() as gen_tape:
            gen_img = self.generator(tf.random.normal((size, self.noise_vector_size, 1)), training=True)

            y_pred = self.discriminator(gen_img, training=False)

            total_gen_loss = self.gen_loss(tf.zeros_like(y_pred), y_pred)

        gen_grad = gen_tape.gradient(total_gen_loss, self.generator.trainable_variables)
        self.gen_opt.apply_gradients(zip(gen_grad, self.generator.trainable_variables))

        return {"dis_loss": total_dis_loss, "gen_loss": total_gen_loss}

    # TODO: implement variable number of layers
    @staticmethod
    def build_generator(noise_vector_size=256):
        # 2^n_layers = 16
        init_shape = int(noise_vector_size/16)

        # dense layer and reshaping for (init_shape, init_shape, noise_vector_size) (16, 16, 256)
        input_layer = Input(noise_vector_size)
        dense = Dense(noise_vector_size * init_shape * init_shape, )(input_layer)
        reshape = Reshape((init_shape, init_shape, noise_vector_size))(dense)

        # upsampling shape = (32, 32, 256)
        upsample_1 = UpSampling2D(size=(2, 2))(reshape)
        conv2d_1 = Conv2D(noise_vector_size, kernel_size=5, activation="relu", padding="same")(upsample_1)

        # upsampling, shape = (64, 64, 128)
        upsample_2 = UpSampling2D(size=(2, 2))(conv2d_1)
        conv2d_2 = Conv2D(noise_vector_size/2, kernel_size=4, activation="relu", padding="same")(upsample_2)

        # upsampling, shape = (128, 128, 64)
        upsample_3 = UpSampling2D(size=(2, 2))(conv2d_2)
        conv2d_3 = Conv2D(noise_vector_size/4, kernel_size=4, activation="relu", padding="same")(upsample_3)
        
        # upsampling, shape = (256, 256, 64)
        upsample_4 = UpSampling2D(size=(2, 2))(conv2d_3)
        conv2d_4 = Conv2D(noise_vector_size/4, kernel_size=3, activation="relu", padding="same")(upsample_4)

        # output
        # CHANGE IN CASE OF 256, 256
        output_layer = Conv2D(3, kernel_size=3, activation="sigmoid", padding="same")(conv2d_2)

        return Model(input_layer, output_layer)

    @staticmethod
    def build_discriminator(input_shape=(64, 64, 3)):
        model = Sequential()

        # First Conv Block
        model.add(Conv2D(32, 5, activation="relu", input_shape=input_shape))
        model.add(Dropout(0.4))

        # Second Conv Block
        model.add(Conv2D(64, 5, activation="relu"))
        model.add(Dropout(0.4))

        # Third Conv Block
        model.add(Conv2D(128, 5, activation="relu"))
        model.add(Dropout(0.4))

        # Flatten then pass to dense layer
        model.add(Flatten())
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))

        return model

    @staticmethod
    def build_transfer_learning_discriminator(input_shape=(64, 64, 3), pretrained_model=None, freeze_pretrained_layers=False):
        if pretrained_model is None:
            input_layer = Input(shape=input_shape)

            # Convolutional layer 1
            conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
            dropout1 = Dropout(0.25)(conv1)

            # Convolutional layer 2
            conv2 = Conv2D(64, (3, 3), activation='relu')(dropout1)
            dropout2 = Dropout(0.25)(conv2)

            # Convolutional layer 3
            conv3 = Conv2D(128, (3, 3), activation='relu')(dropout2)
            dropout3 = Dropout(0.25)(conv3)

            # Convolutional layer 4
            conv4 = Conv2D(256, (3, 3), activation='relu')(dropout3)
            dropout4 = Dropout(0.25)(conv4)

            # Flatten the output from the previous layer
            flatten = Flatten()(dropout4)

        else:
            # freezing
            if freeze_pretrained_layers:
                for layer in pretrained_model.layers:
                    layer.trainable = False

            input_layer = pretrained_model.input

            flatten = Flatten()(pretrained_model.output)

        dense_last = Dense(128, activation='relu')(flatten)
        output_layer = Dense(1, activation='sigmoid')(dense_last)

        return Model(input_layer, output_layer)



