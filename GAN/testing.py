from GAN.model import GAN
import matplotlib.pyplot as plt
import numpy as np

noise_vector_size = 256

generator = GAN.build_generator(noise_vector_size)
discriminator = GAN.build_discriminator()

print(discriminator.summary())
noise = np.random.randn(4, 256, 1)

imgs = generator.predict(noise)
print("SHAPE:", imgs[0].shape)

fakeness = discriminator.predict(imgs)

plt.imshow(imgs[0])
plt.show()
