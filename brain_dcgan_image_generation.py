import numpy as np
import matplotlib.pyplot as plt
from brain_mri_dcgan import GAN, build_generator, build_discriminator

latent_dim = 128
g_model = build_generator(latent_dim)

g_model.load_weights("saved_model/g_model.h5")

n_samples = 3
noise = np.random.normal(size=(n_samples, latent_dim))
examples = g_model.predict(noise)

_, axs = plt.subplots(1, 3, figsize=(1, 3))
axs = axs.flatten()
for img, ax in zip(examples, axs):
    ax.imshow(img)

plt.show()