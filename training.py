from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from adversarial_networks import create_generator, create_discriminator, create_gan, INPUT_SIZE
from data import load_images, save_images, generate_noise
from image import denormalize_image
from plot import plot_images, plot_loss


PLOT_FRECUENCY = 50


def training(epochs=1, batch_size=32):
    #Loading Data
    x_train = load_images()
    batches = x_train.shape[0] / batch_size

    # Creating GAN
    generator = create_generator()
    discriminator = create_discriminator()
    gan = create_gan(generator, discriminator)
    
    # Adversarial Labels
    y_valid = np.ones(batch_size)*0.9
    y_fake = np.zeros(batch_size)
    discriminator_loss, generator_loss = [], []

    for epoch in range(1, epochs+1):
        print('-'*15, 'Epoch', epoch, '-'*15)
        g_loss = 0; d_loss = 0

        for _ in tqdm(range(int(batches))):
            # Random Noise and Images Set
            noise = generate_noise(batch_size)
            image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

            # Generate Fake Images
            generated_images = generator.predict(noise)
            
            # Train Discriminator (Fake and Real)
            discriminator.trainable = True
            d_valid_loss = discriminator.train_on_batch(image_batch, y_valid)
            d_fake_loss = discriminator.train_on_batch(generated_images, y_fake)            

            d_loss += (d_fake_loss + d_valid_loss)/2
            
            # Train Generator
            noise = generate_noise(batch_size)
            discriminator.trainable = False
            g_loss += gan.train_on_batch(noise, y_valid)
            
        discriminator_loss.append(d_loss/batches)
        generator_loss.append(g_loss/batches)
            
        if epoch % PLOT_FRECUENCY == 0:
            plot_images(epoch, generator)
            plot_loss(epoch, generator_loss, discriminator_loss)

    save_images(generator)


if __name__ == '__main__':
    training(epochs=200)
    