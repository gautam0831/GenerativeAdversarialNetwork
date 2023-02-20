from numpy import expand_dims
from numpy import ones
from numpy import zeros
from numpy.random import rand
from numpy.random import randn
from numpy.random import randint

from keras.datasets.cifar10 import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.layers import MaxPool2D
from keras.layers import Reshape
from keras.layers import Conv2DTranspose
from keras.utils.vis_utils import plot_model

import matplotlib.pyplot as plt

# Define the baseline discriminator model
# Model has a normal Conv layer followed by 3 downsampling (via stride) Conv layers
# LeakyReLU has shown promising results as an activation fn for GANs 

def define_discriminator(in_shape=(32,32,3)):
  '''
  Input will be a 32x32x3 image (with 3 for RGB channel)
  Output will be a binary classification label of 1 (real) or 0 (fake)
  '''
  model = Sequential()
  model.add(Conv2D(64, (3,3), padding='same', input_shape=in_shape))
  model.add(LeakyReLU(alpha=0.2))
  # Add convolution layer and downsample
  model.add(Conv2D(128, (3,3), padding='same'))
  model.add(MaxPool2D(pool_size=(2,2), padding='same'))
  model.add(LeakyReLU(alpha=0.2))
  
  # Add convolution layer and downsample
  model.add(Conv2D(128, (3,3), padding='same'))
  model.add(MaxPool2D(pool_size=(2,2), padding='same'))
  model.add(LeakyReLU(alpha=0.2))
 
  # Add convolutional layer and downsample
  model.add(Conv2D(256, (3,3), padding='same'))
  model.add(MaxPool2D(pool_size=(2,2), padding='same'))
  model.add(LeakyReLU(alpha=0.2))

  # Final layer
  model.add(Flatten())
  # Added dropout to minimize overfitting
  model.add(Dropout(0.4))
  model.add(Dense(1, activation='sigmoid'))
  # compile model
  opt = Adam(lr=0.0002, beta_1=0.5)
  model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
  return model

# Define generator model
def define_generator(latent_dim):
  '''
  Input is the dimension of a point from the latent space
  Output is a 2-dimensional, colored, square image of 32x32 with pixel values [-1,1]
  '''
  model = Sequential()
  # Allow enough nodes for 256 4x4 feature maps
  num_nodes = 256 * 4 * 4
  model.add(Dense(num_nodes, input_dim=latent_dim))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Reshape((4, 4, 256)))
  # Upsample to 8x8 using Conv2DTranspose and stride 2x2 quadruples area of map
  model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
  model.add(LeakyReLU(alpha=0.2))
  # Upsample to 16x16
  model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
  model.add(LeakyReLU(alpha=0.2))
  # Upsample to 32x32
  model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
  model.add(LeakyReLU(alpha=0.2))
  # Output layer with tanh activation which has high proven performance for generators
  model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))
  return model


#Create GAN model stacking generator and discriminator
#Input points from latent space --> generator --> discriminator
# --> updates weights of generator

def define_GAN(gen_model, disc_model):
  # Make discriminator weights untrainable affecting only the weights seen by 
  # the GAN model, not the standalone discriminator 
  disc_model.trainable = False
  #Stack models
  model = Sequential()
  model.add(gen_model)
  model.add(disc_model)

  #Compile model
  opt = Adam(lr=0.0002, beta_1=0.5)
  model.compile(loss='binary_crossentropy', optimizer=opt)
  return model


# Load CIFAR-10 training images
def load_real_samples():
  # Load data from the dataset
 (trainX, _), (_, _) = load_data()
 # convert to floats
 X = trainX.astype('float32')
 # scale from [0,255] to [-1,1] centering around 0 now instead of 127.5
 X = (X - 127.5) / 127.5
 return X

# Generate num_samples latent points 
def gen_latent(latent_dim, num_samples):
  # Generate random points
  X = randn(latent_dim * num_samples)
  # Reshape into appropriate dimensions
  X = X.reshape(num_samples, latent_dim)
  return X

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(gen_model, latent_dim, num_samples):
 # generate points in latent space
 inputX = gen_latent(latent_dim, num_samples)
 # Use generator to predict output
 X = gen_model.predict(inputX)
 # Create 'fake' class labels
 y = zeros((num_samples, 1))
 return X, y
 
# Generate real samples from dataset
def generate_real_samples(dataset, num_samples):
 # Random selection of indices
 ix = randint(0, dataset.shape[0], num_samples)
 # Obtain selected images
 X = dataset[ix]
 # Generate class labels for these real images (1)
 y = ones((num_samples, 1))
 return X, y
 
# Creating a plot of generated images
def create_image_plot(samples, epoch, n=7):
  # Re-scale pixel values from [-1,1] to [0,1]
  samples = (samples + 1) / 2.0
  # plot images
  for i in range(n * n):
    # Subplot
    plt.subplot(n, n, 1 + i)
    # No axis
    plt.axis('off')
    # Plot pixel data
    plt.imshow(samples[i])
  # save plot to file
  file_nm = 'generated_image_plot_e%03d.png' % (epoch+1)
  plt.savefig(file_nm)
  plt.close()

#Summarize performance of discriminator model for later analysis
def summarize_performance(epoch, gen_model, disc_model, data, latent_dim, num_samples=150):
  # Generate real samples
  X_real, y_real = generate_real_samples(data, num_samples)
  # Evaluate discriminator on real samples
  _, acc_real = disc_model.evaluate(X_real, y_real, verbose=0)
  # Generate fake samples
  X_fake, y_fake = generate_fake_samples(gen_model, latent_dim, num_samples)
  # Evaluate discriminator on fake samples
  _, acc_fake = disc_model.evaluate(X_fake, y_fake, verbose=0)
  # Summarize performance of discriminator
  print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))

  # Save the generator model tile file
  file_nm = 'generator_model_%03d.h5' % (epoch+1)
  gen_model.save(file_nm)

  # Create and save plot
  create_image_plot(X_fake, epoch)


# Train GAN model
# Update discriminator on real and fake samples and then update generator through 
# the combined model
def train_GAN(gen_model, disc_model, GAN_model, data, latent_dim, num_epochs=180, num_batch=128):
  half_batch = int(num_batch / 2)
  batch_per_epoch = int(data.shape[0]/num_batch)
  
  for i in range(num_epochs):
    for j in range(batch_per_epoch):
      # Generate real samples
      X_real, y_real = generate_real_samples(data, half_batch)
      # Update discriminator on real samples
      disc_loss1, _ = disc_model.train_on_batch(X_real, y_real)
      # Generate fake samples
      X_fake, y_fake = generate_fake_samples(gen_model, latent_dim, half_batch)
      # Update discriminator on fake samples
      disc_loss2, _ = disc_model.train_on_batch(X_fake, y_fake)

      # Obtain latent points to pass into generator
      X_GAN = gen_latent(latent_dim, num_batch)
      # Label all fake images 1 (inverted labels) so discriminator can think the generated images are real
      y_GAN = ones((num_batch, 1))
      loss = GAN_model.train_on_batch(X_GAN, y_GAN)

      # Summarize loss on this batch
      print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
      (i+1, j+1, batch_per_epoch, disc_loss1, disc_loss2, loss))
  
    # Evaluate GAN model performance every 10 epochs
    if (i+1) % 10 == 0:
      summarize_performance(i, gen_model, disc_model, data, latent_dim)


#############################################
# TRAINING #
#############################################

#Size of latent space
latent_dim = 200

#Define generator
gen_model = define_generator(latent_dim)

#Define discriminator
disc_model = define_discriminator()

#Define GAN
GAN_model = define_GAN(gen_model, disc_model)

#Load data
data = load_real_samples()

#Train GAN model
train_GAN(gen_model, disc_model, GAN_model, data, latent_dim)
