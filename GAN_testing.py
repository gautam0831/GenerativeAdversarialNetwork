#############################################
# TESTING #
#############################################
# Loading generator model and generating images
from keras.models import load_model
from numpy.random import randn
import matplotlib.pyplot as plt

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, num_samples):
  # Generate latent space points
  input_X = randn(latent_dim * num_samples)
  # Reshape into input batch for generator network
  input_X = input_X.reshape(num_samples, latent_dim)
  return input_X

# Plot the generated images
def create_plot(samples, n):
  # plot images
  for i in range(n * n):
    # Subplot
    plt.subplot(n, n, 1 + i)
    # Turn off axis
    plt.axis('off')
    # Plot pixel data
    plt.imshow(samples[i, :, :])
  plt.show()
 
# load model
model = load_model('generator_model_080.h5')
# generate images
latent_points = generate_latent_points(200, 100)
# generate images
X = model.predict(latent_points)
# scale from [-1,1] to [0,1]
X = (X + 1) / 2.0
# plot the result
create_plot(X, 5)
