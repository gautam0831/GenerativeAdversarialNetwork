# Generative Adversarial Network
Building a GAN trained on the CIFAR-10 dataset for image generation

For this project, I used standard Python libraries and Tensorflow (Keras) to build a GAN model to generate realistic images that look similar to the data the model was trained on. I trained the model on the CIFAR-10 public dataset (50k training and 10k test images of 32x32 size in 10 classes). 

The approach was to begin with a defining a generator and a discriminator model, and subsequently stacking them to create a GAN. The discriminator’s job is to take in a 32x32x3 image (RGB) and outputting a label to predict if the image was real (1) or fake (0). This model consisted of 4 convolutional layers and 3 max pooling layers interspersed for downsampling. A final dense layer was added with dropout (to prevent overfitting) and sigmoid activation for binary classification. LeakyReLU was used as the activation function through the convolutional layers, which has been proven as a highly performing function for discriminator models. The final model was Adam-optimized and compiled with a binary crossentropy loss since this, again, was a binary classification task. 

Secondly, the generator model was defined to take as input the dimension of a point from the latent space (which was defined as 200-dim for this project) and output a 2D colorized, square image of 32x32 size. It is effective to have parallel feature maps to be learned from, so I created a dense layer with enough nodes for 256 4x4 feature maps. This was followed by 3 Conv2DTranspose layers with LeakyReLU activations whose goals were to combine upsampling and running a convolution in a single layer. The output layer was another convolutional layer with three filters (RGB), applying the tanh activation (proven highly for generators) to output a 32x32x3 image-like tensor.

Finally, I defined a GAN to take in the generator and discriminator to stack respectively. This is so that the points from the latent space enter the generator which creates a fake image, which is falsely labeled as “real” to trick the discriminator. The discriminator then takes this image and classifies it as real or fake, and the error from the prediction is used to update the weights and params of the generator to improve.

The training of the GAN was run for 180 epochs which was the max I was able to run given a limited use of GPU and RAM affordances on my free Google Colab account. The discriminator was trained twice per half-batch (one on real and one on fake samples) per batch. There were 390 batches per epoch with each batch containing 128 samples, slightly under the 50k samples in the dataset. Every 10 epochs the model was summarized and saved along with a grid plot of 100 generated images, which I analyzed qualitatively to note how the model improved with performance. There were major improvements from epoch 1 to 100, but observed a marginal diminishing performance growth over the next 80 epochs. 

The final results were more grainy and blurry than desired. This could be due to a need to train for more epochs or a change in the latent space dimensions. A larger dimensional latent space might be able to allow for cleaner results. Additionally, I could have added more complexity into the layers of the network to be deeper in architecture or updated the discriminator and generator models to utilize batch normalization, normalizing contributions to a layer for every mini-batch.
