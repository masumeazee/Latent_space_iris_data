# VAE Iris
Have you ever thought about how it is possible to use the latent space of VAE as a sampling method 
and distribution model to organize data ?

## Description
This project demonstrates how the latent space of a Variational Autoencoder (VAE) can be used as an unsupervised clustering
method for the Iris dataset. The Iris dataset comprises measurements for four features across three species of iris flowers.
The goal is to show that the VAE can cluster the Iris data based on its underlying structure.

## How it Works
The first step is to load the Iris dataset and scale the data using MinMaxScaler. Then, the dataset is split into a training set
and a test set using the train_test_split function from scikit-learn.
Next, the dimensions of the input and latent space are defined. The input dimension is set to the number of features in the scaled
Iris dataset, which is 4 in this case. The latent dimension is set to 2.
After that, the sampling function is defined. This function is used to sample from the latent space using the mean and log variance
of the encoded input. It uses the reparameterization trick, which allows backpropagation to work through the sampling process.
Then, the encoder model is defined using the Keras functional API. It takes the input data and produces the mean and log variance
of the encoded input, as well as the encoded input itself.
The decoder model is also defined using the Keras functional API. It takes the encoded input and produces the reconstructed output.
The VAE model is then defined by combining the encoder and decoder models. The loss function is defined as a combination of the 
reconstruction loss and the KL divergence loss. The VAE model is compiled using the Adam optimizer and trained on the training set.
Once the VAE model is trained, the encoder is used to get the latent space representation of the test set. Then, a scatter plot of
the latent space and a scatter plot of the original test data are created using Matplotlib.
The scatter plot of the latent space shows how the different Iris species are clustered together in the latent space, 
even though the VAE was trained in an unsupervised manner. This demonstrates that the VAE has learned to cluster the Iris data 
based on its underlying structure.

## Dependencies
•	Python 3.7 or later
•	TensorFlow 2.x
•	NumPy
•	Matplotlib
•	scikit-learn
