from tensorflow import keras
# *** VAE for Iris dataset ***
# *** Author: [Masume.Azz] ***

# Dependencies: TensorFlow, scikit-learn, NumPy, Matplotlib
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K

# Load the Iris dataset
iris = load_iris()

# Scale the dataset using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(iris.data)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, iris.target, test_size=0.2, random_state=42)

# Define the dimensions of the input and latent space
input_dim = X_scaled.shape[1]
latent_dim = 2

# Define the sampling function
def sampling(args):
    z_mean, z_log_var = args
    epsilon = keras.backend.random_normal(shape=(keras.backend.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=1.)
    return z_mean + keras.backend.exp(z_log_var / 2) * epsilon

# Define the encoder model
inputs = keras.Input(shape=(input_dim,))
x = layers.Dense(256, activation='relu')(inputs)
x = layers.Dense(128, activation='relu')(x)
z_mean = layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
z = layers.Lambda(sampling)([z_mean, z_log_var])
encoder = keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')

# Define the decoder model
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(128, activation='relu')(latent_inputs)
x = layers.Dense(256, activation='relu')(x)
outputs = layers.Dense(input_dim, activation='sigmoid')(x)
decoder = keras.Model(latent_inputs, outputs, name='decoder')

# Define the VAE model as a combination of the encoder and decoder
outputs = decoder(z)
vae = keras.Model(inputs, outputs, name='vae')

# Define the loss function as a combination of reconstruction loss and KL divergence loss
reconstruction_loss = keras.losses.mean_squared_error(inputs, outputs)
reconstruction_loss *= input_dim
kl_loss = 1 + z_log_var - keras.backend.square(z_mean) - keras.backend.exp(z_log_var)
kl_loss = keras.backend.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)

# Compile the VAE model
vae.compile(optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

# Train the VAE model
vae.fit(X_train, X_train, epochs=500, batch_size=64, validation_data=(X_test, X_test))

# Use the encoder to get the latent space representation of the test set
_, _, encoded_X_test = encoder.predict(X_test)

# Create a scatter plot of the latent space
plt.scatter(encoded_X_test[:, 0], encoded_X_test[:, 1], c=y_test)
plt.colorbar()
plt.show()

# Create a scatter plot of the original test data
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
plt.colorbar()
plt.show()
