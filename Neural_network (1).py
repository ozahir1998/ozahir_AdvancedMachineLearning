#!/usr/bin/env python
# coding: utf-8

# # Advanced Machine Learning
# 
# # Assignment 1: Neural Networks
# 
# # Osama Bin Zahir

# In[113]:


get_ipython().run_cell_magic('capture', '', '# Installing required packages \n!pip install tensorflow\n!pip install tensorflow-datasets\n')


# In[114]:


(imdb_a_train, imdb_b_train), (imdb_a_test, imdb_b_test) = imdb.load_data(num_words=10000)


# In[115]:


# Importing required Libraries

import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# In[116]:


max([max(sequence) for sequence in imdb_a_train])


# In[117]:


# Preparing the data for the model

# Retrieving a dictionary mapping words to their index in the IMDB dataset
word_index = imdb.get_word_index()

# Creating a reverse dictionary and mapping the indices back to the original words
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()])
# Create models with different configurations
# Decoding reviews from the IMDB dataset
decoded_review = " ".join([reverse_word_index.get(i - 3, "?") for i in imdb_a_train[0]])


# In[118]:


# Encoding the integer sequences via multi-hot encoding

import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results
imdb_train = vectorize_sequences(imdb_a_train)
imdb_test = vectorize_sequences(imdb_a_test)


# In[119]:


imdb_train[0]


# In[120]:


value_train = np.asarray(imdb_b_train).astype("float32")
value_test = np.asarray(imdb_b_test).astype("float32")


# In[121]:


# Create models with different configurations
# One hidden layer and 32 hidden units using Tanh activation function instead of relu

model_one_hidden_layer = Sequential()
model_one_hidden_layer.add(Dense(32, activation='tanh'))
model_one_hidden_layer.add(Dense(1, activation='sigmoid'))


# In[122]:


# Compiling model using MSE

model_one_hidden_layer.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])


# In[123]:


# Validation

a_val = imdb_train[:10000]
partial_a_train = imdb_train[10000:]
b_val = value_train[:10000]
partial_b_train = value_train[10000:]


# In[124]:


# Training the model with validation set

history_one_hidden_layer = model_one_hidden_layer.fit(partial_a_train, 
                                                      partial_b_train, 
                                                      epochs=20, 
                                                      batch_size=512, 
                                                      validation_data=(a_val,b_val))


# In[125]:


history_dict = history_one_hidden_layer.history
history_dict.keys()


# In[126]:


# Plotting the training and validation loss

import matplotlib.pyplot as plt
history_dict = history_one_hidden_layer.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# In[127]:


# Plotting the training and validation accuracy

plt.clf()
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# In[128]:


results = model_one_hidden_layer.evaluate(imdb_test,value_test)


# In[129]:


# Adding Dropout Layer & Regulaizers

from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense
from tensorflow.keras import regularizers
from keras.layers import Dropout


model_one_hidden_layer = Sequential()
model_one_hidden_layer.add(Dense(32, activation='tanh', activity_regularizer=regularizers.L2(0.01)))
model_one_hidden_layer.add(Dropout(0.5))
model_one_hidden_layer.add(Dense(1, activation='sigmoid'))

# compiling the model

model_one_hidden_layer.compile(optimizer="adam",
                              loss="mean_squared_error",
                              metrics=["accuracy"])

# Validating the model

a_val = imdb_train[:10000]
partial_a_train = imdb_train[10000:]
b_val = value_train[:10000]
partial_b_train = value_train[10000:]

history = model_one_hidden_layer.fit(partial_a_train, 
                                     partial_b_train, 
                                     epochs=20, 
                                     batch_size=512, 
                                     validation_data=(a_val,b_val))


# In[130]:


plt.clf()
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# In[131]:


results = model_one_hidden_layer.evaluate(imdb_test,value_test)


# In[132]:


# Adding more hidden layers to examine how it affects the accuracy. Here three hidden layers are used with 32 hidden units. 

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers

model_three_hidden_layers = Sequential()
model_three_hidden_layers.add(Dense(32, activation='tanh', activity_regularizer=regularizers.L2(0.01)))
model_three_hidden_layers.add(Dropout(0.5))
model_three_hidden_layers.add(Dense(32, activation='tanh', activity_regularizer=regularizers.L2(0.01)))  # Second hidden layer
model_three_hidden_layers.add(Dropout(0.5))
model_three_hidden_layers.add(Dense(32, activation='tanh', activity_regularizer=regularizers.L2(0.01)))  # Third hidden layer
model_three_hidden_layers.add(Dropout(0.5))
model_three_hidden_layers.add(Dense(1, activation='sigmoid'))  # Output layer

# Compliling the model

model_three_hidden_layers.compile(optimizer="adam",
                                loss="mean_squared_error",
                                metrics=["accuracy"])
# Data Splitting

a_val = imdb_train[:10000]
partial_a_train = imdb_train[10000:]
b_val = value_train[:10000]
partial_b_train = value_train[10000:]

history = model_three_hidden_layers.fit(partial_a_train, 
                                     partial_b_train, 
                                     epochs=20, 
                                     batch_size=512, 
                                     validation_data=(a_val,b_val))


# In[133]:


plt.clf()
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# In[134]:


results = model_three_hidden_layers.evaluate(imdb_test,value_test)


# In[135]:


# Building a model with 64 hidden units and one hidden layer to examine the accuracy

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers

model_one_hidden_layer_64_units = Sequential()
model_one_hidden_layer_64_units.add(Dense(64, activation='tanh', activity_regularizer=regularizers.L2(0.01)))  # Hidden layer with 64 units
model_one_hidden_layer_64_units.add(Dropout(0.5))
model_one_hidden_layer_64_units.add(Dense(1, activation='sigmoid'))  # Output layer

# Compliling the model

model_one_hidden_layer_64_units.compile(optimizer="adam",
                                loss="mean_squared_error",
                                metrics=["accuracy"])
# Data Splitting

a_val = imdb_train[:10000]
partial_a_train = imdb_train[10000:]
b_val = value_train[:10000]
partial_b_train = value_train[10000:]

history = model_one_hidden_layer_64_units.fit(partial_a_train, 
                                     partial_b_train, 
                                     epochs=20, 
                                     batch_size=512, 
                                     validation_data=(a_val,b_val))


# In[136]:


plt.clf()
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# In[137]:


results =  model_one_hidden_layer_64_units.evaluate(imdb_test,value_test)


# 
# 
# * Neural Network Configurations for IMDB Sentiment Analysis 
# 
# * Introduction
# 
# This report presents the results of training several neural network models with different configurations for sentiment analysis on the IMDB dataset. The goal is to determine the optimal model architecture for accurately classifying movie reviews as positive or negative.
# 
# * Experimental Setup
# 
# Dataset: The IMDB dataset consists of 50,000 movie reviews labeled as positive or negative.
# Model Configurations: Four different neural network configurations were evaluated:
# One hidden layer with 32 units and Tanh activation function
# One hidden layer with 32 units, Tanh activation function, and dropout regularization
# Three hidden layers with 32 units each, Tanh activation function, and dropout regularization
# One hidden layer with 64 units, Tanh activation function, and dropout regularization
# 
# * Results and Analysis
# 
# * Model 1: One Hidden Layer (32 Units, Tanh Activation)
# Accuracy: 85.54%
# Validation Accuracy: 86.60%
# Observations: The model shows a good performance with decent accuracy on both training and validation sets. However, there is a slight overfitting as the training accuracy is slightly higher than the validation accuracy.
# 
# * Model 2: One Hidden Layer (32 Units, Tanh Activation, Dropout Regularization)
# Accuracy: 81.88%
# Validation Accuracy: 83.29%
# Observations: Introducing dropout regularization slightly reduces overfitting compared to Model 1, but there is still room for improvement in validation accuracy.
# 
# * Model 3: Three Hidden Layers (32 Units, Tanh Activation, Dropout Regularization)
# Accuracy: 81.14%
# Validation Accuracy: 83.04%
# Observations: Adding more hidden layers does not significantly improve performance. The model seems to suffer from overfitting, similar to Model 2.
# 
# * Model 4: One Hidden Layer (64 Units, Tanh Activation, Dropout Regularization)
# Accuracy: 80.02%
# Validation Accuracy: 81.76%
# Observations: Increasing the number of units in the hidden layer does not lead to better performance. The model exhibits similar overfitting issues as previous configurations.
# 
# | Model                         | Validation Accuracy | Test Loss | Test Accuracy |
# |-------------------------------|---------------------|-----------|---------------|
# | One Hidden Layer (32 units)   | 0.8660              | 0.1132    | 0.8554        |
# | One Hidden Layer (32 units)   | 0.8329              | 0.1526    | 0.8188        |
# | Three Hidden Layers (32 units)| 0.8304              | 0.1644    | 0.8114        |
# | One Hidden Layer (64 units)   | 0.8176              | 0.1711    | 0.8002        |
# 
# 
# * Conclusion
# 
# Based on the experiments conducted, the model with one hidden layer consisting of 32 units and Tanh activation function (Model 1) performs the best, achieving the highest accuracy on both the training and validation sets. Introducing dropout regularization helps mitigate overfitting to some extent, but adding more hidden layers or increasing the number of units does not yield significant improvements. Therefore, Model 1 is recommended as the optimal configuration for sentiment analysis on the IMDB dataset. However, further experimentation with hyperparameter tuning and different architectures could potentially lead to even better results.
