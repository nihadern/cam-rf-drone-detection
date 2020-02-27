#!/usr/bin/env python
# coding: utf-8

# # Neccesary modules 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split


# # Get the data

# In[2]:


background = np.load("data/background_rf_LH_normalized.npy")
drone = np.load("data/drone_rf_LH_normalized.npy")


# In[3]:


print(background.shape)
print(drone.shape)


# In[4]:


num = random.randint(0, len(background)-1)
channel = 1
plt.plot(background[num][channel], label="background")
plt.plot(drone[num][channel],label="drone")
plt.legend(loc='upper right')


# # Train/ test split and data formatting

# In[5]:


Y = np.array([0 for i in enumerate(background)] + [1 for i in enumerate(drone)])
X = np.append(background,drone,axis=0)
Y = Y.reshape(-1,1)


# In[6]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[7]:


def split_rf(x_data):
    low = []
    high = []
    for x in x_data:
        low.append(x[0].flatten().astype(np.float32))
        high.append(x[1].flatten().astype(np.float32))
    low = np.array(low)
    high = np.array(high)
    return [low, high]
x_train = split_rf(x_train)
x_test = split_rf(x_test)
    


# In[8]:


x_train[0]


# # Model Specification

# In[9]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, concatenate
from tensorflow.keras.layers import Input


# define two sets of inputs
low_rf  = Input(shape=(X.shape[2],))
high_rf = Input(shape=(X.shape[2],))

# the first branch operates on the first input
x1 = Dense(500 , activation="relu")(low_rf)
# x1 = Dense(500, activation="relu")(x1)
x1 = Model(inputs=low_rf, outputs=x1)

# the second branch operates on the second input
x2 = Dense(500 , activation="relu")(high_rf)
# x2 = Dense(500, activation="relu")(x2)
x2 = Model(inputs=high_rf, outputs=x2)

# combine the output of the two branches
combined = concatenate([x1.output, x2.output])

# apply a FC layer and then a regression prediction on the
# combined outputs
z = Dense(100, activation="relu")(combined)
z = Dense(1, activation="sigmoid")(z)

model = Model(inputs=[x1.input, x2.input], outputs=z)
model.summary()


# In[10]:


model.compile(optimizer ='adam' , loss = "binary_crossentropy", metrics=["accuracy"])


# # Train Model

# In[ ]:


batch_size =4
epochs = 10
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# # Free memory

# In[12]:


# del X 
# del Y
# del background
# del drone


# In[ ]:




