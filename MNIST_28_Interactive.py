import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout 
from sklearn.metrics import confusion_matrix
import os
import cv2
import time

model = tf.keras.models.load_model('C:\\Users\\08aks\\OneDrive\\Desktop\\AI files\\MNIST_28_model_4.keras') # loading the saved model

import subprocess
subprocess.Popen('mspaint') #we will open microsoft paint using subprocess 

folder_path = "C:\\Users\\08aks\\OneDrive\\Desktop\\MNIST_save"# It is the folder where we save the paint drawing in 28 X 28 pixel format 
last_seen_files = set(os.listdir(folder_path))# Gives the set of all files before running the program in folder path  
#print(last_seen_files)

print("Waiting for new image in:", folder_path)

timeout = 60
start_time = time.time() # Notes the clock time
new_file = None

while True:
    current_files = set(os.listdir(folder_path))# Gives the set of all files after running the program in folder path  
    added_files = current_files - last_seen_files # will give the name of the added file in .png format 

    for file in added_files:# if added_files is empty then this block will not execute 
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            new_file = os.path.join(folder_path, file) # makes new_file as the path of the created image
            break

    if new_file:
        print("Found new image:", new_file)
        try:
            img = cv2.imread(new_file, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28)) # resizing just to insure running the program if someone inputs different pixel format 
            img = np.invert(img) # inverting the image 
            img = img.reshape(1, 784) # reshaping to an array 
            img = img/255 # normalizing to enhance gradient descent

            prediction = model.predict(img)
            prediction = tf.nn.softmax(prediction) # because using From_logits = True and setting a linear layer as the last layer 
            result = np.argmax(prediction)

            print("The number predicted is:", result) #printing the outcome
            plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary) # plotting the saved image 
            plt.title(f"Predicted: {result}")
            plt.axis('off')
            plt.show()
        except Exception as e:
            print("Error while processing image:", e)
        break

    if time.time() - start_time > timeout: # time.time() gives the noted time while running the program at this time 
        print("Timeout: No image saved within 60 seconds.")
        break

    time.sleep(1) # slows the program by 1 second to prevent infinite looping 