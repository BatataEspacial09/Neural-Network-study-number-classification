import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import numpy as np
import os

def create_folder(folder_name):
    try:
        os.mkdir(folder_name)
        print(f"Folder '{folder_name}' created successfully.")
    except FileExistsError:
        print(f"Folder '{folder_name}' already exists.")

def create_folder_within_parent(parent_folder, new_folder):
    try:
        os.mkdir(os.path.join(parent_folder, new_folder))
        print(f"Folder '{new_folder}' created successfully within '{parent_folder}'.")
    except FileExistsError:
        print(f"Folder '{new_folder}' within '{parent_folder}' already exists.")

create_folder("images")

create_folder_within_parent("images", "original")
create_folder_within_parent("images", "converted")

def treat_image (item, flag):
    og = Image.open(f'images/original/{item}')
    converted = og.resize((28, 28))
    converted = converted.convert('L')
    if flag == 'Y': converted.save(f'images/converted/{item}')
    return converted

def classify_image (tensor):
    tf_image_tensor = np.array(tensor)
    tf_image_tensor = tf_image_tensor.reshape(-1, 28*28).astype("float32")/255.0 #Normalizes the values 

    average = tf.reduce_mean(tf_image_tensor) 

    if average > 0.6: #if the average value of the tensor is higher than 0.6 it means that it's predominantly white and needs to be inverted
        tf_image_tensor = tf_image_tensor - 1
        tf_image_tensor = tf_image_tensor * -1

    prediction = model.predict(tf_image_tensor)

    return np.argmax(prediction)


model = keras.models.load_model('saved_model/')

print("Save converted images ? [Y/N]")
flag = input()

images = os.listdir('images/original')

for item in images:
    conv = treat_image(item, flag)
    pred = classify_image(conv)
    print(f"for item:{item}\nprediction: {pred}")
