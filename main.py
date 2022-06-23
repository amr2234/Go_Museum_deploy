from flask import Flask, request, jsonify
import os

from Model_files.code import *
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.keras.preprocessing import image


app = Flask('prediction')
ResNet_model = tf.keras.models.load_model('./Model_files/ResNet_Classification_egy.h5')
categories = ['Amenemhat III','Bust of an elderly man with short hair','Hakuno','Hatchebsout','Head Of Queen Tiye','Heteb','King Chephren','Mask of Thuya','Mask of Yuya','Mother Of King Thutmose III','Narmer_paltte','Nefertiti','OsorkonII','RamsesII','Senwsret III','Sontm','Statue en granit gris','Statue of Mentuhotep Nebheptre','Statuette of King Seti I','The Priest Hetepdief','Thutmose III','Tout ankh amon','Userkaf','champollion',]

gen_model = load_model('./Model_files/Center_final_1000.pt')

@app.route('/upload_class',methods =["POST"])    
def upload_class():  
    if(request.method== "POST"):
        imgee = request.files['image']
        filename = imgee.filename
        path = os.path.join("./Image_uploud/"+filename)
        imgee.save(path)
        img_path =path
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        preds = ResNet_model.predict(x)       
        output = categories[np.argmax(preds[0])]
        Probability = preds[0][np.argmax(preds)] * 100
        print(output)
        return jsonify({
            "massege":"image uplouded",
            "output": output,
            "Probability" : Probability 
            })    
@app.route('/upload_with_crop',methods =["POST"])
def upload_with_crop():
    if(request.method== "POST"):
        imgee = request.files['image']
        filename = imgee.filename
        path = os.path.join("./Image_uploud/"+filename)
        imgee.save(path)
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (192, 192))
        n = random.randint(0,100000000)
        path ='./Output'
        cv2.imwrite(path+'/image'+str(n)+'.jpg',image)
        resized_path=path+'/image'+str(n)+'.jpg'
        Croped,num = center_crop(resized_path)
        input_img = cv2.imread(path+'/image_croped'+str(num)+'.jpg')[:, :, :3]
        output_img, blended_img = perform_outpaint(gen_model, input_img)
        plt.imsave( path+'/image_output'+str(n)+'.jpg', blended_img)
        img_path= path+'/image_output'+str(n)+'.jpg'
        img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
        x = tf.keras.utils.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        preds = ResNet_model.predict(x)
        print("Model predicts a \"{}\" with {:.2f}% probability".format(categories[np.argmax(preds[0])], preds[0][np.argmax(preds)] * 100))
        output = categories[np.argmax(preds[0])]
        Probability = preds[0][np.argmax(preds)] * 100
        print(output)
        return jsonify({
            "massege":"Done",
            "output": output,
            "Probability" : Probability 
            })


@app.route('/', methods=['GET','POST'])
def ping():
    return "Pinging Model!!"


