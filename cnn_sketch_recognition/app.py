from flask import Flask,render_template,url_for,request
import flask
import pickle
import numpy as np
import cv2
import tensorflow as tf
import base64
import os
import keras

#initialize base64 image
init_Base64 = 21

#labels that we can predict
labels = {0:'Cat', 1:'Giraffe', 2:'Sheep', 3:'Bat', 4:'Octopus', 5:'Camel'}

#initialize graph
graph = tf.compat.v1.get_default_graph()

#load trained CNN model
with open(f'./model_cnn.pkl', 'rb') as f:
        model = pickle.load(f)

#initialize Flask app
app = flask.Flask(__name__, template_folder='templates')

#input drawing page
@app.route('/')
def home():
	return render_template('draw.html')

#prediction page
@app.route('/predict', methods=['POST'])
def predict():
        global graph
        with graph.as_default():
            if request.method == 'POST':
                    final_pred = None
                    #Preprocessing

                    draw = request.form['url']
                    draw = draw[init_Base64:]
                    #Decoding
                    draw_decoded = base64.b64decode(draw)
                    image = np.asarray(bytearray(draw_decoded), dtype="uint8")
                    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
                    resized = cv2.resize(image, (28,28), interpolation = cv2.INTER_AREA)
                    vect = np.asarray(resized, dtype="uint8")
                    vect = vect.reshape(1, 1, 28, 28).astype('float32')
                    
                    #predict the label
                    my_prediction = model.predict(vect)
                    index = np.argmax(my_prediction[0])
                    #Associating the index and its value within the dictionnary
                    final_pred = labels[index]

        return render_template('results.html', prediction = final_pred)

if __name__ == '__main__':
	app.run(debug=True)