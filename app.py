#from crypt import methods
from unittest import result
from flask import Flask, render_template , request
import pickle
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
app = Flask(__name__)

#pickle.load(open('/Users/bilalkhan/Desktop/history.pkl', 'rb'))

model = load_model('/Users/bilalkhan/Desktop/agfowmodel.h5')

labels = pd.read_csv('labels.txt',sep='\n').values

@app.route('/', methods=['GET'])
def index():
    return render_template("index1.html", data=' flask to html')

@app.route('/prediction', methods=["GET","Post"])
def prdiction():
    img = request.files['img']

    img.save("img.jpg")

    image = cv2.imread('img.jpg')

    image = cv2.resize(image, (256,256))

    image = np.reshape(image , (1,256,256,3))

    pred = model.predict(image)

    pred = np.argmax(pred)

    result = labels[pred]

    return render_template("/prediction.html", data=result)

if __name__ == "__main__":
   app.run(debug=True)


