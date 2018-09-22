import requests, numpy as np
from flask import Flask, request, make_response,jsonify
import numpy as np
import json
import urllib
import random
##
from keras.models import load_model
from keras.preprocessing.image import  load_img, img_to_array
#import urllib2
import io, os

model = load_model('my_model.h5')
model.load_weights('basic_cnn_20_epochs.h5')

app = Flask(__name__,static_url_path='')

img_width, img_height = 512, 512

def preprocess_img(im,target_size=(512,512)): #นี่ด้วย
    imgg=load_img(im,target_size=(512,512))
    predictg = img_to_array(imgg)
    #img = predictg.reshape((1,img_width, img_height,3))
    os.remove(im)
    return predictg

def load_im_from_url(url):
    r = requests.get(url)
    name = random.randint(0, 10000)

    with open(str(name)+'.jpg', 'wb') as f:
        f.write(r.content)
    return str(name)+'.jpg'

def predict(url): #แก้นี่
    img=load_im_from_url(url)
    img=preprocess_img(img)
    preds = model.predict_classes(img.reshape((1,img_width,img_height,3)),batch_size=8, verbose=0)
    print(preds)
    return preds

@app.route('/classify', methods=['GET'])
def classify():
    image_url = request.args.get('imageurl')
    resp = predict(image_url)
    if resp[0] == [1]:
        result = 'Not Cancer'
    elif resp[0] == [0]:
        result = 'Cancer'
    results=[]
    results.append({"class_name":result,"score":'None'})
    return jsonify({'results':results})

@app.route('/', methods=['GET'])
def root():
    return app.send_static_file('index.html')
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
