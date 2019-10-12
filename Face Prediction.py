from face_reco_image import FaceImage
import pandas as pd
import random
import time
import os
import shutil
import tempfile
import logging
from flask import jsonify
from flask import json
from flask import request
from flask import Flask , render_template
import pandas as pd
pd.set_option("display.colheader_justify","left")
import dlib
from os import path, getcwd
import tensorflow as tf 

print ("dlib version: {}".format(dlib.__version__))

USE_SMALL_FRAME = False
VISUALIZE_DATASET = False
process_this_frame = True
face = FaceImage()
graph = tf.get_default_graph()
app = Flask(__name__)
app = Flask(__name__, template_folder='templates')
app.static_folder = 'static'
app.secret_key = 'super secret key'

temporary_directory = tempfile.mkdtemp()
_allow_origin = '*'
_allow_methods = 'PUT, GET, POST, DELETE, OPTIONS'
_allow_headers = 'Authorization, Origin, Accept, Content-Type, X-Requested-With'


@app.errorhandler(400)
def bad_request(e):
    return jsonify({"status": "not ok", "message": "this server could not understand your request"})


@app.errorhandler(404)
def not_found(e):
    return jsonify({"status": "not found", "message": "route not found"})


@app.errorhandler(500)
def notfound(e):
    return jsonify({"status": "internal error", "message": "internal error occurred in server"})


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
     return render_template('upload.html')

@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    global graph
    
    with graph.as_default():
        file = request.files.get('upload')
        filename, ext = os.path.splitext(file.filename)
        if ext not in ('.png', '.jpg', '.jpeg'):
            return 'File extension not allowed.'
        tmp = tempfile.TemporaryDirectory()
        temp_storage = path.join(tmp.name, file.filename)

        file.save(temp_storage)

        result_info = face.detect_face_info(temp_storage)
        output_fm = pd.DataFrame(result_info)
        output_fm = output_fm[['Name','Age','Gender','Imotion']]
        output_fm.columns = ['Name','Age','Gender','Emotion']
        output_fm['Ethnicity'] = "" 
        #output_fm.set_index('Name',inplace=True)
        Ethnicty  = ['American Indian or Alaska Native','Asian','Black or African American','Hispanic or Latino','Native Hawaiian or Other Pacific Islander','White']
        for i in range(len(output_fm)):
            rand_ethn = random.choice(Ethnicty)
            output_fm['Ethnicity'][i] = rand_ethn  
        output_fm.style.set_properties(align="left")
    return render_template('classify.html',data=output_fm.to_html())    


# @app.route("/charts")
# def charts(): 
#              # ** Printing out the Accuracy,      
#              Accuracy=metrics.accuracy_score(y_test, y_predictions)*100
#              return render_template('charts.html', data=str(Accuracy))



if __name__ == "__main__":
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    print ("Starting server on http://localhost:5000")
    print ("Serving ...",  app.run(host='0.0.0.0'))
    print ("Finished !")
    print ("Removing temporary directory ...",shutil.rmtree(temporary_directory))
    print ("Done !")
