import flask
from flask import request
import pickle
import librosa
import pandas as pd
import os
import numpy as np

UPLOAD_FOLDER = '.'


def load_audio(audio):
	list_=[]
	index=0
	cols=["mfkk"+str(i) for i in range(20)]
	for row in ["zero","centroid","rolloff","chroma"]:
	    cols.append(row)
	x,sr=librosa.load(os.path.join(app.config['UPLOAD_FOLDER'],audio),duration=5,res_type='kaiser_fast')
	list_.append([np.mean(x) for x in librosa.feature.mfcc(x,sr=sr)])
	list_[index].append(sum(librosa.zero_crossings(x)))
	list_[index].append(np.mean(librosa.feature.spectral_centroid(x)))
	list_[index].append(np.mean(librosa.feature.spectral_rolloff(x,sr=sr)))
	list_[index].append(np.mean(librosa.feature.chroma_stft(x,sr=sr)))
	return pd.DataFrame(list_,columns=cols)

# Use pickle to load in the pre-trained model.
with open(f'model/randomforest_trained.da', 'rb') as f:
    model = pickle.load(f)
app = flask.Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/', methods=['GET', 'POST'])
def main():
	if flask.request.method == 'GET':
		return(flask.render_template('main.html'))
	if flask.request.method == 'POST':
		audio = request.files.get('audio')
		#return "hi"
		audio.save(os.path.join(app.config['UPLOAD_FOLDER'],audio.filename))
		input_variables = load_audio(audio.filename)
		prediction = model.predict(input_variables)[0]
		if(prediction==0):
			pred="NORMAL"
		elif(prediction==1):
			pred="MURMUR"
		elif(prediction==2):
			pred="ARTIFACT"
		return flask.render_template('main.html',
					     original_input={'Audio':audio.filename},
		                             result=pred,
		                             )
if __name__ == '__main__':
    app.run()
