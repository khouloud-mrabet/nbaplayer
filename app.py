import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

#Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


#default page of our web-app
@app.route('/')
def home():
	return render_template('index.html')

#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
	'''
	For rendering results on HTML GUI
	'''
	int_features = [float(x) for x in request.form.values()]
	final_features = [np.array(int_features)]
	prediction = model.predict(final_features)
	if prediction == 0:
		output = 'a target'
	else :
		output = 'not a taget'
	return render_template('index.html', prediction_text='this player is {}'.format(output))

if __name__ == "__main__":
	app.run(debug=True)	
#app.debug=True
	#app.run(host = '0.0.0.0', port = 5000)
 

