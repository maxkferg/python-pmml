import os
import sys
import glob
import numpy as np
import heapq
import base64
import cStringIO
import time
from flask import Flask, jsonify, abort, request
from parsers.pmml.gpr import GaussianProcessParser

def load_examples():
	"""Load example PMML files"""
	models = dict();
	parser = GaussianProcessParser()
	root = os.path.dirname(os.path.realpath(__file__))
	examples = os.path.join(root,'examples','pmml')
	pattern = examples + '/*.pmml'

	for filepath in glob.iglob(pattern):
		print 'Loading ' + os.path.basename(filepath)
		name = os.path.basename(filepath)
		models[name] = parser.parse(filepath)
	return models

##############################
# Set up REST Flask server
##############################
app = Flask(__name__)
models = load_examples()


@app.route('/predict/<model>',methods=['POST'])
def predict_score(model):
	"""Use one of the PMML models to score a new observation"""
	if not model in models:
		message = "Unknown model "+model 
		return jsonify(status=404, error=message);

	data = request.get_json()
	if not data:
		message = "Request does not contain form data" 
		return jsonify(status=404, error=message);

	if 'xnew' not in data:
		message = "Request does not contain xnew" 
		return jsonify(status=404, error=message);

	try:
		start = time.time()
		xnew = format_xnew(data['xnew'])
		scores = models[model].score(xnew)
		print 'Total prediction duration %f s'%(time.time()-start)
	except ValueError as e: 
		return jsonify(status=200, error=e.message)
	
	return jsonify(status=200, **scores)



def format_xnew(xnew):
	"""Convert xnew from JSON/Python matrix to 2D Numpy matrix"""
	xnew = np.array(xnew)
	if len(xnew.shape)==1:
		xnew = xnew.reshape(1,-1)
	return xnew


if __name__ == '__main__':
	app.run(debug=True)
