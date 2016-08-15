import os
import sys
import glob
import numpy as np
import heapq
import base64
import cStringIO
import timeit
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
		t = timeit.Timer('char in text', setup='text = "sample string"; char = "g"')
		scores = models[model].score(data['xnew'])
		print 'Total prediction duration %f ms'%(1000*t.timeit())
	except ValueError as e: 
		return jsonify(status=200, error=e.message)
	
	return jsonify(status=200, **scores)



if __name__ == '__main__':
	app.run(debug=True)
