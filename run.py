import os
import sys
import numpy as np
import heapq
import base64
import cStringIO
from flask import Flask, jsonify, abort, request
from parsers.pmml.gpr import GaussianProcessParser

parser = GaussianProcessParser()

model1 = parser.parse('examples/tool-condition.pmml')
model2 = parser.parse('examples/energy-prediction.pmml')


##############################
# Set up Rest Flask server
##############################
app = Flask(__name__)



@app.route('/examples/tool-condition', methods=['POST'])
def predict_score_condition():
	data = request.get_json()
	if not data:
		print 'Aborting, no form data'
		abort(400)

	if 'xnew' not in data:
		print 'Aborting, no new observation'
		abort(400)

	scores = model1.score(data['xnew'])
	return jsonify(status=200, **scores)



@app.route('/examples/energy-prediction', methods=['POST'])
def predict_score_energy():
	data = request.get_json()
	if not data:
		print 'Aborting, no form data'
		abort(400)

	if 'xnew' not in data:
		print 'Aborting, no new observation'
		abort(400)

	scores = model1.score(data['xnew'])
	return jsonify(status=200, **scores)



if __name__ == '__main__':
	app.run(debug=True)
