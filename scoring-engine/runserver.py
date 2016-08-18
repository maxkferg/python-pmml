import os
import sys
import glob
import numpy as np
import heapq
import base64
import cStringIO
import time
import tornado.ioloop
import tornado.web
import tornado.escape
import tornado.httpserver
from parsers.pmml.gpr import GaussianProcessParser
PORT=80
VERSION = '0.0.0'
PROCESSES = 4


def load_models():
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

# Load all PMML when the server starts
models = load_models();


class VersionHandler(tornado.web.RequestHandler):
	def get(self):
		"""Return the version"""
		response = { 'version': VERSION}
		self.write(response)



class PredictHandler(tornado.web.RequestHandler):
	def post(self,model):
		"""Use one of the PMML models to score a new observation"""
		if not model in models:
			message = "Unknown model "+model 
			self.set_status(400)
			return self.finish({'error': message});

		data = tornado.escape.json_decode(self.request.body)
		if not data:
			message = "Request does not contain form data" 
			self.set_status(400)
			return self.finish({'error': message});

		if 'xnew' not in data:
			message = "Request does not contain xnew" 
			self.set_status(400)
			return self.finish({'error': message});

		try:
			start = time.time()
			xnew = self._format_xnew(data['xnew'])
			scores = models[model].score(xnew)
			print 'Total prediction duration %f s'%(time.time()-start)
		except ValueError as e: 
			self.set_status(500)
			return self.finish({'error': e.message});

		return self.finish(scores);


	def _format_xnew(self,xnew):
		"""Convert xnew from JSON/Python matrix to 2D Numpy matrix"""
		xnew = np.array(xnew)
		if len(xnew.shape)==1:
			xnew = xnew.reshape(1,-1)
		return xnew


# Define the tornado routes
application = tornado.web.Application([
	(r"/predict/([^/]+)", PredictHandler),
	(r"/", VersionHandler)
])


if __name__ == '__main__':
        print 'Starting tornado on port %i'%PORT
	server = tornado.httpserver.HTTPServer(app)
	server.bind(PORT)
	server.start(PROCESSES)
	tornado.ioloop.IOLoop.current().start()
