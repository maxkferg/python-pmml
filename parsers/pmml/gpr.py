from models.gpr import GaussianProcess 



class GaussianProcessParser():
	
	def __init__(self):
		pass

	def parse(self,filename):
		"""Return a new Gaussian Process Model"""
		return GaussianProcess()