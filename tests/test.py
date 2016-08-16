import time
import requests



class TestServer():

	def test(self):
		headers = {'Content-Type': 'application/json', 'Accept':'application/json'}
		payload = {'xnew': [1,2,3,4,5]}
		r = requests.post('http://localhost:5000/predict/energy-prediction-1.pmml',json=payload, headers=headers)
		print r.text

	def test_array(self):
		headers = {'Content-Type': 'application/json', 'Accept':'application/json'}
		payload = {'xnew': [[1,2,3,4,5],[2,3,4,5,6]]}
		print requests.post('http://localhost:5000/predict/energy-prediction-1.pmml',json=payload, headers=headers).text


	def test_speed(self):
		start = time.time();
		headers = {'Content-Type': 'application/json', 'Accept':'application/json'}
		payload = {'xnew': [1,2,3,4,5]}	

		for i in range(10):
			r = requests.post('http://localhost:5000/predict/energy-prediction-1.pmml',json=payload, headers=headers)
			print 'Total request duration %f ms'%(time.time()-start)
		print r.text
		print 'Total time for 10 requests %f'%(time.time()-start)



if __name__ == '__main__':
	TestServer().test()
	TestServer().test_array()
	TestServer().test_speed()