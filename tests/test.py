import requests



class TestServer():

	def test(self):
		headers = {'Content-Type': 'application/json', 'Accept':'application/json'}
		payload = {'xnew': [1,2,3,4,5,6,7,8,9]}

		r = requests.post('http://localhost:5000/examples/tool-condition',json=payload, headers=headers)
		print r.text

		r = requests.post('http://localhost:5000/examples/energy-prediction',json=payload, headers=headers)
		print r.text



if __name__ == '__main__':
	TestServer().test()