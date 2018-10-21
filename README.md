# Python PMML

Generate and Evaluation PMML in Python.
Currently, this pacakge supports the Gaussian Process Regression and DeepNeuralNetwork model types.

# Installation
All code is written in Python 3. First install the required packages
```sh
pip3 install -r requirements.txt
```
Then run the tests to make sure everything is working
```sh
python tests.py
```
If all the tests pass, then you are good to start using the package.

# Command line interface 

Evaluating a PMML file with the commandline is straightforward.
Models inputs are defined in json or image files.
```sh
	# Gaussian process regression evaluation
	python pmml.py predict \
		--model=examples/gpr/energy-prediction-1.pmml \
		--input=test/assets/energy-inputs.json

	# Image classification with a deep neural network
    python pmml.py predict \
        --model=examples/deepnetwork/VGG16/model.pmml \
        --input=test/assets/cat.jpg
```

A scoring engine server can also be started from a model file.
Inputs to the scoring engine can also be sent as image files or json.

```sh
	# Gaussian process regression evaluation scoring engine
	python pmml.py runserver \
	   --model=examples/gpr/energy-prediction-1.pmml \
	   --port=5000

	# Neural network scoring engine
    python pmml.py runserver \
    	--model=examples/deepnetwork/VGG16/model.pmml \
    	--port=5000 \
```

Most of the exmaples models have been converted from other open-source models.
To regenerate the example models:

```sh
	# Build new PMML files
	python pmml.py build_examples
```


## Scoring Engine Server

Queries can be sent to the scoring engine server using standard JSON:
```python
#!Python3
r = HTTP.get({'xnew': [1,4,5,3,5,7,8,4,3,6,7,1]})
r.response # -> {mu:1.45324344,sd:3.2214342}
```

## Contributors

* Max Ferguson: [@maxkferg](https://github.com/maxkferg)
* Stanford Engineering Informatics Group: [eil.stanford.edu](http://eil.stanford.edu/index.html)

