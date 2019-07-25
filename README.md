# Python PMML

Generate and Evaluation PMML in Python.
Currently, this pacakge supports the Gaussian Process Regression and DeepNeuralNetwork model types.

## Installation
All code is written in Python 3. First install the required packages

```sh
conda env create -f environment.yml -n pmml3
conda activate pmml3
```

Then run the tests to make sure everything is working
```sh
python tests.py
```

If all the tests pass, then you are good to start using the package.

## Command line interface

Evaluating a PMML file with the commandline is straightforward.
Models inputs are defined in json or image files.
```sh
# Gaussian process regression evaluation
python main.py predict \
	--model=examples/gpr/energy-prediction-1.pmml \
	--input=test/assets/energy-inputs.json

# Image classification with a deep neural network
python main.py predict \
    --model=examples/deepnetwork/VGG16/model.pmml \
    --input=test/assets/cat.jpg
```

Any PMML file can be validated against the DeepNetwork schema:
```sh
# Validate examples/deepnetwork/VGG16/model.pmml
python main.py validate --filename=examples/deepnetwork/VGG16/model.pmml

# Validate all of the examples
python main.py validate
```


A scoring engine server can also be started from a model file.
Inputs to the scoring engine can also be sent as image files or json.

```sh
# Gaussian process regression evaluation scoring engine
python main.py runserver \
   --model=examples/gpr/energy-prediction-1.pmml \
   --port=5000

# Neural network scoring engine
python main.py runserver \
	--model=examples/deepnetwork/VGG16/model.pmml \
	--port=5000 \
```

Most of the examples models have been generated from other open-source projects.
To regenerate the example models:

```sh
# Build new PMML files
python main.py build_keras_examples
python main.py build_pytorch_examples
```


## Scoring Engine Server

Queries can be sent to the scoring engine server using standard JSON:
```python
#!Python3
r = HTTP.get({'xnew': [1,4,5,3,5,7,8,4,3,6,7,1]})
r.response # -> {mu:1.45324344,sd:3.2214342}
```

## Managing dependencies
The build hash in not included in the dependencies, making it easier to install the environment on different platforms. To export the environment:

```sh
conda env export --no-builds | grep -v "prefix" > environment.yml
```

## Contributors

* Max Ferguson: [@maxkferg](https://github.com/maxkferg)
* Stanford Engineering Informatics Group: [eil.stanford.edu](http://eil.stanford.edu/index.html)

