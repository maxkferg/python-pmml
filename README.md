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




## Binary Dependencies
```sh
sudo apt install python-dev
sudo apt install python-pip
sudo apt install libxml2-dev libxslt1-dev # XML dependency
sudo apt install libblas-dev liblapack-dev libatlas-base-dev gfortran # Numpy dependency
sudo apt install lib32z1-dev #lxml dependency
pip install lxml
pip install Cython
```

## Installation
`pmml-scoring-engine` requires the development version of ScikitLearn for 
the GaussianProcessRegressor class. Install the development version from Github:

```sh
git clone https://github.com/scikit-learn/scikit-learn.git
cd scikit-learn
python setup.py build
sudo python setup.py install
```

Install dependencies with setuptools
```python
sudo python setup.py install
```


## Usage
```sh
git clone https://github.com/maxkferg/pmml-scoring-engine.git
cp some-pmml-file.pmml pmml-scoring-engine/scoring-engine/examples/pmml
cd pmml-scoring-engine/scoring-engine
python runserver.py
```

## Client
```python
r = HTTP.get('/pmml/some-pmml-filename.pmml',{xnew:[1,4,5,3,5,7,8,4,3,6,7,1]})
r.response # -> {mu:1.45324344,sd:3.2214342}
```

## Contributors

* Max Ferguson: [@maxkferg](https://github.com/maxkferg)
* Stanford Engineering Informatics Group: [eil.stanford.edu](http://eil.stanford.edu/index.html)

