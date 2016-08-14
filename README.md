# PMML Scoring Engine
Scoring engine for PMML models implemented using python. The PMML scoring engine exposes predictive machine learning models as REST endpoints. Clients can send new observations to the scoring engine in the JSON file format. The scoring engine returns a JSON response containing the new scores

## Usage
```sh
git clone https://github.com/maxkferg/pmml-scoring-engine.git
cp some-pmml-file.pmml pmml-scoring-engine/examples
cd pmml-scoring-engine
python run.py
```

## Client
```python
r = HTTP.get('/pmml/some-pmml-filename.pmml',{xnew:[1,4,5,3,5,7,8,4,3,6,7,1]})
r.response -> {mu:1.45324344,sd:3.2214342}
```


## License 
MIT
