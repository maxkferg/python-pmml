# Models

Models understand the mathematical apsect of a machine learning model.
They are created by parsers, who parse text files to obtain model parameters.
Models can be initialized with model parameters, and should always use named arguments
All models must be able to generate scores for new observations

## Model(mean_kernel, cov_kernel, hyperparams, training_data)

## Model.score(xnew)
Return a dict that represents the predicted score. 
The score should either be in the form {y:1212.3} or {mu:12231.231, sd:233.24}