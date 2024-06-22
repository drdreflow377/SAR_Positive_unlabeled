# Positive and Unlabeled Learning
simulating unlabeled data in the breast cancer data set, then building a classifer using propensity
Uses Selected at Random Assumption and propensity. Adapted from "Machine Learning from Weak Supervision" https://mitpress.mit.edu/9780262047074/machine-learning-from-weak-supervision/


# Propensity Model and Propensity Breast Cancer

This project contains a Propensity Model and a Jupyter notebook `propensity_breast_cancer.ipynb` that applies the model to a breast cancer dataset.

## Propensity Model

The Propensity Model is implemented in the `propensity_ops.py` file. It contains the following functions:


- `estimate_propensity_scores(X, y)`: Estimates the propensity scores using logistic regression. The propensity score for an observation is the probability that the observation is in the positive class.

This project contains a Propensity Model and a Jupyter notebook `propensity_breast_cancer.ipynb` that applies the model to a breast cancer dataset.

- `compute_class_weights(y, propensity_scores)`: Computes the class weights for the positive class. The class weight is the inverse of the mean propensity score for the positive class.

The Propensity Model is implemented in the `propensity_ops.py` file. It contains the following functions:

- `train_pu_model(X, y, class_weights)`: Trains a logistic regression model on the data `X` and labels `y` using the provided class weights.

- `estimate_propensity_scores(X, y)`: This function estimates the propensity scores using logistic regression. The propensity scores are the probabilities that each instance belongs to the positive class.

- `calculate_optimal_threshold(y, y_probs)`: Calculates the optimal threshold for classifying observations as positive or negative. The optimal threshold is the one that minimizes the Euclidean distance from the top left corner of the ROC curve.

## Propensity Breast Cancer Notebook

The `propensity_breast_cancer.ipynb` notebook applies the Propensity Model to a breast cancer dataset. The notebook contains the following steps:

1. Load the data.
2. Estimate the propensity scores.
3. Compute the class weights.
4. Split the data into a training set and a test set.
5. Train the PU model on the training data.
6. Predict the probabilities of the positive class for the test data.
7. Calculate the optimal threshold for classifying observations.
8. Classify the test observations as positive or negative based on the optimal threshold.
9. Save the predicted positive observations to a CSV file.



## Propensity Breast Cancer Notebook

To run the notebook, you need to have Python and Jupyter installed on your machine. You can install them using pip:

```
The `propensity_breast_cancer.ipynb` notebook applies the Propensity Model to a breast cancer dataset. 
```

- Loading the data
Then, you can start Jupyter by running:

- Estimating the propensity scores
- Computing the class weights
- Training the PU model
- Calculating the optimal threshold
- Making predictions on the test set
- Saving the predictions to a CSV file


## Dependencies

This project requires the following Python libraries:

- numpy
To run the notebook, you need to have Jupyter installed. You can start Jupyter by running `jupyter notebook` or `jupyter lab` in your terminal, and then navigate to the `propensity_breast_cancer.ipynb` notebook.

- pandas
- sklearn

