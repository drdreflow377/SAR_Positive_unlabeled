# Positive and Unlabeled Learning
simulating unlabeled data in the breast cancer data set, then building a classifer using propensity
Uses Selected at Random Assumption and propensity. Adapted from "Machine Learning from Weak Supervision" https://mitpress.mit.edu/9780262047074/machine-learning-from-weak-supervision/


# Propensity Model and Propensity Breast Cancer

This project contains a Propensity Model and a Jupyter notebook `propensity_breast_cancer.ipynb` that applies the model to a breast cancer dataset.

## Propensity Model

The Propensity Model is implemented in the `propensity_ops.py` file. It contains the following functions:

# Propensity Model and Propensity Breast Cancer

- `estimate_propensity_scores(X, y)`: Estimates the propensity scores using logistic regression. The propensity score for an observation is the probability that the observation is in the positive class.

This project contains a Propensity Model and a Jupyter notebook `propensity_breast_cancer.ipynb` that applies the model to a breast cancer dataset.

## Propensity Model

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
- `compute_class_weights(y, propensity_scores)`: This function computes the class weights for the positive class. The class weight is the inverse of the mean propensity score for the positive class.

4. Split the data into a training set and a test set.
5. Train the PU model on the training data.
6. Predict the probabilities of the positive class for the test data.
7. Calculate the optimal threshold for classifying observations.
- `train_pu_model(X, y, class_weights)`: This function trains a logistic regression model on the data `X` and labels `y`, using the provided class weights.

8. Classify the test observations as positive or negative based on the optimal threshold.
9. Save the predicted positive observations to a CSV file.

## How to Run

- `calculate_optimal_threshold(y, y_probs)`: This function calculates the optimal threshold for classifying instances as positive or negative. The optimal threshold is the one that minimizes the Euclidean distance to the top left corner of the ROC curve.

## Propensity Breast Cancer Notebook

To run the notebook, you need to have Python and Jupyter installed on your machine. You can install them using pip:

```
The `propensity_breast_cancer.ipynb` notebook applies the Propensity Model to a breast cancer dataset. The notebook includes the following steps:

pip install python jupyter
```

- Loading the data
Then, you can start Jupyter by running:

- Estimating the propensity scores
- Computing the class weights
- Training the PU model
```
jupyter notebook
```

- Calculating the optimal threshold
- Making predictions on the test set
- Saving the predictions to a CSV file

In the Jupyter dashboard, navigate to the directory containing the notebook and click on it to open it.

## How to Run

To run the cells in the notebook, you can click on a cell and then click the "Run" button in the toolbar, or you can press Shift+Enter.

## Dependencies

This project requires the following Python libraries:

- numpy
To run the notebook, you need to have Jupyter installed. You can start Jupyter by running `jupyter notebook` or `jupyter lab` in your terminal, and then navigate to the `propensity_breast_cancer.ipynb` notebook.

- pandas
- sklearn

You can install them using pip:

```
pip install numpy pandas sklearn
```To run the `propensity_ops.py` script, you can use a Python interpreter. For example, you can run `python propensity_ops.py` in your terminal.

## Dependencies

