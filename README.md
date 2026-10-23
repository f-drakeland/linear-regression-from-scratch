# Linear Regression From Scratch

An educational implementation of univariate linear regression
and batch gradient descent using Python and NumPy.

## Objective

Investigate whether iris flower petal length can be predicted
from sepal length using a simple linear regression model.

The project emphasizes understanding the mathematics,
implementing optimization, and evaluating model limitations.

## Dataset

Fisher's Iris dataset, accessed through scikit-learn.

- 150 observations
- Input: sepal length (cm)
- Target: petal length (cm)
- Training observations: 120
- Test observations: 30
- Random state: 42

All three iris species are included.

## Method

The model uses:

    prediction = w * x + b

Training minimizes squared-error cost:

    J(w,b) = sum((prediction - actual)^2) / (2*m)

Both parameters are updated using batch gradient descent.

Training configuration:

- Initial w: 0
- Initial b: 0
- Learning rate: 0.01
- Total iterations: 25,000

The implementation is independently checked against
NumPy's least-squares solution.

## Results

| Metric | Value |
|---|---:|
| Final w | 1.823716 |
| Final b | -6.866448 |
| Training cost | 0.390808 |
| Test cost | 0.299055 |
| Test MAE | 0.6186 cm |
| Reference least-squares cost | 0.390772 |

These results were obtained using an 80/20 train-test split.

## Findings

Gradient descent substantially reduced the training cost
and approached the independent least-squares solution.

Test results demonstrate that the trained model can make
predictions on held-out observations.

However, residual analysis suggests systematic prediction
errors. A single linear relationship using only sepal length
does not capture all variation in petal length.

The results are based on one train-test split and should
not be interpreted as comprehensive generalization evidence.

## Implementation Notes

- Regression predictions are implemented directly.
- Cost and gradients are calculated manually.
- Gradient descent does not use a library optimizer.
- scikit-learn supplies the dataset and train-test split.
- NumPy is used for numerical operations and verification.
- Matplotlib is used for visualization.

## Reproducibility

Create a Python virtual environment and install dependencies:

    python -m pip install -r requirements.txt

Launch JupyterLab:

    jupyter lab

Open regression.ipynb and execute all cells in order.

## Scope

This is a foundational machine learning experiment, not
a production prediction system.

Its purpose is to demonstrate model construction,
optimization, evaluation, and the limitations of
a simple linear regression approach.
