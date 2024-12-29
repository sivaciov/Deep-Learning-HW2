# Deep Learning HW2: Implementation of Logistic Regression

This repository contains the implementation of **Logistic Regression** for binary classification as part of Homework 2 for a Deep Learning course. The model is trained using gradient descent to minimize the cross-entropy loss function, a common objective for classification tasks.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Usage](#usage)
  - [Installation](#installation)
  - [Running the Script](#running-the-script)
- [Example Output](#example-output)
- [Dependencies](#dependencies)
- [License](#license)

## Overview
Logistic Regression is a foundational model in machine learning for binary classification. This implementation demonstrates the application of gradient-based optimization to train the model, providing a clear understanding of how logistic regression works under the hood.

## Features
- **Customizable Learning Rate:** Set the learning rate to control the gradient descent step size.
- **Regularization Support:** Option to include L2 regularization to prevent overfitting.
- **Batch or Full Gradient Descent:** Flexibility to choose the gradient update strategy.
- **Performance Metrics:** Outputs accuracy and loss for evaluation.

## Usage

### Installation
Clone the repository to your local machine:

```bash
git clone https://github.com/sivaciov/Deep-Learning-HW2.git
cd Deep-Learning-HW2
```

### Running the Script
Ensure you have Python 3.6 or later installed. Run the `logistic_regression.py` script as follows:

```bash
python logistic_regression.py --train_file <path_to_training_data> --test_file <path_to_test_data> --learning_rate <lr> --num_epochs <epochs> [--lambda <reg_lambda>]
```

#### Command-line Arguments
- `--train_file`: Path to the training dataset (CSV or compatible format).
- `--test_file`: Path to the test dataset (CSV or compatible format).
- `--learning_rate`: Learning rate for gradient descent.
- `--num_epochs`: Number of training epochs.
- `--lambda`: Optional regularization parameter (default is 0, meaning no regularization).

Example:
```bash
python logistic_regression.py --train_file data/train.csv --test_file data/test.csv --learning_rate 0.01 --num_epochs 100 --lambda 0.001
```

## Example Output
After running the script, the model will output:
1. Training and test accuracy.
2. Cross-entropy loss during training.

Sample output:
```
Epoch 1/100: Loss = 0.693, Accuracy = 50.0%
Epoch 100/100: Loss = 0.453, Accuracy = 85.2%
Final Test Accuracy: 84.7%
```

## Dependencies
This implementation uses only the Python standard library. For enhanced performance, you can optionally install:
- `numpy`

Install using:
```bash
pip install numpy
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to use and adapt this code for your learning or projects. Contributions are welcome to improve the implementation or add new features!
