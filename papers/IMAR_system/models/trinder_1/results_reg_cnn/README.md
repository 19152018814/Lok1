# Trinder-1 CNN Regression Weights — best_model.pth

Place your trained CNN weight file at:

```
models/trinder_1/results_reg_cnn/best_model.pth
```

This file is saved automatically by the training script
`conc_reg_multi_cnn.py` upon training completion.

## CNN Architecture

The architecture MUST match `infer.py::CNNModel` exactly:

```
Conv2d(3, 64)    -> ReLU -> MaxPool2d
Conv2d(64, 128)  -> ReLU -> MaxPool2d
Conv2d(128, 256) -> ReLU -> MaxPool2d
Flatten
Dropout(0.25)
Linear(256*5*5, 512) -> ReLU
Dropout(0.25)
Linear(512, 1)          <- raw regression output
```

## Output

The network outputs a raw scalar value. For the Trinder-1 training setup
(labels 0-7 corresponding to true concentrations 1-8 mmol/L), add 1 to the
raw mean to obtain the final glucose concentration:

```
predicted_concentration (mmol/L) = raw_mean + 1
```

**Unit**: mmol/L (glucose concentration, Trinder-1 method)
