# multilayer_perceptron
Classify **benign** vs **malignant** tumor by implementing a [Feedforward Neural Network](https://en.wikipedia.org/wiki/Feedforward_neural_network) in Python. (42 Silicon Valley)

<p float="left">
  <img src="https://github.com/ashih42/multilayer_perceptron/blob/master/Screenshots/pair_plot.png" width="350" />
  <img src="https://github.com/ashih42/multilayer_perceptron/blob/master/Screenshots/online.png" width="360" />
</p>

## Prerequisites

You have `python3` installed.

## Installing

```
./setup/setup.sh
```

## Running

### Describing Data
```
python3 describe.py data.csv
```

### Histograms
```
python3 histogram.py data.csv
```

### Scatter Plots
```
python3 scatter_plot.py data.csv
```

### Scatter Plot Matrix
```
python3 pair_plot.py data.csv
```

### Training

#### Split `data.csv` into **training set** and **test set**.
```
python3 split_data.py
```
* Here I use the test set as the cross-validation set.

#### Train with cross-validation.

```
python3 train.py training_data.csv validation_data.csv [ -o ]
```
* Use *batch* gradient-descent by default.
  * Converges *slowly*.
* `-o` Use **online** gradient-descent.
  * Can converge *quickly*.
* Export environment variable `MP_PLOTS`=`TRUE` to show plots during training.

### Predicting
```
python3 predict.py param.dat data.csv
```
