# EuroSAT land use, MLP vs CNN

My final project. Two neural networks classify 64x64 satellite images from the
EuroSAT dataset into 10 land use classes (Forest, River, Highway and so on). One
is a simple MLP, the other a small CNN. The full write up is in
`Final Project Mustafa Cil.docx`.

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
sh data/get_data.sh
```

`get_data.sh` downloads the images (about 95 MB) from Zenodo into
`data/EuroSAT_RGB`. Skip it if that folder is already there.

## Running

```
cd ~/Developer/eurosat-mlp-cnn
source .venv/bin/activate
python code/mlp.py
python code/cnn.py
```

- `mlp.py` trains the MLP for 20 epochs and prints the loss and validation
  accuracy per epoch. Then it tests the model and prints the scikit-learn
  classification report (precision, recall, F1, accuracy) and draws the
  confusion matrix.
- `cnn.py` does the same for the CNN with 10 epochs.

`dataset.py` and `common.py` are helper files. The other scripts import them,
so there is no need to run them yourself. `dataset.py` loads the images and
splits them into train, validation and test, and `common.py` has the training
loop and the predict function.

The trained models are saved in `models/`, so the scripts just load them and
finish in a few seconds. To train from scratch, delete `models/mlp.pt` and
`models/cnn.pt`. On my laptop the MLP takes under a minute and the CNN about
5 minutes. All plots go to `outputs/`.

## Demo

```
sh demo/start.sh
```

Starts a small local web page on http://127.0.0.1:8000 and opens it. You can
pick a random test image or upload your own and see what both models think.
The page also shows the data, the models, the training curves and the
confusion matrices. If something is missing (libraries, images, models), the
script sets it up first. Ctrl+C stops it.
