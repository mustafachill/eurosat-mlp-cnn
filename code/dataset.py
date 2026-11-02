from pylab import *
import os
import torch
from torch import nn

torch.manual_seed(0)
rng = np.random.default_rng(0)

folder = 'data/EuroSAT_RGB'
names = sorted(os.listdir(folder))

x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], []
for c in range(10):
    files = sorted(os.listdir(folder + '/' + names[c]))
    n = len(files)
    # shuffle each class: 72% train, 8% validation, 20% test
    order = rng.permutation(n)
    part = zeros(n, int)
    part[order[int(n * 0.72):]] = 1
    part[order[int(n * 0.8):]] = 2
    for i in range(n):
        img = imread(folder + '/' + names[c] + '/' + files[i])
        if part[i] == 0:
            x_train.append(img)
            y_train.append(c)
        elif part[i] == 1:
            x_val.append(img)
            y_val.append(c)
        else:
            x_test.append(img)
            y_test.append(c)

def to_tensor(images):
    # pytorch wants the colour first: (n, 3, 64, 64), values between 0 and 1
    a = array(images, float32).transpose(0, 3, 1, 2) / 255
    return torch.tensor(a)

x_train, x_val, x_test = to_tensor(x_train), to_tensor(x_val), to_tensor(x_test)
y_train = torch.tensor(y_train)
y_val = torch.tensor(y_val)
y_test = torch.tensor(y_test)
