# small local web page to try the two models
# run it from the final-project folder: python3 demo/app.py
import json
import os
import numpy as np
import torch
from torch import nn
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

folder = 'data/EuroSAT_RGB'
names = sorted(os.listdir(folder))

# same split as code/dataset.py, so the test images are in the same order
rng = np.random.default_rng(0)
test, ytest, split = [], [], []
for c in range(10):
    files = sorted(os.listdir(folder + '/' + names[c]))
    n = len(files)
    order = rng.permutation(n)
    part = np.zeros(n, int)
    part[order[int(n * 0.72):]] = 1
    part[order[int(n * 0.8):]] = 2
    for i in range(n):
        if part[i] == 2:
            test.append('/img/' + names[c] + '/' + files[i])
            ytest.append(c)
    # images, train, val, test of this class for the table on the page
    split.append([n, int(sum(part == 0)), int(sum(part == 1)), int(sum(part == 2))])

# same layers as code/mlp.py and code/cnn.py
models = {}
models['MLP'] = nn.Sequential(nn.Flatten(), nn.Linear(12288, 256), nn.ReLU(),
                              nn.Linear(256, 10))
models['CNN'] = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(4096, 64), nn.ReLU(), nn.Linear(64, 10))

info = {'names': names, 'split': split, 'test': test, 'models': {}}
for m in models:
    saved = torch.load('models/' + m.lower() + '.pt')
    models[m].load_state_dict(saved['model'])
    models[m].eval()

    # confusion matrix and metrics from the saved test predictions, like compare.py
    pred = np.load('outputs/pred_' + m.lower() + '.npy')
    cm = np.zeros((10, 10), int)
    for i in range(len(ytest)):
        cm[ytest[i], pred[i]] += 1
    precision = np.diag(cm) / cm.sum(axis=0)
    recall = np.diag(cm) / cm.sum(axis=1)
    f1 = 2 * precision * recall / (precision + recall)

    # the layers and how many parameters each one has
    layers, layer_params, total = [], [], 0
    for layer in models[m]:
        layers.append(str(layer))
        count = 0
        for w in layer.parameters():
            count += w.numel()
        if count > 0:
            layer_params.append([str(layer).split('(')[0], count])
        total += count

    info['models'][m] = {
        'layers': layers, 'params': total, 'layer_params': layer_params,
        'epochs': len(saved['loss']), 'loss': saved['loss'], 'val': saved['val'],
        'accuracy': np.trace(cm) / cm.sum(), 'precision': precision.mean(),
        'recall': recall.mean(), 'f1': f1.mean(), 'class_recall': recall.tolist(),
        'cm': cm.tolist(), 'pred': pred.tolist(),
    }

def predict(pixels):
    # pixels: 64*64*3 numbers from 0 to 255, row by row, RGB
    x = torch.tensor(pixels, dtype=torch.float32).reshape(64, 64, 3) / 255
    x = x.permute(2, 0, 1).reshape(1, 3, 64, 64)
    res = {}
    with torch.no_grad():
        for m in models:
            p = torch.softmax(models[m](x), 1)[0].tolist()
            rows = []
            for c in range(10):
                rows.append({'name': names[c], 'p': p[c]})
            rows.sort(key=lambda r: r['p'], reverse=True)
            res[m] = rows
    return res

class Handler(BaseHTTPRequestHandler):
    def reply(self, body, kind, code=200):
        self.send_response(code)
        self.send_header('Content-Type', kind)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.path
        if p == '/':
            self.reply(open('demo/index.html', 'rb').read(), 'text/html; charset=utf-8')
        elif p == '/info':
            self.reply(json.dumps(info).encode(), 'application/json')
        elif p in test:
            self.reply(open(folder + p[4:], 'rb').read(), 'image/jpeg')
        else:
            self.reply(b'not found', 'text/plain', 404)

    def do_POST(self):
        n = int(self.headers['Content-Length'])
        pixels = json.loads(self.rfile.read(n))['pixels']
        self.reply(json.dumps(predict(pixels)).encode(), 'application/json')

print(len(test), 'test images, models loaded')
print('open http://127.0.0.1:8000  (Ctrl+C to stop)')
ThreadingHTTPServer(('127.0.0.1', 8000), Handler).serve_forever()
