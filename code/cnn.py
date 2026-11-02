from common import *
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(4096, 64), nn.ReLU(), nn.Linear(64, 10))
print('parameters:', sum([p.numel() for p in model.parameters()]))
print('train', len(y_train), 'val', len(y_val), 'test', len(y_test))

epochs = 10
loss, val = fit(model, epochs, 'models/cnn.pt')
for e in range(epochs):
    print(f'epoch {e + 1}: train loss {loss[e]:.4f}, val acc {val[e]:.4f}')

# results on the test set
guesses = predict(model, x_test)
np.save('outputs/pred_cnn.npy', guesses)
print(classification_report(y_test, guesses, target_names=names, digits=4))

x = range(1, epochs + 1)
figure(figsize=(9, 3.5))
subplot(1, 2, 1)
plot(x, loss, 'o-')
xlabel('epoch')
ylabel('mean train loss')
grid(alpha=0.3)
subplot(1, 2, 2)
plot(x, val, 'o-', color='orange')
xlabel('epoch')
ylabel('validation accuracy')
grid(alpha=0.3)
suptitle('CNN training curve')
tight_layout()
savefig('outputs/fig_curve_cnn.png', dpi=150)
close()

figure(figsize=(8, 7))
ConfusionMatrixDisplay.from_predictions(y_test, guesses, display_labels=names,
                                        xticks_rotation=90, cmap='Blues', ax=gca())
title('CNN confusion matrix, test set')
tight_layout()
savefig('outputs/fig_cm_cnn.png', dpi=150)
close()
