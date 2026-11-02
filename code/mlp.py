from common import *

model = nn.Sequential(nn.Flatten(), nn.Linear(12288, 256), nn.ReLU(),
                      nn.Linear(256, 10))
print('parameters:', sum([p.numel() for p in model.parameters()]))
print('train', len(y_train), 'val', len(y_val), 'test', len(y_test))

epochs = 20
loss, val = fit(model, epochs, 'models/mlp.pt')
for e in range(epochs):
    print(f'epoch {e + 1}: train loss {loss[e]:.4f}, val acc {val[e]:.4f}')

guesses = predict(model, x_test)
print(f'test accuracy {mean(guesses == y_test.numpy()):.4f}')
np.save('outputs/pred_mlp.npy', guesses)

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
suptitle('MLP training curve')
tight_layout()
savefig('outputs/fig_curve_mlp.png', dpi=150)
close()
