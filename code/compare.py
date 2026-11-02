from common import *

y = y_test.numpy()
guesses, recall = {}, {}
for m in ['mlp', 'cnn']:
    guesses[m] = np.load('outputs/pred_' + m + '.npy')
    # confusion matrix: rows are the true class, columns the guess
    cm = zeros((10, 10), int)
    for i in range(len(y)):
        cm[y[i], guesses[m][i]] += 1
    accuracy = trace(cm) / cm.sum()
    precision = diag(cm) / cm.sum(axis=0)
    recall[m] = diag(cm) / cm.sum(axis=1)
    f1 = 2 * precision * recall[m] / (precision + recall[m])
    print(m, f'accuracy {accuracy:.4f}, precision {mean(precision):.4f}, '
             f'recall {mean(recall[m]):.4f}, f1 {mean(f1):.4f}')
    mistakes = []
    for i in range(10):
        for j in range(10):
            if i != j:
                mistakes.append([cm[i, j], names[i], names[j]])
    mistakes.sort(reverse=True)
    print(m, 'most common mistakes:')
    for k in range(4):
        print('  ', mistakes[k][1], 'taken as', mistakes[k][2], mistakes[k][0])
    figure(figsize=(7, 6))
    imshow(cm, cmap='Blues')
    for i in range(10):
        for j in range(10):
            color = 'white' if cm[i, j] > 300 else 'black'
            text(j, i, cm[i, j], ha='center', va='center', fontsize=7, color=color)
    xticks(range(10), names, rotation=90)
    yticks(range(10), names)
    xlabel('predicted class')
    ylabel('true class')
    title(m.upper() + ' confusion matrix, test set')
    tight_layout()
    savefig('outputs/fig_cm_' + m + '.png', dpi=150)
    close()

print('recall of each class (mlp, cnn) and the guesses for its first test image')
figure(figsize=(12, 6))
for c in range(10):
    i = list(y).index(c)
    mlp_guess = names[guesses['mlp'][i]]
    cnn_guess = names[guesses['cnn'][i]]
    print('  ', names[c], f"{recall['mlp'][c]:.4f} {recall['cnn'][c]:.4f}",
          mlp_guess, cnn_guess)
    subplot(2, 5, c + 1)
    imshow(x_test[i].permute(1, 2, 0).numpy())
    title(names[c] + '\nMLP: ' + mlp_guess + '\nCNN: ' + cnn_guess, fontsize=8)
    axis('off')
tight_layout()
savefig('outputs/fig_samples.png', dpi=150)
close()
