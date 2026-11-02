from dataset import *

def predict(model, x):
    model.eval()
    guesses = []
    with torch.no_grad():
        for i in range(0, len(x), 500):
            guesses.append(model(x[i:i + 500]).argmax(1))
    return torch.cat(guesses).numpy()

def fit(model, epochs, path):
    # if the model is already saved, load it and skip training
    if os.path.exists(path):
        saved = torch.load(path)
        model.load_state_dict(saved['model'])
        return saved['loss'], saved['val']
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(0)
    loss, val = [], []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(y_train), generator=generator)
        total = 0
        for i in range(0, len(y_train), 64):
            batch = order[i:i + 64]
            optimizer.zero_grad()
            batch_loss = loss_fn(model(x_train[batch]), y_train[batch])
            batch_loss.backward()
            optimizer.step()
            total += batch_loss.item() * len(batch)
        loss.append(total / len(y_train))
        guesses = predict(model, x_val)
        val.append(float(mean(guesses == y_val.numpy())))
    torch.save({'model': model.state_dict(), 'loss': loss, 'val': val}, path)
    return loss, val
