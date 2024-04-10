import torch


def train_loop(path, dataloader, model, loss_fn, optimizer, alpha):
    model.train()

    full_model_loss = 0
    full_reg_loss = 0

    for batch_idx, (X, y) in enumerate(dataloader):
        print(f"Current batch idx: {batch_idx}")

        optimizer.zero_grad()

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        reg_loss = 0
        for param in model.parameters():
            reg_loss += torch.sum(param**2)

        # Normalize losses per number of datapoints
        full_model_loss += loss.item() / y.shape[0]
        full_reg_loss += reg_loss / y.shape[0]

        loss += alpha * reg_loss

        # Backpropagation
        loss.backward()
        optimizer.step()

    else:
        # Losses were already adjusted, so now we just need to normalize per number of batches (and alpha)
        adjusted_model_loss = full_model_loss / len(dataloader)
        adjusted_reg_loss = alpha * full_reg_loss / len(dataloader)

        # Print and write info
        print(f"Train loss: {adjusted_model_loss}")
        print(f"\tReg Loss: {adjusted_reg_loss}")

        with open(path, "a+") as f:
            f.write(f"T:{adjusted_model_loss}\nR:{adjusted_reg_loss}\n")


def test_loop(path, dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y)

    with open(path, "a+") as f:
        f.write(f"V:{test_loss / size}\n")

    test_loss /= size
    print(f"Eval loss: {test_loss:>8f}")
