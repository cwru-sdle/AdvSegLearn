import torch
import matplotlib.pyplot as plt
import numpy


def train_model(
    model: torch.nn.Module,
    train_dl: torch.utils.data.DataLoader,
    val_dl: torch.utils.data.DataLoader,
    epochs: int,
    accumulation_steps: int = 5,
    optimizer = None,
    loss_fn = torch.nn.BCELoss(),
    save = None,
    learning_rate: float = 0.0001,
):
    save_path = save if save is not None else ''
    optimizer = initialize_optimizer(model, optimizer, learning_rate)
    training_losses = []
    validation_losses = []

    for epoch in range(epochs):
        print(f'Start training epoch {epoch}')
        epoch_train_losses = train_epoch(model, train_dl, optimizer, loss_fn, accumulation_steps, save_path, save, epoch, epochs)
        training_losses.append(numpy.mean(epoch_train_losses))

        print(f'Start validation epoch {epoch}')
        epoch_validation_losses = validate_epoch(model, val_dl, loss_fn)
        validation_losses.append(numpy.mean(epoch_validation_losses))

    plot_losses(training_losses, validation_losses, epochs, save_path)

    if save is not None:
        torch.save(model, save_path + '/model.pt')

def save_tensor(tensor, title, save_path):
    plt.imshow(
        tensor
        .squeeze(0)
        .cpu()
        .detach()
        .numpy()
    )
    plt.savefig(save_path + title)

def initialize_optimizer(model, optimizer, learning_rate):
    if optimizer is None:
        return torch.optim.Adam(model.parameters(), learning_rate)
    return optimizer

def train_epoch(model, train_dl, optimizer, loss_fn, accumulation_steps, save_path, save, epoch, epochs):
    train_losses = []
    optimizer.zero_grad()
    for idx, datapoint in enumerate(train_dl):
        model_output = model(datapoint[0])
        loss = loss_fn(model_output, datapoint[1])

        loss.backward()
        if (idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        train_losses.append(loss.item())

        if (idx + 1) == len(train_dl) and epoch == epochs - 1 and save is not None:
            save_tensor(model_output.select(1, 0), '/net_output_model_fatigue.png', save_path)
            save_tensor(model_output.select(1, 1), '/net_output_model_overload.png', save_path)
            save_tensor(datapoint[1].select(1, 0), '/net_output_expected_fatigue.png', save_path)
            save_tensor(datapoint[1].select(1, 1), '/net_output_expected_overload.png', save_path)
    return train_losses

def validate_epoch(model, val_dl, loss_fn):
    valid_losses = []
    for idx, datapoint in enumerate(val_dl):
        model_output = model(datapoint[0])
        loss = loss_fn(model_output, datapoint[1])
        valid_losses.append(loss.item())
    return valid_losses

def plot_losses(training_losses, validation_losses, epochs, save_path):
    plt.plot(range(epochs), training_losses, label='Training Loss')
    plt.plot(range(epochs), validation_losses, label='Validation Loss')
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path + '/loss_figure.png')
    else:
        plt.show()

