import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def compute_validation_loss(model, valid_loader, loss_function, data_preprocess):
    with torch.no_grad():
        loss = 0
        count = 0
        for idx, batch in enumerate(valid_loader):
            count += 1
            images, labels = batch
            images = images.to(device)
            if data_preprocess is not None:
              preprocessed_images = data_preprocess(images)
            else:
              preprocessed_images = images.clone()
            model_output = model(preprocessed_images)
            loss += loss_function(model_output, images, labels, model).item()
    return loss / count


def train(model, train_dataset, valid_dataset, batch_size, epochs, lr, train_loss_function, valid_loss_function, data_preprocess=None):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, drop_last=False)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses_list = []
    validation_losses_list = []
    best_valid_loss = float('inf')
    model.train()
    loss_decreases = 0
    for ep in range(epochs):
        for idx, batch in tqdm(enumerate(train_dataloader), desc='train loop', leave=True):
            images, labels = batch
            images = images.to(device)
            if data_preprocess is not None:
              preprocessed_images = data_preprocess(images)
            else:
              preprocessed_images = images.clone()
            model_output = model(preprocessed_images)
            optimizer.zero_grad()
            loss = train_loss_function(model_output, images, labels, model)
            loss.backward()
            optimizer.step()
            losses_list.append(loss.item())
        
        valid_loss = compute_validation_loss(model, valid_dataloader, valid_loss_function, data_preprocess)
        validation_losses_list.append(valid_loss)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            loss_decreases = 0 
        else:
            loss_decreases += 1
            if loss_decreases > 2:
                return losses_list, validation_losses_list

        
    return losses_list, validation_losses_list