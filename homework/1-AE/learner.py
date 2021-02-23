import torch
import torch.nn as nn
import torch.optim as optim


def train(model, dataset, batch_size, epochs, lr, loss_function, data_preprocess):
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses_list = []
    for ep in range(epochs):
        for idx, batch in enumerate(dataloader):
            images, labels = batch
            preprocessed_images = data_preprocess(images)
            model_output = model(preprocessed_images)
            loss = loss_function(images, model_output)
            loss.backward()
            optimizer.step()
            losses_list.append(loss.item())
    return losses_list