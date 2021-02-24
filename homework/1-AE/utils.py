import torch 
import torch.nn as nn
import torch.nn.functional as F


def custom_cross_entropy(model_output, images, labels, model):
    criterion = nn.CrossEntropyLoss()
    return criterion(model_output, labels)


def custom_mse_loss(model_output, images, labels, model):
  return F.mse_loss(model_output, images)


def noisy_preprocess(data):
  return data + torch.randn_like(data) * 0.1


def l1_loss(x):
    return torch.mean(torch.sum(torch.abs(x), dim=1))


def sparse_loss(model_output, images, labels, model):
    loss = 0
    x = images
    for block in model.encoder[:-1]:
      x = block.conv(x)
      loss += l1_loss(x)
      x = block.activ(block.batchnorm(x))
    x = model.encoder[-1](x)
    loss += l1_loss(x)

    for block in model.decoder[:-1]:
      x = block.conv(x)
      loss += l1_loss(x)
      x = block.activ(block.batchnorm(x))
    x = model.decoder[-1](x)
    loss += l1_loss(x)

    total_loss = 0.001 * loss + F.mse_loss(model_output, images)
    return total_loss