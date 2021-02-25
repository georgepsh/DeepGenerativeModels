import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from train import device
import wandb

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

def plot_samples(model, dataset, num_samples):
  test_loader = DataLoader(dataset, batch_size=num_samples, drop_last=True)

  test_batch = next(iter(test_loader))
  model.eval()

  for i, image in enumerate(test_batch[0]):
      model_output = model(image.unsqueeze(0).to(device)).detach().to('cpu').squeeze().numpy()
      image = image.squeeze(0).cpu().numpy()
      wandb.log({f"{model.name}_example_{i+1}": [wandb.Image(image), wandb.Image(model_output)]})


def get_latent_features(model, dataset):
    features = np.array)
    labels = []
    model.eval()
    dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False)
    
    for (image, target) in dataloader:
        image = image.to(device)
        target = target.cpu().numpy()
        latent = model.get_latent_features(image).cpu().numpy()
        features.extend(latent)
        labels.extend(target)

    return features.T, labels