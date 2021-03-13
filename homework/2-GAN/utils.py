import torch
from torch import nn


def compute_gradient_penalty(critic, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(real_samples.device())
    mix = alpha * real_samples + (1 - alpha) * fake_samples
    mix = mix.requires_grad_(True)
    out, _ = critic(mix)

    ones = torch.ones(out.size()).to(real_samples.device())
    grads = torch.autograd.grad(
        outputs=out,
        inputs=mix,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
        only_inputs=True
    )[0]
    grads = grad.view(grad.size(0), -1)
    grad_norm = torch.sqrt(torch.sum(grads**2, dim=1) + 1e-12)
    grad_penalty = (grad_norm - 1).pow(2).mean()
    return grad_penalty

def permute_labels(labels):
    # YOUR CODE
