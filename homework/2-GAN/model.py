import torch
from torch import nn
from torch import optim
from utils import compute_gradient_penalty, permute_labels
from calculate_fid import calculate_fid
from torch.utils.data import DataLoader

import wandb


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, c_dim=10):
        super().__init__()

        self.c_dim = c_dim
        
        self.down_sampling = nn.Sequential(
            nn.Conv2d(3 + c_dim, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(1278, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

        self.bottleneck = nn.Sequential(
            *[ResidualBlock(256, 256) for _ in range(6)]
        )

        self.up_sampling = nn.Sequential( 
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh()
        )
        
    def forward(self, x, labels):
        c = labels.view(labels.size(0), labels.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        inp = torch.cat([x, c], dim=1)
        inp = self.down_sampling(inp)
        inp = self.bottleneck(inp)
        inp = self.up_sampling(inp)
        return inp

        
class Critic(nn.Module):
    def __init__(self, c_dim=10):
        super().__init__()
        layers = []

        in_channels = 3
        out_channels = 64
        for _ in range(6):
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
            )
            layers.append(
                nn.LeakyReLU(0.01)
            )
            in_channels = out_channels
            out_channels *= 2
        
        self.network = nn.Sequential(*layers)
        self.out_src = nn.Conv2d(out_channels, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.out_cls = nn.Conv2d(out_channels, c_dim, kernel_size=1, bias=False)
        

    def forward(self, x):
        x = self.network(x)
        out_src = self.out_src(x)
        out_cls = self.out_cls(x)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
        

class StarGAN:
    def __init__(self, train_dataset, test_dataset, config, classifier=None):
        self.G = Generator(config['c_dim'])
        self.D = Critic(config['c_dim'])
        
        if not classifier:
            model = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=True)
            module_list = list(model.children())[:-1]
            module_list.append(nn.Flatten())
            self.classifier = nn.Sequential(*module_list)
        else:
            self.classifier = classifier

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.c_dim = config['c_dim']
        self.lam_cls = config['lam_cls']
        self.lam_rec = config['lam_rec']
        self.lam_gp = config['lam_gp']
        
    def train(self):
        self.G.train()
        self.D.train()
        
    def eval(self):
        self.G.eval()
        self.D.eval()

    def to(self, device):
        self.D.to(device)
        self.G.to(device)
        
    def lr_decay(self, optimizer, lr_start, iter_curr, iter_total):
        coef = 1.0 - iter_curr / iter_total
        lrnow = lr_start * coef
        optimizer.param_groups[0]['lr'] = lrnow
    
    def log_d_losses(d_loss):
        wandb.log(d_loss['Total'])
        wandb.log(d_loss['EMD'])
        wandb.log(d_loss['Classification'])
        wandb.log(d_loss['Gradient Penalty'])
    
    def log_g_losses(g_loss):
        wandb.log(g_loss['Total'])
        wandb.log(g_loss['EMD'])
        wandb.log(g_loss['Classification'])
        wandb.log(g_loss['Reconstruction'])
    
    def save_optim_states(self, g_optimizer, d_optimizer, iteration, best=False):
        if best:
            iteration = 'best'
        torch.save(g_optimizer.state_dict(), f'g_optimizer_{iteration}.pkl')
        torch.save(d_optimizer.state_dict(), f'd_optimizer_{iteration}.pkl')
    
    def save_model_states(self, iteration, best=False):
        if best:
            iteration = 'best'
        torch.save(self.G.state_dict(), f'generator_{iteration}.pkl')
        torch.save(self.D.state_dict(), f'discriminator_{iteration}.pkl')
    
    
    def start_training(self, train_config, device):
        self.train()
        self.to(device)
        epochs = train_config['epochs']
        batch_size = train_config['batch_size']
        g_lr = train_config['g_lr']
        d_lr = train_config['d_lr']
        n_critic = train_config['n_critic']
        beta_1 = train_config['beta_1']
        beta_2 = train_config['beta_2']
        decay_start = train_config['decay_start']

        decay_epochs = epochs - decay_start

        dataloader = DataLoader(self.dataset, batch_size=16, shuffle=True)
        g_optim = optim.Adam(self.G.parameters(), g_lr, [beta_1, beta_2])
        d_optim = optim.Adam(self.D.parameters(), d_lr, [beta_1, beta_2])

        d_iter_curr = 0
        g_iter_curr = 0
        best_fid = float('inf')
        for ep in range(epochs):
            for images, labels in dataloader:
                d_iter_curr += 1
                images_real = images.to(device)
                labels_source = labels.to(device)
                rand_idx = torch.randperm(labels_source.size(0))
                labels_target = labels_source[rand_idx]

                d_loss = self.trainD(images_real, labels_source, labels_target, d_optim)
                self.log_d_losses(d_loss)

                if iter_curr % n_critic == 0:
                    g_iter_curr += 1
                    g_loss = self.trainG(images_real, labels_source, labels_target, g_optim)
                    self.log_g_loss(g_loss)
            
            if ep > decay_start:
                self.lr_decay(d_optim, d_lr, ep - decay_start, decay_epochs)
                self.lr_decay(g_optim, g_lr, ep - decay_start, decay_epochs)
            
            if ep % 3 == 0:
                test_dataloader = Dataloader(self.test_dataset, batch_size=batch_size)
                fid = calculate_fid(test_dataloader, self.G, self.classifier)

                if fid > best_fid:
                    best_fid = fid
                    self.save_model_states(ep, best=True)
                    self.save_optim_states(g_optim, d_optim, ep, best=True)
                wandb.log(fid)
            
            self.save_model_states(ep)
            self.save_optim_states(g_optim, d_optim, ep)


    def trainG(self, images_real, label_source, label_target, optimizer):
        BCE = nn.BCEWithLogitsLoss()

        images_fake = self.G(images_real, labels_target)
        images_rec = self.G(images_fake, label_source)
        out_src, out_cls = self.D(images_fake)

        g_loss_fake = -torch.mean(out_src)
        g_loss_cls = torch.mean(BCE(out_cls, labels_target))
        g_loss_rec = torch.mean(torch.abs(images_real - images_rec))
        
        total_loss = g_loss_fake + g_loss_cls + g_loss_rec
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return total_loss.item(), g_loss_fake.item(), g_loss_cls.item() + g_loss_rec.item()

    def trainD(self, images_real, label_source, label_target, optimizer):
        BCE = nn.BCEWithLogitsLoss()

        images_fake = self.G(images_real, labels_target)

        out_src_real, out_cls_real = self.D(images_real)
        out_src_fake, out_cls_fake = self.D(images_fake.detach())

        d_loss_real = -torch.mean(out_src_real)
        d_loss_fake = torch.mean(out_src_fake)

        d_loss_cls = torch.mean(BCE(out_cls_real, labels_source))

        grad_penalty = compute_gradient_penalty(self.D, images_real, images_fake)

        d_loss_wd = d_loss_real + d_loss_fake
        total_loss = d_loss_wd + self.lam_cls * d_loss_cls + self.lam_gp * grad_penalty
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return total_loss.item(), d_loss_wd.item(), d_loss_cls.item(), grad_penalty.item()

    def generate(self, image, label):
        with torch.no_grad():
            return self.G(image, label)