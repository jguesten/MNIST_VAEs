# src/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=10, dim_latent = None, beta = None):
        super().__init__()
        self.conv = nn.Sequential(
        nn.Conv2d(1,32,3,padding = 1), ## 28 x 28
        nn.ReLU(),
        nn.MaxPool2d(2), ##  14 x 14
        nn.Conv2d(32,64,3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2), ##  7 x 7
        )
        self.lin = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7,num_classes)
        )
    
    def forward(self, x):
        return self.lin(self.conv(x))
    
    def loss_fn(self, batch): ## neg_elbo_loss_fn
        images, labels = batch 
        return nn.functional.cross_entropy(self(images), labels)
    
    def metrics(self, batch):
        images, labels = batch
        logits=self(images)
        val_pred = logits.argmax(dim=1)
        return {
        "val_loss" : nn.functional.cross_entropy(logits,labels).item(),
        "val_acc" : (val_pred == labels).float().mean().item()
        }
        
    
class MLP(nn.Module):
    def __init__(self, num_classes = 10, dim_latent = None, beta = None):
        super().__init__()
        self.lin = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,num_classes)
        )
    def forward(self, x):
        return self.lin(x)
    
    def loss_fn(self, batch):
        images, labels = batch 
        return nn.functional.cross_entropy(self(images), labels)
    
    def metrics(self, batch):
        images, labels = batch
        logits=self(images)
        val_pred = logits.argmax(dim=1)
        return {
        "val_loss" : nn.functional.cross_entropy(logits,labels).item(),
        "val_acc" : (val_pred == labels).float().mean().item()
        }
    
class BaseVAE(nn.Module):
    def __init__(self, dim_latent=10, beta=1.0):
        super().__init__()
        self.dim_latent = dim_latent
        self.beta = beta

    def loss_terms(self, batch):
        """Return per-image D (reconstruction) and R (KL) as tensors."""
        raise NotImplementedError

    def loss_fn(self, batch):
        terms = self.loss_terms(batch)
        return terms["D"] + self.beta * terms["R"]

    def metrics(self, batch):
        terms = self.loss_terms(batch)
        return {
            "val_loss": (terms["D"] + self.beta * terms["R"]).item(),
            "val_elbo": -(terms["D"] + terms["R"]).item(),     # beta = 1, comparable across models
            "val_D": terms["D"].item(),
            "val_R": terms["R"].item(),
        }
    
    def encode(self, x):
        """Deterministic latent code used for plots: the encoder mean."""
        return self.encoder(x)[:, 0]
    
class VAE(BaseVAE):
    def __init__(self, dim_latent = 10, beta = 1):
        super().__init__(dim_latent, beta)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,2*dim_latent),
            nn.Unflatten(1,(2, dim_latent))
        )
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dim_latent,256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,28*28),
            nn.Sigmoid(), ## Bernoulli decoder needs p
            nn.Unflatten(1,(1,28,28))
        )

    def forward(self, x):
        params = self.encoder(x)
        mu = params[:,0, :]
        logstd = params[:,1, :]  
        #logstd = logstd.clamp(-10, 5)
        eps = torch.randn_like(mu)
        z = mu + logstd.exp() * eps
        return self.decoder(z), mu,logstd

    def loss_terms(self, batch): ## neg_elbo_loss_fn
        x, labels = batch 
        recon, mu, logstd = self(x)
        recon = recon.clamp(1e-7, 1 - 1e-7)
        D = -(x * torch.log(recon) + (1 - x) * torch.log(1-recon)).sum()/len(x) ## Bernoulli Decoder
        R = (1/2 * (mu**2+(logstd.exp())**2-1-2*logstd)).sum()/len(x)
        return {"D":D, "R":R}
    
class cVAE(BaseVAE):
    def __init__(self, dim_latent = 10, beta = 1):
        super().__init__(dim_latent, beta)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,2*dim_latent),
            nn.Unflatten(1,(2, dim_latent))
        )
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dim_latent,256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,28*28),
            nn.Sigmoid(), ## Bernoulli decoder needs p
            nn.Unflatten(1,(1,28,28))
        )
        self.classmeans=nn.Embedding(10,dim_latent)

    def forward(self, x):
        params = self.encoder(x)
        mu = params[:,0, :]
        logstd = params[:,1, :]  
        #logstd = logstd.clamp(-10, 5)
        eps = torch.randn_like(mu)
        z = mu + logstd.exp() * eps
        return self.decoder(z), mu,logstd

    def loss_terms(self, batch): ## neg_elbo_loss_fn
        x, labels = batch 
        recon, mu, logstd = self(x)
        recon = recon.clamp(1e-7, 1 - 1e-7)
        D = -(x * torch.log(recon) + (1 - x) * torch.log(1-recon)).sum()/len(x) ## Bernoulli Decoder
        R = (1/2 * (mu**2+(logstd.exp())**2- 2 * mu * self.classmeans(labels) + self.classmeans(labels)**2-1-2*logstd)).sum()/len(x)
        return {"D":D, "R":R}

class Conv_VAE(BaseVAE):
    def __init__(self,dim_latent = 10, beta = 1):
        super().__init__(dim_latent, beta)
        self.encoder = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),## 28 x 28
            nn.ReLU(),
            nn.MaxPool2d(2), ##  14 x 14
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*7*7,150),
            nn.ReLU(),
            nn.Linear(150,2*dim_latent), 
            nn.Unflatten(1,(2, dim_latent))
        )   
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dim_latent,150),
            nn.ReLU(),
            nn.Linear(150,64*7*7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),                         # -> (64, 7, 7)
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        ) ## Bernoulli decoder needs p

    def forward(self, x):
        params = self.encoder(x)
        mu = params[:,0, :]
        logstd = params[:,1, :]
        #logstd = logstd.clamp(-15, 10)
        eps = torch.randn_like(mu)
        z = mu + logstd.exp() * eps
        return self.decoder(z), mu,logstd
    
    def loss_terms(self, batch): ## neg_elbo_loss_fn
        x, labels = batch 
        recon, mu, logstd = self(x)
        recon = recon.clamp(1e-7, 1 - 1e-7)
        D = -(x * torch.log(recon) + (1 - x) * torch.log(1-recon)).sum()/len(x) ## Bernoulli Decoder  
        R = (1/2 * (mu**2+(logstd.exp())**2-1-2*logstd).sum(dim=0)).sum()/len(x)
        return {"D":D, "R":R}

class MaskedLinear(nn.Linear):
    def __init__(self, dim_input, dim_output, mask):
            super().__init__(dim_input,dim_output)
            self.register_buffer("mask", mask)              # shape (out_features, in_features)

    def forward(self, x):
            return F.linear(x, self.weight * self.mask, self.bias)
    
class MADE(nn.Module):
    def __init__(self, input_dim, hidden = (512,256)):
        super().__init__()
        degrees = [torch.arange(1, input_dim + 1)] # tensor([1, 2, ..., latent_dim])
        for h in hidden:
            low = degrees[-1].min().item()
            degrees.append(torch.randint(low, input_dim, (h,)))
        
        layers = []
        for i in range(len(hidden)):
                mask = (degrees[i+1][:,None] >= degrees[i][None,:]).float()
                layers.append(MaskedLinear(len(degrees[i]),len(degrees[i+1]), mask))
                layers.append(nn.ReLU())

        outmask = (degrees[-1][None,:] < torch.cat([degrees[0], degrees[0]])[:,None]).float()
        layers.append(MaskedLinear(len(degrees[-1]),2*len(degrees[0]), outmask))
        self.net = nn.Sequential(*layers)
    
    def forward(self,x):
        return self.net(x)

class IAF_conv_VAE(BaseVAE):
    def __init__(self, dim_latent = 10, beta = 1):
            super().__init__(dim_latent, beta)
            self.made = MADE(dim_latent)
            self.encoder = nn.Sequential(
                nn.Conv2d(1,32,3,padding=1), ## 28 x 28
                nn.ReLU(),
                nn.MaxPool2d(2), ##  14 x 14
                nn.Conv2d(32,64,3,padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(64*7*7,150),
                nn.ReLU(),
                nn.Linear(150,2*dim_latent), 
                nn.Unflatten(1,(2, dim_latent))
            )   
            self.decoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(dim_latent,150),
                nn.ReLU(),
                nn.Linear(150,64*7*7),
                nn.ReLU(),
                nn.Unflatten(1, (64, 7, 7)),                         # -> (64, 7, 7)
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
                nn.Sigmoid()
        ) ## Bernoulli decoder needs p
    
    def IAF_flow(self, z):
        out = self.made(z)
        mu_1, s_1 = out[:, :self.dim_latent], out[:, self.dim_latent:]
        std_1 = torch.sigmoid(s_1)
        z = std_1 * z + (1 - std_1) * mu_1
        log_det = F.logsigmoid(s_1).sum() # log |det dz_T/dz_0| = sum of log sigma
        self.std_1 = std_1
        self.mu_1 = mu_1
        self.logstd_1 = F.logsigmoid(s_1)
        self.log_det=log_det
        return z

    def forward(self, x):
        params = self.encoder(x)
        mu_0 = params[:,0, :]
        logstd_0 = params[:,1, :]
        self.logstd_0=logstd_0
        #logstd = logstd.clamp(-15, 10)
        eps = torch.randn_like(mu_0)
        z_0 = mu_0 + logstd_0.exp() * eps
        self.z_0 = z_0
        self.z_1 = self.IAF_flow(z_0)
        ##self.z = self.IAF_flow(z)
        return self.decoder(self.z_1)

    def loss_terms(self, batch): ## neg_elbo_loss_fn
        x, labels = batch 
        recon = self(x)
        recon = recon.clamp(1e-7, 1 - 1e-7)
        D = -(x * torch.log(recon) + (1 - x) * torch.log(1-recon)).sum()/len(x) ## Bernoulli Decoder
        #R = 1/2 * (mu**2+(logstd.exp())**2+1-torch.log((logstd.exp())**2)).sum() ## analytical -KL(q(z|x)||p(z))

        log_q0 = (-0.5-self.logstd_0 - self.logstd_1)   # analytic E[log q(z'|x)]
        log_p = 0.5 * ((self.std_1 * self.z_0 + (1- self.std_1)*self.mu_1)**2)# -log p(z'), sampled
        R = (log_q0 + log_p).sum()/len(x)
        return {"D":D, "R":R}  
    
    def encode(self, x):
        """Encoder mean passed through the flow, i.e. a point in the decoder's latent space."""
        return self.IAF_flow(self.encoder(x)[:, 0])

MODELS = {
    "VAE": VAE,
    "cVAE": cVAE,
    "Conv_VAE": Conv_VAE,
    "IAF_conv_VAE": IAF_conv_VAE,
    "MLP": MLP,
    "CNN": CNN,
}

def build_model(cfg):
    params = {k: v for k, v in cfg.items() if k != "type"}
    return MODELS[cfg["type"]](**params)