import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import matplotlib.cm as cm
import random

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

# 1D- Gaussian in 2D ray space (u-t)
def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def pdf(mu, std):
    return torch.exp(-mu**2/(2*std**2))

def plot_scene(u_mu, t_mu, u_std, wi, savedir):
    plt.axis([-5, 5, 0, 10])
    colormap = cm.get_cmap('coolwarm')
    # plt.scatter(u_mu, t_mu, color=cm.rainbow(alpha), cmap=colormap) 
    sc = plt.scatter(u_mu, t_mu, s=wi*1000, color=cm.coolwarm(wi*255), cmap=colormap) 
    # plt.colorbar(sc)
    plt.plot([u_mu - 2*u_std, u_mu + 2*u_std], [t_mu, t_mu], 'black', linestyle="--", linewidth=0.5)
    plt.plot([0, 0], [0, t_far], 'b')
    plt.savefig(os.path.join(savedir, f"it_{str(iter).zfill(4)}.png"))
    plt.clf()

if __name__ == "__main__":
    ##### parameters #####
    t_near = 0
    t_far = 10
    u_near = -1.5
    u_far = 1.5
    u_std = 1
    n_samples = 100
    alpha = 0.1
    n_iters = 150

    savedir = "./test"
    os.makedirs(savedir, exist_ok=True)

    ######################

    u_mu = (u_far - u_near) * np.random.rand(n_samples) + u_near
    t_mu = np.linspace(t_near, t_far, n_samples+1, endpoint=False)[1:]
    
    mu = torch.from_numpy(np.stack([u_mu, t_mu], axis=1).transpose(0, 1)).to("cuda")
    std = torch.from_numpy(np.ones((mu.shape[0], 1)) * u_std).to("cuda")
    opacity = inverse_sigmoid(torch.ones(std.shape[0]) * 0.1).to("cuda")
    mu.requires_grad = True
    std.requires_grad = True
    opacity.requires_grad = True
    optimizer = torch.optim.Adam([mu, std, opacity], lr=1e-1)

    for iter in tqdm(range(n_iters), total=n_iters):
        sidx = torch.argsort(mu[:, 1])
        mu_sorted = mu[sidx]
        std_sorted = std[sidx]
        alpha = torch.sigmoid(opacity)
        aGi = alpha * pdf(mu_sorted[:, 0], std_sorted[:, 0])
        
        w = torch.cat((torch.Tensor([0]).cuda(), aGi))
        w = torch.cumprod((1-w), dim=0)[:-1]
        wi = aGi * w
        
        # whether to detach weights
        # wi = wi.detach()
        # See 2DGS appendix. A for details
        Ai = torch.cumsum(wi, dim=0)
        mi = mu_sorted[:, 1]
        Di = torch.cumsum(wi * mi, dim=0)
        Di_sq = torch.cumsum(wi * (mi**2), dim=0)

        d_loss = torch.sum(wi * ((mi**2) * Ai + Di_sq - 2 * mi * Di))
        d_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        with torch.no_grad():
            plot_scene(mu.detach().cpu().numpy()[:, 0], mu.detach().cpu().numpy()[:, 1], torch.squeeze(std).detach().cpu().numpy(), wi.detach().cpu().numpy(), savedir)

