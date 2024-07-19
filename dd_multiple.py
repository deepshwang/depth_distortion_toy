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

def pdf(mu, std, mu_c):
    return torch.exp(-(mu-mu_c)**2/(2*std**2))

def plot_scene(u_mus, t_mus, u_stds, wis, savedir, u_c_min, u_c_max, u_cs):
    plt.axis([u_c_min, u_c_max, -2, t_far])
    colormap = cm.get_cmap('jet')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("Screen Space (u/v)")
    plt.ylabel("Ray (t)")
    for i, (u_mu, t_mu, u_std, wi, u_c) in enumerate(zip(u_mus, t_mus, u_stds, wis, u_cs)):
        sc = plt.scatter(u_mu, t_mu, s=wi*1000, color=cm.jet(i/wis.shape[0]), cmap=colormap) 
        # plt.colorbar(sc)
        plt.plot([u_mu - 2*u_std, u_mu + 2*u_std], [t_mu, t_mu], 'black', linestyle="--", linewidth=0.3)
        plt.plot([u_c, u_c], [-2, t_far], color=cm.jet(i/wis.shape[0]), linewidth=1.2)
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
    n_iters = 250
    u_c_min = -20
    u_c_max = 20
    u_c_interval = 2
    u_cs = list(range(u_c_min, u_c_max+u_c_interval, u_c_interval))
    savedir = "./test_multiple"
    os.makedirs(savedir, exist_ok=True)

    ######################
    mus = []
    stds = []
    opacities = []
    for u_c in u_cs:
        u_mu = (u_far - u_near) * np.random.rand(n_samples) + u_near + u_c
        t_mu = np.linspace(t_near, t_far, n_samples+1, endpoint=False)[1:] + 10 * (t_far-t_near)/n_samples * np.random.randn(n_samples)
        mu = torch.from_numpy(np.stack([u_mu, t_mu], axis=1).transpose(0, 1)).to("cuda")
        std = torch.from_numpy(np.ones((mu.shape[0], 1)) * u_std).to("cuda")
        opacity = inverse_sigmoid(torch.ones(std.shape[0]) * 0.1).to("cuda")
        mus.append(mu)
        stds.append(std)
        opacities.append(opacity)
    mu = torch.stack(mus)
    std = torch.stack(stds)
    opacity = torch.stack(opacities)
    mu.requires_grad = True
    std.requires_grad = True
    opacity.requires_grad = True
    optimizer = torch.optim.Adam([mu, std, opacity], lr=1e-1)
    u_cs = torch.Tensor(u_cs).cuda()[..., None]

    for iter in tqdm(range(n_iters), total=n_iters):
        sidxs = torch.argsort(mu[..., 1])
        mu_sorted = []
        std_sorted = []
        for i, sidx in enumerate(sidxs):
            mu_sorted.append(mu[i, sidx, :])
            std_sorted.append(std[i, sidx, :])
        mu_sorted = torch.stack(mu_sorted)
        std_sorted = torch.stack(std_sorted)
        alpha = torch.sigmoid(opacity)
        aGi = alpha * pdf(mu_sorted[..., 0], std_sorted[..., 0], u_cs)
        w = torch.cat((torch.zeros(aGi.shape[0], 1).cuda(), aGi), dim=-1)
        w = torch.cumprod((1-w), dim=-1)[..., :-1]
        wi = aGi * w
        wi = wi.detach()
        
        # See 2DGS appendix. A for details
        Ai = torch.cumsum(wi, dim=-1)
        mi = mu_sorted[..., 1]
        Di = torch.cumsum(wi * mi, dim=-1)
        Di_sq = torch.cumsum(wi * (mi**2), dim=-1)

        d_loss = torch.sum(wi * ((mi**2) * Ai + Di_sq - 2 * mi * Di))
        d_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        with torch.no_grad():
            plot_scene(mu_sorted.detach().cpu().numpy()[..., 0], mu_sorted.detach().cpu().numpy()[..., 1], torch.squeeze(std_sorted).detach().cpu().numpy(), wi.detach().cpu().numpy(), savedir, u_c_min, u_c_max, torch.squeeze(u_cs).detach().cpu().numpy())
    os.system("conda deactivate")
    os.system(f"cd {savedir}")
    os.system("ffmpeg -framerate 60 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p out.mp4")
    os.system("conda activate gaussian_splatting")