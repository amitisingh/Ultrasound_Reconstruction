import os
import argparse
import numpy as np
import torch
import h5py
import scipy.sparse as sp

from scipy.io import loadmat
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from step1_utils.models.unet import create_model
import step1_utils.utils as utils
from step1_utils.DDIM_sampler import Sampler as DDIM
from step1_utils.data.dataloader import get_dataset, get_dataloader

def pack_flat_to_3ch(A_funcs, x_flat):
    B, N = x_flat.shape
    assert N == A_funcs.N, f"Expected N={A_funcs.N}, got {N}"

    x0 = x_flat.reshape(B, A_funcs.H, A_funcs.W)
    zeros = torch.zeros_like(x0)
    return torch.stack([x0, zeros, zeros], dim=1)   # (B, 3, H, W)


def load_BH_as_torch_csr(mat_path, group="BH", device="cuda", dtype=torch.float32):
    with h5py.File(mat_path, "r") as f:
        grp = f[group]
        data = grp["data"][:]
        ir   = grp["ir"][:]
        jc   = grp["jc"][:]

    ncols = jc.size - 1
    nrows = int(ir.max()) + 1

    BH_csc = sp.csc_matrix(
        (data.astype(np.float32), ir.astype(np.int64), jc.astype(np.int64)),
        shape=(nrows, ncols)
    )
    BH_csr = BH_csc.tocsr()

    crow = torch.from_numpy(BH_csr.indptr.astype(np.int64))
    cols = torch.from_numpy(BH_csr.indices.astype(np.int64))
    vals = torch.from_numpy(BH_csr.data.astype(np.float32))

    BH = torch.sparse_csr_tensor(crow, cols, vals, size=(nrows, ncols), dtype=dtype)
    BH = BH.to(device)

    return BH, BH.transpose(0,1)

class BHOperator:
    def __init__(self, BH, BH_T, H=256, W=256, channels=3, device="cuda", scale=None):
        self.BH = BH
        self.BH_T = BH_T
        self.N = H * W
        self.H = H
        self.W = W
        self.C = channels
        self.device = device
        self.scale = scale

    def A(self, x):
        B, C, H, W = x.shape
        assert C == 3, "diffusion model is RGB; got non-3-channel input."

        x0 = x[:,0].view(B, self.N).T    
        y0 = torch.sparse.mm(self.BH, x0).T

        if self.scale is not None:
            y0 = self.scale * y0

        y = torch.cat([y0, y0, y0], dim=1)
        return y

    @torch.no_grad()
    def At(self, r):
        B = r.shape[0]
        r0 = r[:, :self.N]

        R = r0.T
        x0 = torch.sparse.mm(self.BH_T, R).T

        if self.scale is not None:
            x0 = (1.0/self.scale) * x0

        grad_flat = torch.cat([x0, torch.zeros_like(x0), torch.zeros_like(x0)], dim=1)
        return grad_flat



# Config
class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.conf = None

        self.parser.add_argument('--dataset', type=str, default="CelebA_HQ")
        self.parser.add_argument('--out_path', type=str, default="results/dds_us")
        self.parser.add_argument('--sigma_y', type=float, default=0.0)
        self.parser.add_argument('--ps_type', type=str, default="DPS")

        self.parser.add_argument('--diff_timesteps', type=int, default=1000)
        self.parser.add_argument('--desired_timesteps', type=int, default=1000)
        self.parser.add_argument('--eta', type=float, default=1.0)
        self.parser.add_argument('--schedule', type=str, default="1000", help="regular/irregular schedule to use (jumps)")

        # image params
        self.parser.add_argument('--H', type=int, default=256)
        self.parser.add_argument('--W', type=int, default=256)
        self.parser.add_argument('--channels', type=int, default=3)


        # BH params
        self.parser.add_argument('--bh_mat', type=str, default="BH.mat")
        self.parser.add_argument('--bh_group', type=str, default="BH")

        # By
        self.parser.add_argument('--by_mat', type=str, default="By.mat")

    def parse(self):
        self.conf = self.parser.parse_args()
        return self.conf


class posterior_samplers():
    def __init__(self, conf, sampler_operator, score_model):
        self.conf = conf
        self.sampler_operator = sampler_operator
        self.score_model = score_model

    def predict_x0_hat(self, x_t, t, model_output):
        Ct = 1/torch.sqrt(self.sampler_operator.alphas_cumprod)
        Dt = -torch.sqrt(1 - self.sampler_operator.alphas_cumprod) / \
             torch.sqrt(self.sampler_operator.alphas_cumprod)

        x0_hat = (utils.extract_and_expand(Ct, t, x_t) * x_t +
                  utils.extract_and_expand(Dt, t, x_t) * model_output)
        return utils.clip_denoised(x0_hat)

    def sample_ddim(self, x_t, t, x0_hat, model_output):
        At = torch.sqrt(self.sampler_operator.alphas_cumprod_prev)
        sigma = self.conf.eta * torch.sqrt(
            (1 - self.sampler_operator.alphas_cumprod_prev) /
            (1 - self.sampler_operator.alphas_cumprod)
        ) * torch.sqrt(1 - (self.sampler_operator.alphas_cumprod /
                            self.sampler_operator.alphas_cumprod_prev))

        Bt = torch.sqrt(1 - self.sampler_operator.alphas_cumprod_prev - sigma**2)

        x_t_prev = (utils.extract_and_expand(At, t, x_t) * x0_hat +
                     utils.extract_and_expand(Bt, t, x_t) * model_output)

        noise = torch.randn_like(x_t)
        if (t != 0).any():
            x_t_prev += utils.extract_and_expand(sigma, t, noise) * noise

        return x_t_prev


    def dps(self, x_t, t, model_t, measurement, A_funcs):

        x_t = x_t.clone().detach().requires_grad_(True)

        model_out = self.score_model(x_t, model_t)
        model_out, _ = torch.split(model_out, x_t.shape[1], dim=1)

        x0_hat = self.predict_x0_hat(x_t, t, model_out)
        x_pred = self.sample_ddim(x_t, t, x0_hat, model_out)

        residual = A_funcs.A(x0_hat) - measurement
        loss = (residual ** 2).sum()

        grad = torch.autograd.grad(loss, x_t)[0]
        grad_norm = grad.flatten(1).norm(dim=1).view(-1,1,1,1) + 1e-8
        grad = grad / grad_norm

        zeta = 0.8
        return x_pred - zeta * grad

    def dds(self, x_t, t, model_t, measurement, A_funcs):

        with torch.no_grad():
            model_out = self.score_model(x_t, model_t)
            model_out, _ = torch.split(model_out, x_t.shape[1], dim=1)
            x0_hat = self.predict_x0_hat(x_t, t, model_out) 

            x0_hat_flat = x0_hat[:, 0].reshape(x_t.shape[0], -1) 
            rhs_full = A_funcs.At(measurement[:, :A_funcs.N]) 
            rhs = rhs_full[:, :A_funcs.N]                     
            x_cg = self.CG_normal_eq(A_funcs, rhs, x0_hat_flat)

            x0_hat_proj = x0_hat.clone()
            x0_hat_proj[:, 0] = x_cg.reshape_as(x0_hat[:, 0])

            x_pred = self.sample_ddim(x_t, t, x0_hat_proj, model_out)
            return x_pred

    def CG_normal_eq(self, A_funcs, rhs, x0, max_iter=5, tol=1e-6):

        x = x0.clone()

        Ax_3N = A_funcs.A(pack_flat_to_3ch(A_funcs, x))          
        Ax_phys = Ax_3N[:, :A_funcs.N]                           
        AtAx_full = A_funcs.At(Ax_phys)                          
        AtAx = AtAx_full[:, :A_funcs.N]                          
        r = rhs - AtAx                                           

        p = r.clone()
        rs_old = (r * r).sum(dim=1, keepdim=True)                 

        for _ in range(max_iter):
            # Ap = Aᵀ (A p)
            Ap_3N = A_funcs.A(pack_flat_to_3ch(A_funcs, p))         
            Ap_phys = Ap_3N[:, :A_funcs.N]                          
            AtAp_full = A_funcs.At(Ap_phys)                         
            Ap_vec = AtAp_full[:, :A_funcs.N]                       

            denom = (p * Ap_vec).sum(dim=1, keepdim=True).clamp_min(1e-12)
            alpha = rs_old / denom                                 
            x = x + alpha * p                                      
            r = r - alpha * Ap_vec                                 

            rs_new = (r * r).sum(dim=1, keepdim=True)      
            if torch.sqrt(rs_new).mean() < tol:
                break

            beta = rs_new / rs_old.clamp_min(1e-12)               
            p = r + beta * p                                      
            rs_old = rs_new

        return x

# MAIN
def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    conf = Config().parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading diffusion model...")
    model_config = utils.load_yaml(f"step1_utils/models/{conf.dataset}_model_config.yaml")
    score_model = create_model(**model_config).to(device).eval()

    # load dummy dataset (needed for pipeline structure)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((conf.H, conf.W)),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])
    dataset = get_dataset(conf.dataset, f"step1_utils/data/{conf.dataset}/", transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    print("Loading BH operator...")
    BH, BH_T = load_BH_as_torch_csr(conf.bh_mat, conf.bh_group, device=device)
    A_funcs = BHOperator(BH, BH_T, H=conf.H, W=conf.W, channels=conf.channels, device=device)

    print("Loading By (ultrasound measurement)...")
    mat = loadmat(conf.by_mat)
    By_np = mat["By"].astype(np.float32).reshape(1, -1)
    By = torch.from_numpy(By_np).to(device)
    print("Loaded measurement shape:", By.shape)

    if A_funcs.scale is None:
        p99 = torch.quantile(By.abs().flatten(), 0.99).item() + 1e-8
        A_funcs.scale = 1.0 / p99
        print(f"Auto-scale BH by: {A_funcs.scale:.4f}")

    By = A_funcs.scale * By
    By_3ch = torch.cat([By, By, By], dim=1)

    # Now sampling
    for idx_img, ref_img in enumerate(loader):
        print(f"Starting DPS for image {idx_img + 1}")

        x_t = utils.get_noise_x_t(device)   # (1,3,H,W)
        sampler = DDIM(conf)
        ps_ops = posterior_samplers(conf, sampler, score_model)
        pbar = list(range(conf.desired_timesteps))[::-1]
        time_map = sampler.recreate_alphas().to(device)

        measurement = By_3ch

        for step in tqdm(pbar):
            time = torch.tensor([step], device=device)

            if conf.ps_type == "DPS":
                x_t = ps_ops.dps(x_t, time, time_map[time], measurement, A_funcs)
            elif conf.ps_type == "DDS":
                x_t = ps_ops.dds(x_t, time, time_map[time], measurement, A_funcs)
            else:
                raise ValueError("Use --ps_type DPS")

        # Extract grayscale result
        x_gray = x_t[0,0].detach().cpu().numpy() 

        eps = 1e-12
        im = np.abs(x_gray)
        im = 20 * np.log10(im / (im.max() + eps))


        vrange = [-60, 0]
        im = np.clip(im, vrange[0], vrange[1])
        im_norm = (im - vrange[0]) / (vrange[1] - vrange[0])
        out_dir = os.path.join(conf.out_path, "US_DPS")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"recon_{idx_img+1}.png")

        plt.imsave(out_path, im_norm, cmap='gray')
        print("Saved to", out_path)


        break

    print("FINISHED.")


if __name__ == "__main__":
    main()
