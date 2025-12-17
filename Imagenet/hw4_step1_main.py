import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os
from step1_utils.models.unet import create_model
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import step1_utils.utils as utils
from step1_utils.DDIM_sampler import Sampler as DDIM
import argparse
import torchvision.transforms as transforms
from step1_utils.data.dataloader import get_dataset, get_dataloader
from step1_utils.degradations import GaussianNoise, get_degradation
import numpy as np

class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.conf = None
        
        # hyperparameters for path & dataset
        self.parser.add_argument('--out_path', type=str, default='results/step1_results', help='results file directory')
        self.parser.add_argument('--dataset', type=str, default='CelebA_HQ', help='either choose CelebA_HQ or ImageNet')
        self.parser.add_argument('--sigma_y', type=float, default=0.0, help='measurement noise')
        
        # hyperparameters for sampling
        self.parser.add_argument('--diff_timesteps', type=int, default=1000, help='Original number of steps from Ho et al. (2020) which is 1000 - do not change')
        self.parser.add_argument('--desired_timesteps', type=int, default=1000, help='How many steps do you want?')
        self.parser.add_argument('--eta', type=float, default=1.0, help='Should be between [0.0, 1.0]')
        self.parser.add_argument('--schedule', type=str, default="1000", help="regular/irregular schedule to use (jumps)")
        
        # hyperparameters for algos
        self.parser.add_argument('--ps_type', type=str, default="ILVR", help="choose from projection, DPS, DDNM")
        self.parser.add_argument('--degradation', type=str, default='Inpainting', help='SR or Inpainting')
        
        # hyperparameters for the inpainting mask & SR
        self.parser.add_argument('--mask_type', type=str, default="box", help='box or random')
        self.parser.add_argument('--random_amount', type=float, default=0.8, help='how much do you want to mask out?')
        self.parser.add_argument('--box_indices', type=int, default=[30,30,128,128], help='inpainting box indices - (y,x,height,width)')
        self.parser.add_argument('--scale_factor', type=int, default=4, help='SR scale factor')

    def parse(self, args=None):
        """Parse the configuration"""
        self.conf = self.parser.parse_args(args=args)
        return self.conf

class posterior_samplers():
    def __init__(self, conf, sampler_operator, score_model):
        self.conf = conf
        self.sampler_operator = sampler_operator
        self.score_model = score_model
    
    def predict_x0_hat(self, x_t, t, model_output):
        ############################
        # TODO: Implement the function predicting the clean denoised estimate x_{0|t}
        # Similar to HW3, use utils.extract_and_expand() function when necessary
        ############################
        Ct = 1/torch.sqrt(self.sampler_operator.alphas_cumprod)
        Dt = -torch.sqrt(1 - self.sampler_operator.alphas_cumprod) / torch.sqrt(self.sampler_operator.alphas_cumprod)
        x0_hat = utils.extract_and_expand(Ct, t, x_t) * x_t + utils.extract_and_expand(Dt, t, x_t) * model_output
        return utils.clip_denoised(x0_hat)
    
    def sample_ddim(self, x_t, t, x0_hat, model_output):
        ############################
        # TODO: Implement DDIM sampling
        ############################
        At = torch.sqrt(self.sampler_operator.alphas_cumprod_prev) 
        sigma = self.conf.eta * torch.sqrt((1 - self.sampler_operator.alphas_cumprod_prev) / (1 - self.sampler_operator.alphas_cumprod)) * torch.sqrt(1 - (self.sampler_operator.alphas_cumprod / self.sampler_operator.alphas_cumprod_prev))
        Bt = torch.sqrt(1-self.sampler_operator.alphas_cumprod_prev-sigma**2)
        x_t_prev = utils.extract_and_expand(At, t, x_t) * x0_hat + utils.extract_and_expand(Bt, t, x_t) * model_output
        noise = torch.randn_like(x_t)
        if t != 0:
            x_t_prev += utils.extract_and_expand(sigma, t, noise) * noise
        return x_t_prev
    
    def q_sample(self, data, t):
        ############################
        # TODO: Implement q(xt−1 | x0) = N (xt−1; √¯αt−1 x0, (1 − ¯αt−1)I)
        # Hint-1: Reparametrization Trick
        # Hint-2: You can get \bar{α}_{t−1} from --> self.sampler_operator.alphas_cumprod_prev
        ############################

        device, dtype = data.device, data.dtype
        abar_prev = self.sampler_operator.alphas_cumprod_prev.to(device=device, dtype=dtype)  # (T,)
        mean_coef = torch.sqrt(abar_prev.clamp(min=0.0, max=1.0))
        std_coef  = torch.sqrt((1.0 - abar_prev).clamp(min=0.0, max=1.0))

        mean_c = utils.extract_and_expand(mean_coef, t, data)
        std_c  = utils.extract_and_expand(std_coef,  t, data)

        eps = torch.randn_like(data)
        x_tminus1 = mean_c * data + std_c * eps
        q_xt_x0 = x_tminus1
        return q_xt_x0
    
    def ilvr(self, x_t, t, model_t, measurement, A_funcs):
        ############################
        # TODO: Implement ILVR based on the HW PDF description
        
        # Hint-1: You can get the model output similar to HW3:
        with torch.no_grad():
            model_output = self.score_model(x_t, model_t)
        model_output, _ = torch.split(model_output, x_t.shape[1], dim=1)
        x_t_prime = self.sample_ddim(x_t, t, self.predict_x0_hat(x_t, t, model_output), model_output)
        zeta = 1
        noisy_measurement = self.q_sample(measurement, t)
        residual = A_funcs.A_pinv(noisy_measurement - A_funcs.A(x_t_prime))
        # Hint-2: A, A^T or A^\dagger operations can be performed by:
        # A_funcs.A(), A_funcs.At(), A_funcs.A_pinv()
        ############################
        x_t_prev = x_t_prime + zeta * residual.reshape(x_t.shape)
        return x_t_prev
    
    def mcg(self, x_t, t, model_t, measurement, A_funcs):
        ############################
        # TODO: Implement MCG based on the HW PDF description
        ############################
        with torch.no_grad():
            model_output = self.score_model(x_t, model_t)
        model_output, _ = torch.split(model_output, x_t.shape[1], dim=1)
        x0_hat = self.predict_x0_hat(x_t, t, model_output)
        x_t_prime = self.sample_ddim(x_t, t, x0_hat, model_output)
        zeta = 0.1
        residual = A_funcs.A(x0_hat) - measurement     # shape: (B, n_obs)
        grad_flat = A_funcs.At(residual)               # shape: (B, 3*256*256)
        grad = grad_flat.view_as(x_t)                  # reshape (B,3,256,256)
        x_t_tilda = x_t_prime - zeta * grad
        noisy_measurement = self.q_sample(measurement, t)
        residual = A_funcs.At(noisy_measurement - A_funcs.A(x_t_tilda))
        x_t_prev = x_t_tilda + residual.reshape(x_t.shape)

        return x_t_prev
    
    def ddnm(self, x_t, t, model_t, measurement, A_funcs):
        ############################
        # TODO: Implement DDNM based on the HW PDF description
        ############################
        with torch.no_grad():
            model_output = self.score_model(x_t, model_t)
        model_output, _ = torch.split(model_output, x_t.shape[1], dim=1)
        x0_hat = self.predict_x0_hat(x_t, t, model_output)
        zeta = 1
        x0_prime = x0_hat + zeta * A_funcs.A_pinv(measurement - A_funcs.A(x0_hat)).view_as(x_t)
        x0_prime = utils.clip_denoised(x0_prime)
        x_t_prev = self.sample_ddim(x_t, t, x0_prime, model_output)
        return x_t_prev
    
    def dps(self, x_t, t, model_t, measurement, A_funcs):
        ############################
        # TODO: Implement DPS based on the HW PDF description
        ############################
        with torch.no_grad():
          model_output = self.score_model(x_t, model_t)
          model_output, _ = torch.split(model_output, x_t.shape[1], dim=1)  
          x0_hat = self.predict_x0_hat(x_t, t, model_output)
          x_t_prev_prime = self.sample_ddim(x_t, t, x0_hat, model_output)
          residual = A_funcs.A(x0_hat) - measurement     # shape: (B, n_obs)
          grad_flat = A_funcs.At(residual)               # shape: (B, 3*256*256)
          grad = grad_flat.view_as(x_t)                  # reshape (B,3,256,256)
          denom = grad.flatten(1).norm(p=2, dim=1).view(-1,1,1,1) + 1e-8
          grad = grad / denom
          zeta = 0.9 
          x_t_prev = x_t_prev_prime - zeta * grad
          return x_t_prev

        
def main():
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    conf = Config().parse()
    
    print('*' * 60 + f'\nSTARTED DDIM Sampling with eta = \"%.1f\" \n' %conf.eta)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    loss_fn = lpips.LPIPS(net='alex').to(device)
    
    # Create and config model
    model_config = utils.load_yaml("step1_utils/models/" + conf.dataset + "_model_config.yaml")
    score_model = create_model(**model_config).to(device).eval()
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256)), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(conf.dataset, f"step1_utils/data/{conf.dataset}/", transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)
    noiser = GaussianNoise(conf.sigma_y*2.0)
    A_funcs = get_degradation(conf, device)
    
    # Sampling
    for i, ref_img in enumerate(loader):
        print(f'\nSampling for Image {i+1} has started!')
        sampler_operator = DDIM(conf)
        ref_img = ref_img.to(device)
        measurement = noiser(A_funcs.A(ref_img))
        # x_t = utils.get_noise_x_t(device).requires_grad_()
        x_t = utils.get_noise_x_t(device)
        pbar = (list(range(conf.desired_timesteps))[::-1])
        time_map = sampler_operator.recreate_alphas().to(device)
        ps_ops = posterior_samplers(conf, sampler_operator, score_model)
        
        for idx in tqdm(pbar):
            time = torch.tensor([idx] * x_t.shape[0], device=device)
            if conf.ps_type == "ILVR":
                x_t_prev = ps_ops.ilvr(x_t, time, time_map[time], measurement, A_funcs)    
            elif conf.ps_type == "MCG":
                x_t_prev = ps_ops.mcg(x_t, time, time_map[time], measurement, A_funcs)    
            elif conf.ps_type == "DDNM":
                x_t_prev = ps_ops.ddnm(x_t, time, time_map[time], measurement, A_funcs)         
            elif conf.ps_type == "DPS":
                x_t_prev = ps_ops.dps(x_t, time, time_map[time], measurement, A_funcs)
            elif conf.ps_type == "testing":
                with torch.no_grad():
                    model_output = score_model(x_t, time_map[time])
                model_output, _ = torch.split(model_output, x_t.shape[1], dim=1)
                x_t_prev = ps_ops.sample_ddim(x_t, time, ps_ops.predict_x0_hat(x_t, time, model_output), model_output)
            x_t = x_t_prev
        image_filename = f"recon_{i+1}.png"
        
        image_path = os.path.join(conf.out_path, conf.dataset, conf.ps_type, image_filename)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        plt.imsave(image_path, np.concatenate([utils.clear_color(ref_img), utils.clear_color(A_funcs.A_pinv(measurement).reshape(1,3,256,256)), utils.clear_color(x_t)], axis=1))
        
        ref = to_img(ref_img)              # (1,3,256,256)
        rec = to_img(x_t)                  # (1,3,256,256)

        
        ref_np = ref.squeeze().permute(1,2,0).cpu().numpy()
        rec_np = rec.squeeze().permute(1,2,0).cpu().numpy()

        # --- PSNR ---
        psnr = peak_signal_noise_ratio(ref_np, rec_np, data_range=1.0)

        # --- SSIM ---
        ssim = structural_similarity(ref_np, rec_np, data_range=1.0, channel_axis=-1)

        
        lp = loss_fn.forward(ref_img, x_t).item()
        print(f'Image {i+1} -- PSNR: {psnr:.4f} dB, SSIM: {ssim:.4f}, LPIPS: {lp:.4f}')


    print('\nFINISHED Sampling!\n' + '*' * 60)


def to_img(x):
    
    x = (x.clamp(-1, 1) + 1) / 2     # -> [0,1]
    return x


if __name__ == '__main__':
    main()