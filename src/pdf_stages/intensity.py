import argparse
import ast
import os
import sys
import cv2
from typing import Tuple

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch_ema import ExponentialMovingAverage
from torch.utils.data import Dataset
from tqdm import tqdm
import monai
import albumentations as A

from data_utils.prepare_dataset import prepare_dataset
from nn.scheduler import LinearWarmupCosineAnnealingLR
from utils.attribute_hashmap import AttributeHashmap
from utils.log_util import log
from utils.metrics import psnr, ssim, dice_coeff, hausdorff
from utils.parse import parse_settings
from utils.seed import seed_everything

from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates

from nn.pdf import PDF_intensity

import kornia as K


from skimage.registration import optical_flow_tvl1

def cast_to_0to1(*np_arrays: np.array) -> Tuple[np.array]:
    '''
    Cast image to normal dynamic range between 0 and 1.
    '''
    return [np.clip((_arr + 1) / 2, 0, 1) for _arr in np_arrays]


def add_random_noise(img: torch.Tensor, max_intensity: float = 0.1) -> torch.Tensor:
    intensity = max_intensity * torch.rand(1).to(img.device)
    noise = intensity * torch.randn_like(img)
    return img + noise

def neg_cos_sim(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    '''
    Negative cosine similarity. For SimSiam.
    '''
    z = z.detach() # stop gradient
    p = torch.nn.functional.normalize(p, p=2, dim=1) # l2-normalize
    z = torch.nn.functional.normalize(z, p=2, dim=1) # l2-normalize
    return -(p * z).sum(dim=1).mean()


def visualize_rfm_sample(x,
                          eps,
                          x_s,
                          v_gt,
                          v_pred,
                          mask,
                          save_path,
                          v_clip=5.0):
    """
    x, eps, x_s: [C,H,W]
    v_gt, v_pred: [C,H,W]
    mask: [1,H,W]
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def to_img(t):
        t = t.detach().cpu()
        if t.shape[0] == 1:
            t = t[0]
        else:
            t = t.permute(1, 2, 0)
        return t.numpy()

    # velocity magnitude
    vgt_mag   = torch.norm(v_gt, dim=0).clamp(0, v_clip)
    vpred_mag = torch.norm(v_pred, dim=0).clamp(0, v_clip)

    x_hat = eps + v_pred

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # -------- row 1 --------
    axes[0, 0].imshow(to_img(x), cmap="gray")
    axes[0, 0].set_title("image (GT)")
    axes[0, 1].imshow(vgt_mag.detach().cpu().numpy(), cmap="jet")
    axes[0, 1].set_title("v_gt |mag|")
    axes[0, 2].imshow(to_img(mask), cmap="gray")
    axes[0, 2].set_title("mask")

    # -------- row 2 --------
    axes[1, 0].imshow(to_img(x_s), cmap="gray")
    axes[1, 0].set_title("x_s")
    axes[1, 1].imshow(vpred_mag.detach().cpu().numpy(), cmap="jet")
    axes[1, 1].set_title("v_pred |mag|")
    axes[1, 2].imshow(to_img(x_hat), cmap="gray")
    axes[1, 2].set_title("x_hat")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def visualize_rfm_sample_logit(x,
                          eps,
                          x_s,
                          v_gt,
                          v_pred,
                          mask,
                          save_path,
                          v_clip=5.0):
    """
    x, eps, x_s: [C,H,W]
    v_gt, v_pred: [C,H,W]
    mask: [1,H,W]
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def to_img(t):
        t = t.detach().cpu()
        if t.shape[0] == 1:
            t = t[0]
        else:
            t = t.permute(1, 2, 0)
        return t.numpy()

    # velocity magnitude
    vgt_mag   = torch.norm(v_gt, dim=0).clamp(0, v_clip)
    vpred_mag = torch.norm(v_pred, dim=0).clamp(0, v_clip)

    z_hat = eps + v_pred
    x_hat = torch.sigmoid(z_hat)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # -------- row 1 --------
    axes[0, 0].imshow(to_img(x), cmap="gray")
    axes[0, 0].set_title("image (GT)")
    axes[0, 1].imshow(vgt_mag.detach().cpu().numpy(), cmap="jet")
    axes[0, 1].set_title("v_gt |mag|")
    axes[0, 2].imshow(to_img(mask), cmap="gray")
    axes[0, 2].set_title("mask")

    # -------- row 2 --------
    axes[1, 0].imshow(to_img(x_s), cmap="gray")
    axes[1, 0].set_title("x_s")
    axes[1, 1].imshow(vpred_mag.detach().cpu().numpy(), cmap="jet")
    axes[1, 1].set_title("v_pred |mag|")
    axes[1, 2].imshow(to_img(x_hat), cmap="gray")
    axes[1, 2].set_title("x_hat")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def train(config: AttributeHashmap):
    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')

    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        ],
        additional_targets={
            'image_other': 'image',
        }
    )
    if config.coeff_invariance > 0:
        aug_transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=60, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            ],
        )
        transforms_list = [train_transform, None, None, aug_transform]
    else:
        transforms_list = [train_transform, None, None]

    train_set, val_set, test_set, num_image_channel, max_t = \
        prepare_dataset(config=config, transforms_list=transforms_list)

    # use test set to val
    val_set = test_set

    log('Using device: %s' % device, to_console=True)

    # Build the model
    kwargs = {}
    
    model_registry = {'PDF_intensity': PDF_intensity}
    if config.model not in model_registry:
        raise ValueError('`config.model`: %s not supported.' % config.model)
    model = model_registry[config.model](
        device=device,
        num_filters=config.num_filters,
        depth=config.depth,
        ode_location=config.ode_location,
        in_channels=num_image_channel,
        out_channels=num_image_channel,
        contrastive=config.coeff_contrastive + config.coeff_invariance > 0,
        **kwargs,
    )

    ema = ExponentialMovingAverage(model.parameters(), decay=0.9)

    model.to(device)
    model.init_params()
    ema.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                              warmup_epochs=config.max_epochs//10,
                                              max_epochs=config.max_epochs)

    mse_loss = torch.nn.MSELoss()
    best_val_psnr, best_val_dice = 0, 0
    backprop_freq = config.batch_size

    os.makedirs(config.save_folder + 'train/', exist_ok=True)
    os.makedirs(config.save_folder + 'val/', exist_ok=True)

    # Train the UNet part first and only start training
    # the ODE part once the reconstruction is good enough.
    recon_psnr_thr, recon_good_enough = 25, False

    # Only relevant to ODE
    config.t_multiplier = config.ode_max_t / max_t

    for epoch_idx in tqdm(range(config.max_epochs)):
        model, ema, optimizer, scheduler = train_epoch_RFM(
            config=config,
            device=device,
            train_set=train_set,
            model=model,
            epoch_idx=epoch_idx,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            backprop_freq=backprop_freq
        )

        with ema.average_parameters():
            model.eval()

            val_pred_psnr, val_seg_dice_xT = \
                val_epoch_RFM(
                    config=config,
                    device=device,
                    val_set=val_set,
                    model=model,
                    epoch_idx=epoch_idx
                )

            # Placeholder recon PSNR for the best-model bookkeeping below.
            val_recon_psnr = 0.0

        if val_recon_psnr > recon_psnr_thr:
            recon_good_enough = True

        if val_pred_psnr > best_val_psnr:
            best_val_psnr = val_pred_psnr
            model.save_weights(config.model_save_path.replace('.pty', '_best_pred_psnr.pty'))
            log('%s: Model weights successfully saved for best pred PSNR.' % config.model,
                filepath=config.log_dir,
                to_console=False)

        if val_seg_dice_xT > best_val_dice:
            best_val_dice = val_seg_dice_xT
            model.save_weights(config.model_save_path.replace('.pty', '_best_seg_dice.pty'))
            log('%s: Model weights successfully saved for best dice xT.' % config.model,
                filepath=config.log_dir,
                to_console=False)
        # -----------------------------
        # Save last checkpoint (always)
        # -----------------------------
        model.save_weights(
            config.model_save_path.replace('.pty', '_last.pty')
        )
        log('%s: Model weights successfully saved for last checkpoint.' % config.model,
            filepath=config.log_dir,
            to_console=False)
    return


def mask_to_logit(mask01: torch.Tensor, eps: float = 1e-4):
    # mask01: [B,1,H,W] in {0,1}
    p = torch.clamp(mask01, eps, 1.0 - eps)
    return torch.log(p / (1.0 - p))

def logit_to_prob(logit: torch.Tensor):
    return torch.sigmoid(logit)


def train_epoch_RFM(config,
                    device,
                    train_set,
                    model,
                    epoch_idx,
                    ema,
                    optimizer,
                    scheduler,
                    backprop_freq):

    model.train()
    optimizer.zero_grad()

    mse_loss = torch.nn.MSELoss()
    train_loss = 0.0

    assert len(train_set) == len(train_set.dataset)
    num_train_samples = min(config.max_training_samples, len(train_set))
    plot_freq = max(num_train_samples // config.n_plot_per_epoch, 1)

    for iter_idx, batch in enumerate(tqdm(train_set)):
        if iter_idx >= config.max_training_samples:
            break

        shall_plot = iter_idx % plot_freq == 0

        images, masks, timestamps, vx, vy = batch

        # --------------------------------------------------
        # move to device
        # --------------------------------------------------
        x_list, _ = convert_variables(images, timestamps, device)
        m_list, _ = convert_variables(masks, timestamps, device)

        x0, x1 = x_list          # [B, C, H, W]
        m0, m1 = m_list          # [B, 1, H, W]

        # --------------------------------------------------
        # treat them as independent samples
        # --------------------------------------------------
        x = torch.cat([x0, x1], dim=0)        # [2B, C, H, W]
        m = torch.cat([m0, m1], dim=0)        # [2B, 1, H, W]

        # --------------------------------------------------
        # mask as condition (logit -> prob)
        # --------------------------------------------------
        m = logit_to_prob(mask_to_logit(m))
        
        # -----------------------------
        # IMAGE -> LOGIT SPACE
        # -----------------------------
        eps_img = 1e-6
        x = (x + 1.0) * 0.5
        x = torch.clamp(x, eps_img, 1.0 - eps_img)
        z = torch.log(x / (1.0 - x))      # logit(x)

        
        # -----------------------------
        # noise & interpolation (logit space)
        # -----------------------------
        eps_z = torch.randn_like(z)
        s = torch.rand((z.shape[0], 1, 1, 1), device=device)

        z_s  = (1.0 - s) * eps_z + s * z
        v_gt = z - eps_z


        # --------------------------------------------------
        # predict velocity
        # --------------------------------------------------
        v_pred = model(z_s, m, s.view(-1, 1))

        loss = mse_loss(v_pred, v_gt)
        loss.backward()

        if shall_plot:
            # pick first sample in batch
            idx = 0

            save_path = (
                f"{config.save_folder}/train/"
                f"epoch{epoch_idx:03d}_iter{iter_idx:05d}.png"
            )

            visualize_rfm_sample(
                x=x[idx],
                eps=eps_z[idx],
                x_s=z_s[idx],
                v_gt=v_gt[idx],
                v_pred=v_pred[idx],
                mask=m[idx],
                save_path=save_path
            )

        train_loss += loss.item()

        if (iter_idx + 1) % backprop_freq == 0:
            optimizer.step()
            optimizer.zero_grad()
            ema.update(model.parameters())

    scheduler.step()

    train_loss /= max(1, iter_idx + 1)

    log(
        f"[Train][Epoch {epoch_idx:03d}] "
        f"Loss: {train_loss:.6f}",
        filepath=config.log_dir,
        to_console=True
    )

    return model, ema, optimizer, scheduler

@torch.no_grad()
def val_epoch_RFM(config,
                  device,
                  val_set,
                  model,
                  epoch_idx):

    model.eval()

    mse_list = []

    assert len(val_set) == len(val_set.dataset)
    num_val_samples = min(config.max_validation_samples, len(val_set))
    plot_freq = num_val_samples // config.n_plot_per_epoch

    for iter_idx, batch in enumerate(tqdm(val_set)):
        shall_plot = iter_idx % plot_freq == 0
        images, masks, timestamps, vx, vy = batch

        x_list, _ = convert_variables(images, timestamps, device)
        m_list, _ = convert_variables(masks, timestamps, device)

        x0, x1 = x_list
        m0, m1 = m_list

        x = torch.cat([x0, x1], dim=0)
        m = torch.cat([m0, m1], dim=0)

        m = logit_to_prob(mask_to_logit(m))

        x = (x + 1.0) * 0.5
        z = torch.log(x / (1.0 - x))

        eps_z = torch.randn_like(z)
        s = torch.rand((z.shape[0], 1, 1, 1), device=device)

        v_pred = model(eps_z, m, s.view(-1, 1))
        z_pred = eps_z + v_pred
        x_pred = torch.sigmoid(z_pred)




        mse = ((x_pred - x) ** 2).mean(dim=[1, 2, 3])
        mse_list.append(mse.mean())

        if shall_plot:
            idx = 0

            save_path = (
                f"{config.save_folder}/val/"
                f"epoch{epoch_idx:03d}_iter{iter_idx:05d}.png"
            )

            visualize_rfm_sample(
                x=x[idx],
                eps=eps_z[idx],
                x_s=eps_z[idx],          # During validation, x_s = eps at the s=1 start point.
                v_gt=z[idx] - eps_z[idx],
                v_pred=v_pred[idx],
                mask=m[idx],
                save_path=save_path
            )

    mean_mse = torch.stack(mse_list).mean()
    psnr = 10.0 * torch.log10(1.0 / mean_mse)

    log(
        f"[Val][Epoch {epoch_idx:03d}] PSNR: {psnr.item():.4f}",
        filepath=config.log_dir,
        to_console=True
    )

    return psnr.item(), 0.0

@torch.no_grad()
def rfm_sample_euler(model, eps, m_prob, steps: int = 50):
    """
    Solve dx/ds = v_theta(x, m, s),  s: 0 -> 1
    eps:   [B, C, H, W]  initial noise (x(s=0))
    m_prob:[B, 1, H, W]
    return x: [B, C, H, W]
    """
    x = eps
    B = x.shape[0]
    s_grid = torch.linspace(0.0, 1.0, steps + 1, device=x.device)

    for k in range(steps):
        s_k = s_grid[k]
        ds = s_grid[k + 1] - s_k
        s_in = torch.full((B, 1), float(s_k), device=x.device)

        v = model(x, m_prob, s_in)
        x = x + ds * v

    return x

#     """
#     """






def rfm_sample_heun(model, z0, m, steps=50):
    z = z0
    B = z.shape[0]
    s_grid = torch.linspace(0.0, 1.0, steps + 1, device=z.device)

    for k in range(steps):
        s_k = s_grid[k]
        s_k1 = s_grid[k + 1]
        ds = s_k1 - s_k

        s_in  = torch.full((B, 1), float(s_k), device=z.device)
        s_in1 = torch.full((B, 1), float(s_k1), device=z.device)

        v0 = model(z, m, s_in)
        z_euler = z + ds * v0

        v1 = model(z_euler, m, s_in1)
        z = z + 0.5 * ds * (v0 + v1)

    return z

@torch.no_grad()
def test(config: AttributeHashmap):

    device = torch.device(
        f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu'
    )

    # ------------------------------------------------------------
    # ------------------------------------------------------------
    train_set, val_set, test_set, num_image_channel, max_t = \
        prepare_dataset(config=config)

    assert len(test_set) == len(test_set.dataset)
    num_test_samples = min(config.max_testing_samples, len(test_set))

    # ------------------------------------------------------------
    # ------------------------------------------------------------
    model_registry = {'PDF_intensity': PDF_intensity}
    if config.model not in model_registry:
        raise ValueError(f'`config.model`: {config.model} not supported.')
    model = model_registry[config.model](
        device=device,
        in_channels=num_image_channel,
        num_filters=config.num_filters,
        depth=config.depth,
    )

    model.to(device)
    model.eval()

    # ------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------
    model.load_weights(
        config.model_save_path.replace('.pty', '_last.pty'),
        device=device
    )
    log(f'{config.model}: Model weights successfully loaded.',
        to_console=True)

    # ------------------------------------------------------------
    # Segmentor (optional)
    # ------------------------------------------------------------
    if os.path.isfile(config.segmentor_ckpt):
        segmentor = torch.nn.Sequential(
            monai.networks.nets.DynUNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                kernel_size=[5, 5, 5, 5],
                filters=[16, 32, 64, 128],
                strides=[1, 1, 1, 1],
                upsample_kernel_size=[1, 1, 1, 1]
            ),
            torch.nn.Sigmoid()
        ).to(device)
        segmentor.load_state_dict(
            torch.load(config.segmentor_ckpt, map_location=device)
        )
        segmentor.eval()
    else:
        segmentor = torch.nn.Identity()

    # ------------------------------------------------------------
    # Metrics containers
    # ------------------------------------------------------------
    psnr_list = []
    ssim_list = []
    dice_list = []
    hd_list = []

    # ------------------------------------------------------------
    # Test loop
    # ------------------------------------------------------------
    for iter_idx, batch in enumerate(tqdm(test_set)):
        if iter_idx >= num_test_samples:
            break

        # -----------------------------
        # unpack batch
        # -----------------------------
        images, masks, timestamps, vx, vy = batch
        x_list, _ = convert_variables(images, timestamps, device)
        m_list, _ = convert_variables(masks, timestamps, device)

        # treat x0 and x1 as independent samples
        x0, x1 = x_list
        m0, m1 = m_list

        x = torch.cat([x0, x1], dim=0)    # [2B, C, H, W]
        m = torch.cat([m0, m1], dim=0)    # [2B, 1, H, W]

        m = logit_to_prob(mask_to_logit(m))

        # -----------------------------
        # IMAGE -> LOGIT SPACE
        # -----------------------------
        eps_img = 1e-6
        x = (x + 1.0) * 0.5
        x = torch.clamp(x, eps_img, 1.0 - eps_img)
        z = torch.log(x / (1.0 - x))      # logit(x)

        # -----------------------------
        # -----------------------------
        eps_z = torch.randn_like(z)

        steps = getattr(config, "rfm_sample_steps", 50)
        sampler = getattr(config, "rfm_sampler", "heun")  # "euler" or "heun"

        if sampler == "euler":
            z_pred = rfm_sample_euler(model, eps_z, m, steps=steps)
        else:
            z_pred = rfm_sample_heun(model, eps_z, m, steps=steps)

        x_pred = torch.sigmoid(z_pred)
        x_pred = x_pred * 2.0 - 1.0
        # -----------------------------
        # metrics: image
        # -----------------------------
        mse = ((x_pred - x) ** 2).mean(dim=[1, 2, 3])
        mse = mse.clamp(min=1e-10)

        psnr = 10.0 * torch.log10(1.0 / mse)
        psnr_list.append(psnr.mean())
        from utils.metrics import psnr, ssim
        ssim_batch = []

        for i in range(x.shape[0]):
            x_np = x[i].detach().cpu().numpy()
            x_pred_np = x_pred[i].detach().cpu().numpy()

            # Drop the channel dimension for single-channel images.
            if x_np.shape[0] == 1:
                x_np = x_np[0]
                x_pred_np = x_pred_np[0]

            ssim_i = ssim(x_np, x_pred_np)
            ssim_batch.append(ssim_i)

        ssim_val = sum(ssim_batch) / len(ssim_batch)
        ssim_list.append(ssim_val)


        # -----------------------------
        # -----------------------------
        if 1:
            # pick a mid step to visualize x_s and v_pred
            mid_k = steps // 2
            s_mid = mid_k / steps

            # reconstruct x_mid by re-running up to mid (cheap enough for vis)
            x_mid = eps_z
            B = x_mid.shape[0]
            s_grid = torch.linspace(0.0, 1.0, steps + 1, device=device)
            for k in range(mid_k):
                s_k = s_grid[k]
                ds = s_grid[k + 1] - s_k
                s_in = torch.full((B, 1), float(s_k), device=device)
                v = model(x_mid, m, s_in)
                x_mid = x_mid + ds * v

            s_in_mid = torch.full((B, 1), float(s_mid), device=device)
            v_mid = model(x_mid, m, s_in_mid)

            save_path = (
                f"{config.save_folder}/test/"
                f"iter{iter_idx:05d}.png"
            )

            visualize_rfm_sample(
                x=x[0],
                eps=eps_z[0],
                x_s=x_mid[0],              # x_s at s_mid
                v_gt=z[0] - eps_z[0],         # Ground-truth velocity for the direct noise-to-image path.
                v_pred=v_mid[0],            # predicted v at s_mid
                mask=m[0],
                save_path=save_path
            )

    # ------------------------------------------------------------
    # Aggregate statistics
    # ------------------------------------------------------------
    psnr = torch.stack(psnr_list).mean().item()
    ssim = torch.stack(ssim_list).mean().item()

    log(f"[Test] PSNR: {psnr:.3f}, SSIM: {ssim:.3f}",
        filepath=config.log_dir,
        to_console=True)

    if len(dice_list) > 0:
        dice = torch.stack(dice_list).mean().item()
        hd = torch.stack(hd_list).mean().item()

        log(f"[Test] Dice: {dice:.3f}, HD: {hd:.3f}",
            filepath=config.log_dir,
            to_console=True)

    return


@torch.no_grad()
def rfm_edit_heun(model, x0, m_edit, steps: int = 50):
    """
    Masked flow editing:
        dx/ds = m_edit * v_theta(x, m_edit, s)

    x0     : [B, C, H, W]  initial image
    m_edit : [B, 1, H, W]  edit mask (prob in [0,1])
    """
    x = x0
    B = x.shape[0]
    s_grid = torch.linspace(0.0, 1.0, steps + 1, device=x.device)

    for k in range(steps):
        s_k = s_grid[k]
        s_k1 = s_grid[k + 1]
        ds = s_k1 - s_k

        s_in  = torch.full((B, 1), float(s_k), device=x.device)
        s_in1 = torch.full((B, 1), float(s_k1), device=x.device)

        v0 = model(x, m_edit, s_in)
        x_euler = x + ds * (m_edit * v0)

        v1 = model(x_euler, m_edit, s_in1)

        x = x + 0.5 * ds * (m_edit * (v0 + v1))

    return x

@torch.no_grad()
def rfm_edit_residual_heun(model, x0, m_edit, steps=50):
    delta = torch.zeros_like(x0)
    B = x0.shape[0]
    s_grid = torch.linspace(0, 1, steps + 1, device=x0.device)

    for k in range(steps):
        s_k, s_k1 = s_grid[k], s_grid[k+1]
        ds = s_k1 - s_k

        s_in  = torch.full((B,1), float(s_k), device=x0.device)
        s_in1 = torch.full((B,1), float(s_k1), device=x0.device)

        x_cur = x0 + m_edit * delta
        v0 = model(x_cur, m_edit, s_in)
        delta_euler = delta + ds * v0

        x_euler = x0 + m_edit * delta_euler
        v1 = model(x_euler, m_edit, s_in1)

        delta = delta + 0.5 * ds * (v0 + v1)

    return x0 + m_edit * delta


import torch.nn.functional as F
def gaussian_blur(x, sigma=1.0, kernel_size=5):
    """
    x: [B, C, H, W]
    """
    # build 1D kernel
    device = x.device
    coords = torch.arange(kernel_size, device=device) - kernel_size // 2
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()

    # separable conv
    kernel_x = kernel.view(1, 1, 1, -1)
    kernel_y = kernel.view(1, 1, -1, 1)

    padding = kernel_size // 2
    x = F.conv2d(x, kernel_x.expand(x.shape[1], 1, 1, -1),
                 padding=(0, padding), groups=x.shape[1])
    x = F.conv2d(x, kernel_y.expand(x.shape[1], 1, -1, 1),
                 padding=(padding, 0), groups=x.shape[1])
    return x


def split_low_high(x, sigma=1.0, kernel_size=5):
    x_lp = gaussian_blur(x, sigma=sigma, kernel_size=kernel_size)
    x_hp = x - x_lp
    return x_lp, x_hp

@torch.no_grad()
def rfm_edit_hf_heun(
    model,
    x0,
    m_edit,
    steps=50,
    sigma=1.2,
    kernel_size=5,
):
    """
    High-frequency preserving RFM edit.

    x0     : [B, C, H, W]
    m_edit : [B, 1, H, W]  (prob in [0,1])
    """
    # split x0
    x0_lp, x0_hp = split_low_high(x0, sigma=sigma, kernel_size=kernel_size)

    # only evolve low-frequency part
    x_lp = x0_lp.clone()
    B = x0.shape[0]

    s_grid = torch.linspace(0.0, 1.0, steps + 1, device=x0.device)

    for k in range(steps):
        s_k = s_grid[k]
        s_k1 = s_grid[k + 1]
        ds = s_k1 - s_k

        s_in  = torch.full((B, 1), float(s_k), device=x0.device)
        s_in1 = torch.full((B, 1), float(s_k1), device=x0.device)

        x_cur = x_lp + x0_hp

        v0 = model(x_cur, m_edit, s_in)
        x_lp_euler = x_lp + 0.1 * ds * (m_edit * v0)
        

        x_euler = x_lp_euler + x0_hp
        v1 = model(x_euler, m_edit, s_in1)

        x_lp = x_lp + 0.1 * 0.5 * ds * (m_edit * (v0 + v1))

    # final image: edited low freq + original high freq
    x_pred = x_lp + x0_hp
    return x_pred

@torch.no_grad()
def edit(config: AttributeHashmap):

    device = torch.device(
        f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu'
    )

    # ------------------------------------------------------------
    # ------------------------------------------------------------
    train_set, val_set, test_set, num_image_channel, max_t = \
        prepare_dataset(config=config)

    assert len(test_set) == len(test_set.dataset)
    num_test_samples = min(config.max_testing_samples, len(test_set))


    if os.path.isfile(config.segmentor_ckpt):
        segmentor = torch.nn.Sequential(
            monai.networks.nets.DynUNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                kernel_size=[5, 5, 5, 5],
                filters=[16, 32, 64, 128],
                strides=[1, 1, 1, 1],
                upsample_kernel_size=[1, 1, 1, 1]
            ),
            torch.nn.Sigmoid()
        ).to(device)
        segmentor.load_state_dict(
            torch.load(config.segmentor_ckpt, map_location=device)
        )
        segmentor.eval()
    else:
        segmentor = torch.nn.Identity()

    # ------------------------------------------------------------
    # ------------------------------------------------------------
    model_registry = {'PDF_intensity': PDF_intensity}
    if config.model not in model_registry:
        raise ValueError(f'`config.model`: {config.model} not supported.')
    model = model_registry[config.model](
        device=device,
        in_channels=num_image_channel,
        num_filters=config.num_filters,
        depth=config.depth,
    )

    model.to(device)
    model.eval()

    # ------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------
    model.load_weights(
        config.model_save_path.replace('.pty', '_last.pty'),
        device=device
    )
    log(f'{config.model}: Model weights successfully loaded.',
        to_console=True)
    
    # ------------------------------------------------------------
    # Metrics containers
    # ------------------------------------------------------------
    psnr_x0_list   = []
    ssim_x0_list   = []
    dice_x0_list   = []
    hd_x0_list     = []

    psnr_pred_list = []
    ssim_pred_list = []
    dice_pred_list = []
    hd_pred_list   = []


    # ------------------------------------------------------------
    # ------------------------------------------------------------
    for iter_idx, batch in enumerate(tqdm(test_set)):
        if iter_idx >= num_test_samples:
            break

        images, masks, timestamps, vx, vy = batch
        x_list, _ = convert_variables(images, timestamps, device)
        m_list, _ = convert_variables(masks, timestamps, device)

        # -----------------------------------------
        # -----------------------------------------
        x0, x1 = x_list        # [1,C,H,W]
        m0, m1 = m_list        # [1,1,H,W]

        m1_prob = logit_to_prob(mask_to_logit(m1))

        # -----------------------------------------
        # flow edit: x0 -> x1_pred (only m1 region)
        # -----------------------------------------
        steps = getattr(config, "rfm_edit_steps", 1)
        x1_pred = rfm_edit_heun(model, x0, m1_prob, steps=steps)

        with torch.no_grad():
            x0_seg = (segmentor(x0) > 0.5).float()
            x1_seg = (segmentor(x1) > 0.5).float()
            x1_pred_seg = (segmentor(x1_pred) > 0.5).float()

        x0_np, x1_np, x1_pred_np, \
        x0_seg_np, x1_seg_np, x1_pred_seg_np = numpy_variables(
            x0, x1, x1_pred,
            x0_seg, x1_seg, x1_pred_seg
        )

        # ---- image metrics ----
        psnr_x0 = psnr(x1_np, x0_np)
        ssim_x0 = ssim(x1_np, x0_np)

        psnr_pred = psnr(x1_np, x1_pred_np)
        ssim_pred = ssim(x1_np, x1_pred_np)

        # ---- segmentation metrics ----
        dice_x0 = dice_coeff(x0_seg_np, x1_seg_np)
        hd_x0 = hausdorff(x0_seg_np, x1_seg_np)

        dice_pred = dice_coeff(x1_pred_seg_np, x1_seg_np)
        hd_pred = hausdorff(x1_pred_seg_np, x1_seg_np)

        psnr_x0_list.append(psnr_x0)
        ssim_x0_list.append(ssim_x0)
        dice_x0_list.append(dice_x0)
        hd_x0_list.append(hd_x0)

        psnr_pred_list.append(psnr_pred)
        ssim_pred_list.append(ssim_pred)
        dice_pred_list.append(dice_pred)
        hd_pred_list.append(hd_pred)

        # -----------------------------------------
        # -----------------------------------------
        save_path = (
            f"{config.save_folder}/edit_step_1/"
            f"iter{iter_idx:05d}.png"
        )

        # pick mid-s velocity for visualization
        s_mid = 0.5
        s_in = torch.full((1, 1), s_mid, device=device)
        v_mid = model(x0, m1_prob, s_in)




        plot_flow_edit_2x3_metrics(
            x0=x0[0], m0=m0[0], v=v_mid[0],
            x1=x1[0], m1=m1[0], x1_pred=x1_pred[0],
            x0_seg=x0_seg[0], x1_seg=x1_seg[0], x1_pred_seg=x1_pred_seg[0],
            psnr_x0=psnr_x0, ssim_x0=ssim_x0,
            psnr_pred=psnr_pred, ssim_pred=ssim_pred,
            dice_x0=dice_x0, hd_x0=hd_x0,
            dice_pred=dice_pred, hd_pred=hd_pred,
            save_path=save_path
        )
    

    def mean_std(x):
        x = np.array(x, dtype=np.float32)
        return x.mean(), x.std()

    psnr_x0_mean, psnr_x0_std = mean_std(psnr_x0_list)
    ssim_x0_mean, ssim_x0_std = mean_std(ssim_x0_list)
    dice_x0_mean, dice_x0_std = mean_std(dice_x0_list)
    hd_x0_mean,   hd_x0_std   = mean_std(hd_x0_list)

    psnr_pred_mean, psnr_pred_std = mean_std(psnr_pred_list)
    ssim_pred_mean, ssim_pred_std = mean_std(ssim_pred_list)
    dice_pred_mean, dice_pred_std = mean_std(dice_pred_list)
    hd_pred_mean,   hd_pred_std   = mean_std(hd_pred_list)

    log(
        f"[EDIT SUMMARY]\n"
        f"x0  → x1 reference:\n"
        f"  PSNR = {psnr_x0_mean:.3f} ± {psnr_x0_std:.3f}\n"
        f"  SSIM = {ssim_x0_mean:.4f} ± {ssim_x0_std:.4f}\n"
        f"  Dice = {dice_x0_mean:.4f} ± {dice_x0_std:.4f}\n"
        f"  HD   = {hd_x0_mean:.3f} ± {hd_x0_std:.3f}\n\n"
        f"x0 edit → x1 (pred):\n"
        f"  PSNR = {psnr_pred_mean:.3f} ± {psnr_pred_std:.3f}\n"
        f"  SSIM = {ssim_pred_mean:.4f} ± {ssim_pred_std:.4f}\n"
        f"  Dice = {dice_pred_mean:.4f} ± {dice_pred_std:.4f}\n"
        f"  HD   = {hd_pred_mean:.3f} ± {hd_pred_std:.3f}",
        to_console=True,
        filepath=config.log_dir
    )

    return


def plot_flow_edit_2x3_metrics(
    x0, m0, v,
    x1, m1,
    x1_pred,
    x0_seg, x1_seg, x1_pred_seg,   # <<< add x0_seg
    psnr_x0, ssim_x0,
    psnr_pred, ssim_pred,
    dice_x0, hd_x0,                 # <<< add x0-to-x1 Dice/HD
    dice_pred, hd_pred,
    save_path
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # ---------------- utils ----------------
    def to_img(t):
        if torch.is_tensor(t):
            t = t.detach().cpu().numpy()
        if t.ndim == 3 and t.shape[0] == 1:
            t = t[0]
        elif t.ndim == 3:
            t = np.transpose(t, (1, 2, 0))
        return t

    def to_mask(t):
        if torch.is_tensor(t):
            t = t.detach().cpu().numpy()
        return t.squeeze()

    fig, axes = plt.subplots(2, 4, figsize=(14, 9))

    # ---- x0 ----
    axes[0, 0].imshow(to_img(x0), cmap="gray")
    axes[0, 0].set_title("x0")

    # ---- m0 ----
    axes[0, 1].imshow(to_mask(m0), cmap="gray")
    axes[0, 1].set_title("m0")

    # ---- v ----
    v_img = to_img(v)
    if v_img.ndim == 2:
        v_mag = np.abs(v_img)
    elif v_img.ndim == 3:
        # [C,H,W] or [H,W,C]
        if v_img.shape[0] <= 3:
            v_mag = np.linalg.norm(v_img, axis=0)
        else:
            v_mag = np.linalg.norm(v_img, axis=-1)
    else:
        raise ValueError(f"Unexpected v shape: {v_img.shape}")

    im = axes[0, 2].imshow(v_mag, cmap="jet")
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)
    axes[0, 2].set_title("|v| (edit field)")

    # ---- x0 with metric ----
    axes[0, 3].imshow(to_img(x0), cmap="gray")
    axes[0, 3].contour(to_mask(x0_seg), colors="lime", linewidths=1)  # <<< x0 own segmentation
    axes[0, 3].set_title(
        f"x0\nPSNR={psnr_x0:.2f}, SSIM={ssim_x0:.3f}\n"
        f"Dice={dice_x0:.3f}, HD={hd_x0:.2f}",
        fontsize=10
    )

    # ---- x1 (GT) ----
    axes[1, 0].imshow(to_img(x1), cmap="gray")
    # axes[1, 0].contour(to_mask(x1_seg), colors="lime", linewidths=1)
    axes[1, 0].set_title("x1 (GT)")

    # ---- m1 ----
    axes[1, 1].imshow(to_mask(m1), cmap="gray")
    axes[1, 1].set_title("m1")

    # ---- x1_pred ----
    axes[1, 2].imshow(to_img(x1_pred), cmap="gray")
    axes[1, 2].set_title("x1_pred")

    # ---- x1_pred with metric ----
    axes[1, 3].imshow(to_img(x1_pred), cmap="gray")
    axes[1, 3].contour(to_mask(x1_seg), colors="lime", linewidths=1)       # GT contour in green
    axes[1, 3].contour(to_mask(x1_pred_seg), colors="red", linewidths=1)   # prediction contour in red
    axes[1, 3].set_title(
        f"x1_pred\nPSNR={psnr_pred:.2f}, SSIM={ssim_pred:.3f}\n"
        f"Dice={dice_pred:.3f}, HD={hd_pred:.2f}",
        fontsize=10
    )

    # ---------------- cosmetics ----------------
    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def plot_flow_edit_2x3(x0, m0, v, x1, m1, x1_pred, save_path):
    import matplotlib.pyplot as plt
    import os
    import torch

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def to_img(t):
        t = t.detach().cpu()
        if t.dim() == 3 and t.shape[0] == 1:
            t = t[0]
        elif t.dim() == 3:
            t = t.permute(1, 2, 0)
        return t.numpy()

    v_mag = torch.norm(v, dim=0)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    axes[0, 0].imshow(to_img(x0), cmap="gray")
    axes[0, 0].set_title("x0")

    axes[0, 1].imshow(to_img(m0), cmap="gray")
    axes[0, 1].set_title("m0")

    axes[0, 2].imshow(v_mag.cpu(), cmap="jet")
    axes[0, 2].set_title("v (edit field)")

    axes[1, 0].imshow(to_img(x1), cmap="gray")
    axes[1, 0].set_title("x1 (GT)")

    axes[1, 1].imshow(to_img(m1), cmap="gray")
    axes[1, 1].set_title("m1")

    axes[1, 2].imshow(to_img(x1_pred), cmap="gray")
    axes[1, 2].set_title("x1_pred (edited)")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def convert_variables(images: torch.Tensor,
                      timestamps: torch.Tensor,
                      device: torch.device) -> Tuple[torch.Tensor]:
    '''
    Some repetitive processing of variables.
    '''
    x_start = images[:, 0, ...].float().to(device)
    x_end = images[:, 1, ...].float().to(device)
    if images.shape[1] == 3:
        x_start_aug = images[:, 2, ...].float().to(device)
    t_list = timestamps[0].float().to(device)
    if images.shape[1] == 3:
        return [x_start, x_end, x_start_aug], t_list
    else:
        return [x_start, x_end], t_list

def numpy_variables(*tensors: torch.Tensor) -> Tuple[np.array]:
    '''
    Some repetitive numpy casting of variables.
    '''
    return [_tensor.cpu().detach().numpy().squeeze(0).transpose(1, 2, 0) for _tensor in tensors]

def plot_mask_contours(ax, mask, color, lw=2):
    mask = np.squeeze(mask)
    ax.contour(mask, levels=[0.5], colors=color, linewidths=lw)

def plot_2x4_training_vis(
    x0, x1,
    m0, m1, m1_pred,
    v, u,
    save_path
):
    """
    x*: [H,W] or [H,W,3]
    m*: [H,W]
    v,u: [H,W]
    """

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    plt.rcParams['font.family'] = 'serif'

    # =====================
    # Row 1
    # =====================
    axes[0,0].imshow(x0, cmap='gray')
    axes[0,0].set_title("x0")
    
    axes[0,1].imshow(m0, cmap='gray', vmin=0, vmax=1)
    axes[0,1].set_title("m0")

    im = axes[0,2].imshow(v, cmap='hot')
    axes[0,2].set_title("v")
    plt.colorbar(im, ax=axes[0,2], fraction=0.046)

    axes[0,3].imshow(m1_pred, cmap='gray', vmin=0, vmax=1)
    axes[0,3].set_title("m1_pred")

    # =====================
    # Row 2
    # =====================
    axes[1,0].imshow(x1, cmap='gray')
    axes[1,0].set_title("x1")

    axes[1,1].imshow(m1, cmap='gray', vmin=0, vmax=1)
    axes[1,1].set_title("m1")

    im = axes[1,2].imshow(u, cmap='hot')
    axes[1,2].set_title("u")
    plt.colorbar(im, ax=axes[1,2], fraction=0.046)

    # ---------- overlay ----------
    ax = axes[1,3]
    ax.imshow(np.zeros_like(m1), cmap='gray')

    plot_mask_contours(ax, m0, 'blue')
    plot_mask_contours(ax, m1, 'green')
    plot_mask_contours(ax, m1_pred, 'red')

    d01 = dice_coeff(m0 > 0.5, m1 > 0.5)
    hd01 = hausdorff(m0 > 0.5, m1 > 0.5)

    d_pred = dice_coeff(m1_pred > 0.5, m1 > 0.5)
    hd_pred = hausdorff(m1_pred > 0.5, m1 > 0.5)

    ax.set_title(
        f"Overlay\n"
        f"m0 vs m1: Dice={d01:.3f}, HD={hd01:.2f}\n"
        f"pred vs m1: Dice={d_pred:.3f}, HD={hd_pred:.2f}"
    )

    # =====================
    # =====================
    for ax in axes.flat:
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entry point.')
    parser.add_argument('--mode', help='`train` or `test`?', default='train')
    parser.add_argument('--gpu-id', help='Index of GPU device', default=0)
    parser.add_argument('--run-count', default=None, type=int)

    parser.add_argument('--dataset-name', default='brain_ucsf_growth', type=str)
    parser.add_argument('--target-dim', default='(256, 256)', type=ast.literal_eval)
    parser.add_argument('--output-save-folder', default='$ROOT/results/', type=str)
    parser.add_argument('--segmentor-ckpt', default='$ROOT/checkpoints/segment_retinaUCSF_seed1.pty', type=str)

    parser.add_argument('--model', default='PDF_intensity', type=str)
    parser.add_argument('--random-seed', default=1, type=int)
    parser.add_argument('--learning-rate', default=1e-4, type=float)
    parser.add_argument('--max-epochs', default=120, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--ode-max-t', default=5.0, type=float)        # only relevant to ODE. Bigger is slower.
    parser.add_argument('--ode-location', default='all_connections', type=str)  # only relevant to ODE
    parser.add_argument('--depth', default=5, type=int)                # only relevant to simple unet
    parser.add_argument('--num-filters', default=64, type=int)         # only relevant to simple unet
    parser.add_argument('--num-workers', default=8, type=int)
    parser.add_argument('--train-val-test-ratio', default='6:2:2', type=str)
    parser.add_argument('--max-training-samples', default=2048, type=int)
    parser.add_argument('--max-validation-samples', default=256, type=int)
    parser.add_argument('--max-testing-samples', default=500, type=int)
    parser.add_argument('--n-plot-per-epoch', default=4, type=int)

    parser.add_argument('--no-l2', action='store_true')  # only relevant to ODE
    parser.add_argument('--coeff-smoothness', default=0, type=float)  # only relevant to ODE
    parser.add_argument('--coeff-latent', default=0, type=float)
    parser.add_argument('--coeff-contrastive', default=0, type=float)
    parser.add_argument('--coeff-invariance', default=0, type=float)
    parser.add_argument('--pretrained-vision-model', default='convnext_tiny', type=str)

    args = vars(parser.parse_args())
    config = AttributeHashmap(args)
    config = parse_settings(config, log_settings=config.mode == 'train', run_count=config.run_count)

    assert config.mode in ['train', 'test', 'edit']

    seed_everything(config.random_seed)

    if config.mode == 'train':
        train(config=config)
    elif config.mode == 'test':
        test(config=config)
    elif config.mode == 'edit':
        edit(config=config)
