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

from nn.pdf import PDF_morph, PDF_intensity

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
    model_registry = {'PDF_morph': PDF_morph}
    if config.model not in model_registry:
        raise ValueError(f'Unsupported PDF model: {config.model}')

    model = model_registry[config.model](
        device=device,
        num_filters=config.num_filters,
        depth=config.depth,
        in_channels=num_image_channel,
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
        model, ema, optimizer, scheduler = train_epoch_PDF(
            config=config,
            device=device,
            train_set=train_set,
            model=model,
            epoch_idx=epoch_idx,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            backprop_freq=backprop_freq,
        )

        with ema.average_parameters():
            model.eval()
            val_pred_psnr, val_seg_dice_xT = val_epoch_PDF(
                config=config,
                device=device,
                val_set=val_set,
                model=model,
                epoch_idx=epoch_idx,
            )
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

def spatial_smoothness_loss(v, mask=None):
    """
    v: [B, C, H, W]   (image velocity)
    mask: [B, 1, H, W] or None  (optional mask to restrict region)
    """
    dx = v[:, :, :, 1:] - v[:, :, :, :-1]
    dy = v[:, :, 1:, :] - v[:, :, :-1, :]

    if mask is not None:
        mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        dx = dx * mask_x
        dy = dy * mask_y

    return (dx ** 2).mean() + (dy ** 2).mean()


def signed_distance(mask):
    mask = mask.astype(bool)
    return distance_transform_edt(~mask) - distance_transform_edt(mask)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mask_to_soft(mask, tau=2.0):
    return sigmoid(-signed_distance(mask) / tau)

def signed_distance_torch(mask: torch.Tensor) -> torch.Tensor:
    """
    mask: (B, 1, H, W) or (H, W), values in {0,1} or {0,255}
    return: signed distance map (same shape)
    """
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)

    mask = mask.float()
    mask = (mask > 0).float()

    dist_out = K.contrib.distance_transform(1.0 - mask)
    dist_in  = K.contrib.distance_transform(mask)

    return dist_out - dist_in


def mask_to_soft_torch(mask: torch.Tensor, tau: float = 2.0) -> torch.Tensor:
    sdf = signed_distance_torch(mask)
    return torch.sigmoid( sdf / tau)

def laplacian_2d(x):
    kernel = torch.tensor(
        [[0, 1, 0],
        [1,-4, 1],
        [0, 1, 0]],
        device=x.device,
        dtype=x.dtype
    ).view(1,1,3,3)
    return torch.nn.functional.conv2d(x, kernel, padding=1)

@torch.no_grad()
def compute_vx_vy_from_masks(m0, m1, delta_t,
                             sd_flow_tau=6.0,
                             flow_scale=3.0,
                             vel_sigma=1.0,
                             sd_band_width=10.0):
    """
    m0, m1: [B,1,H,W] binary masks or probabilities; use a consistent representation.
    returns vx, vy: [B,1,H,W]
    """

    B, _, H, W = m0.shape
    vx_all, vy_all = [], []

    for b in range(B):
        # signed distance
        sd0 = signed_distance((m0[b,0] > 0.5).cpu().numpy())
        sd1 = signed_distance((m1[b,0] > 0.5).cpu().numpy())

        # geometry proxy
        g0 = np.tanh(-sd0 / sd_flow_tau).astype(np.float32)
        g1 = np.tanh(-sd1 / sd_flow_tau).astype(np.float32)

        # optical flow (backward)
        v_back, u_back = optical_flow_tvl1(g1, g0)

        vy = -v_back / float(delta_t[b].item()) * flow_scale
        vx = -u_back / float(delta_t[b].item()) * flow_scale

        # smooth velocity
        vx = gaussian_filter(vx, vel_sigma)
        vy = gaussian_filter(vy, vel_sigma)

        # band support
        band = np.abs(sd0) < sd_band_width
        vx *= band
        vy *= band

        vx_all.append(torch.from_numpy(vx))
        vy_all.append(torch.from_numpy(vy))

    vx = torch.stack(vx_all, dim=0).unsqueeze(1).to(m0.device)
    vy = torch.stack(vy_all, dim=0).unsqueeze(1).to(m0.device)

    return vx, vy

def soft_phase_from_logit(l_t, tau=2.0):
    """
    l_t: logits [B,1,H,W]
    tau: controls interface thickness (2~4 usually good)
    """
    # Step 1: hard-ish mask for geometry (detach!)
    m = (l_t > 0).float().detach()

    # Step 2: signed distance for the lesion interface.
    sd = signed_distance_torch(m)

    # Step 3: convert to phase field
    c = torch.sigmoid(sd / tau)
    return c


def pde_subspace_loss(v_pred, c, vx, vy, front_mask, eps=1e-8):

    # ---------- spatial derivatives ----------
    # spatial derivatives
    dy, dx = torch.gradient(c, dim=(2,3))
    lap = (
        -4*c
        + torch.roll(c, 1, 2) + torch.roll(c, -1, 2)
        + torch.roll(c, 1, 3) + torch.roll(c, -1, 3)
    )

    adv = vx * dx + vy * dy   # <-- advection term

    phi1 = lap
    phi2 = c * (1.0 - c)
    phi3 = -adv

    B = v_pred.shape[0]
    loss = 0.0

    for b in range(B):
        mask = front_mask[b,0]

        y = v_pred[b,0][mask]          # [N]
        X = torch.stack([
            phi1[b,0][mask],
            phi2[b,0][mask],
            phi3[b,0][mask],
        ], dim=1)                       # [N,3]

        XtX = X.T @ X + eps * torch.eye(3, device=X.device)
        coef = torch.linalg.solve(XtX, X.T @ y)

        y_hat = X @ coef
        res = y - y_hat

        loss = loss + (res.pow(2).mean() / (y.pow(2).mean() + eps))

    return loss / B


def train_epoch_PDF(config,
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

    lam_m     = getattr(config, "pdf_lambda_mask", 1.0)
    lam_pde     = getattr(config, "pdf_lambda_pde", 0.01)
    train_loss = 0.0
    train_loss_mask = 0.0
    train_loss_pde = 0.0

    assert len(train_set) == len(train_set.dataset)
    num_train_samples = min(config.max_training_samples, len(train_set))
    plot_freq = max(num_train_samples // config.n_plot_per_epoch, 1)


    for iter_idx, batch in enumerate(tqdm(train_set)):
        if iter_idx > config.max_training_samples:
            break

        shall_plot = iter_idx % plot_freq == 0

        images, masks, timestamps, vx, vy = batch

        x_list, t_list = convert_variables(images, timestamps, device)
        m_list, _ = convert_variables(masks, timestamps, device)  # or move masks to device before this call

        vx = vx.to(device)
        vy = vy.to(device)

        x0, x1 = x_list      # [1,C,H,W]
        m0, m1 = m_list      # [1,1,H,W]
        t0, t1 = t_list[0], t_list[1]

        delta_t = (t1 - t0).view(1,1)
        delta_t = torch.clamp(delta_t, min=1e-4)

        m0 = mask_to_logit(m0)
        m1 = mask_to_logit(m1)

        s = torch.rand((x0.shape[0], 1), device=device)

        x_t = (1.0 - s) * x0 + s * x1
        m_t   = (1.0 - s) * m0 + s * m1

        u   = (m1 - m0) / delta_t

        mprob_t = logit_to_prob(m_t)
        v = model(x_t, mprob_t, s, delta_t)

        m1_pred = m0 + delta_t * v
     
        pde = True #False #
        
        loss_mask = torch.mean((v - u)   ** 2)

        
        if pde == True:
            c = soft_phase_from_logit(m0)
            front = (c > 0.05) & (c < 0.95)
            loss_pde = pde_subspace_loss(
                v_pred=v,
                c=c,
                vx=vx[:,0],
                vy=vy[:,0],
                front_mask=front
            )
        else:
            loss_pde = torch.zeros_like(loss_mask)

        loss = lam_m * loss_mask + lam_pde * loss_pde
        train_loss += loss.item()
        train_loss_mask += loss_mask.item()
        train_loss_pde += loss_pde.item()


        loss = loss / backprop_freq
        loss.backward()

        if iter_idx % config.batch_size == config.batch_size - 1:
            optimizer.step()
            optimizer.zero_grad()
            ema.update()

        if shall_plot:
            with torch.no_grad():
                m1_pred_bin = (m1_pred > 0).float()
                x0_np, x1_np = numpy_variables(x0, x1)
                m0_np, m1_np, m1_pred_np = numpy_variables(m0, m1, m1_pred_bin)
                v_np, u_np = numpy_variables(v, u)

                save_path = (
                    f"{config.save_folder}/train/"
                    f"epoch{epoch_idx:03d}_iter{iter_idx:05d}.png"
                )

                plot_2x4_training_vis(
                    x0_np, x1_np,
                    m0_np, m1_np, m1_pred_np,
                    v_np, u_np,
                    save_path
                )

    scheduler.step()

    log(
        f"Train PDF [{epoch_idx+1}/{config.max_epochs}] "
        f"loss: {train_loss/num_train_samples:.4f} "
        f"(mask {train_loss_mask/num_train_samples:.4f}, pde {train_loss_pde/num_train_samples:.4f}, λ={lam_m}, λ_pde={lam_pde})",
        filepath=config.log_dir,
        to_console=False
    )

    return model, ema, optimizer, scheduler

def signed_distance_from_mask(mask):
    """
    mask: [B,1,H,W] binary {0,1}
    return phi: signed distance, same shape
    """
    import cv2
    phi_list = []
    for b in range(mask.shape[0]):
        m = mask[b,0].detach().cpu().numpy().astype(np.uint8)
        dist_in  = cv2.distanceTransform(m, cv2.DIST_L2, 3)
        dist_out = cv2.distanceTransform(1-m, cv2.DIST_L2, 3)
        phi = dist_out - dist_in
        phi_list.append(torch.from_numpy(phi))
    phi = torch.stack(phi_list, dim=0).unsqueeze(1)
    return phi.to(mask.device).float()


@torch.no_grad()
def val_epoch(config: AttributeHashmap,
              device: torch.device,
              val_set: Dataset,
              model: torch.nn.Module,
              epoch_idx: int):
    val_recon_psnr, val_recon_ssim, val_pred_psnr, val_pred_ssim = 0, 0, 0, 0
    val_seg_dice_xT, val_seg_dice_gt = 0, 0

    if os.path.isfile(config.segmentor_ckpt):
        segmentor = torch.nn.Sequential(
            monai.networks.nets.DynUNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                kernel_size=[5, 5, 5, 5],
                filters=[16, 32, 64, 128],
                strides=[1, 1, 1, 1],
                upsample_kernel_size=[1, 1, 1, 1]),
            torch.nn.Sigmoid()).to(device)
        segmentor.load_state_dict(torch.load(config.segmentor_ckpt, map_location=device))
        segmentor.eval()
    else:
        print('Using identity segmentor fallback.')
        segmentor = torch.nn.Identity()

    assert len(val_set) == len(val_set.dataset)
    num_val_samples = min(config.max_validation_samples, len(val_set))
    plot_freq = num_val_samples // config.n_plot_per_epoch
    for iter_idx, (images, timestamps) in enumerate(tqdm(val_set)):
        shall_plot = iter_idx % plot_freq == 0

        if iter_idx > config.max_validation_samples:
            break

        assert images.shape[1] == 2
        assert timestamps.shape[1] == 2

        # images: [1, 2, C, H, W], containing [x_start, x_end]
        # timestamps: [1, 2], containing [t_start, t_end]
        x_list, t_list = convert_variables(images, timestamps, device)
        x_start, x_end = x_list

        x_start_recon = model(x=x_start, t=torch.zeros(1).to(device))
        x_end_recon = model(x=x_end, t=torch.zeros(1).to(device))

        x_end_pred = model(x=x_start, t=torch.diff(t_list) * config.t_multiplier)

        x_start_seg = segmentor(x_start) > 0.5
        x_end_seg = segmentor(x_end) > 0.5
        x_end_pred_seg = segmentor(x_end_pred) > 0.5

        x0_true, x0_recon, xT_true, xT_recon, xT_pred, x0_seg, xT_seg, xT_pred_seg = \
            numpy_variables(x_start, x_start_recon, x_end, x_end_recon, x_end_pred,
                            x_start_seg, x_end_seg, x_end_pred_seg)

        # NOTE: Convert to image with normal dynamic range.
        x0_true, x0_recon, xT_true, xT_recon, xT_pred = \
            cast_to_0to1(x0_true, x0_recon, xT_true, xT_recon, xT_pred)

        val_recon_psnr += psnr(x0_true, x0_recon) / 2 + psnr(xT_true, xT_recon) / 2
        val_recon_ssim += ssim(x0_true, x0_recon) / 2 + ssim(xT_true, xT_recon) / 2
        val_pred_psnr += psnr(xT_true, xT_pred)
        val_pred_ssim += ssim(xT_true, xT_pred)

        val_seg_dice_xT += dice_coeff(xT_seg, xT_pred_seg)
        val_seg_dice_gt += dice_coeff(x0_seg, xT_seg)

        if shall_plot:
            save_path_fig_sbs = '%s/val/figure_log_epoch%s_sample%s.png' % (
                config.save_folder, str(epoch_idx + 1).zfill(5), str(iter_idx + 1).zfill(5))
            plot_side_by_side(t_list, x0_true, xT_true, x0_recon, xT_recon, xT_pred, save_path_fig_sbs,
                              x0_true_seg=x0_seg, xT_pred_seg=xT_pred_seg, xT_true_seg=xT_seg)

    del segmentor

    val_recon_psnr, val_recon_ssim, val_pred_psnr, val_pred_ssim, val_seg_dice_xT, val_seg_dice_gt = \
        [item / num_val_samples for item in (
            val_recon_psnr, val_recon_ssim, val_pred_psnr, val_pred_ssim, val_seg_dice_xT, val_seg_dice_gt)]

    log('Validation [%s/%s] PSNR (recon): %.3f, SSIM (recon): %.3f, PSNR (pred): %.3f, SSIM (pred): %.3f, Dice(xT_true, xT_pred): %.3f, Dice(x0_true, xT_true): %.3f.'
        % (epoch_idx + 1, config.max_epochs, val_recon_psnr,
        val_recon_ssim, val_pred_psnr, val_pred_ssim, val_seg_dice_xT, val_seg_dice_gt),
        filepath=config.log_dir,
        to_console=False)

    return val_recon_psnr, val_pred_psnr, val_seg_dice_xT

@torch.no_grad()
def val_epoch_PDF(config: AttributeHashmap,
                   device: torch.device,
                   val_set: Dataset,
                   model: torch.nn.Module,
                   epoch_idx: int):

    model.eval()

    val_pred_psnr = 0.0
    val_pred_ssim = 0.0

    val_seg_dice_xT = 0.0     # Dice(pred mask, GT mask at T)
    val_seg_hd_xT   = 0.0     # Hausdorff(pred mask, GT mask at T)
    val_seg_dice_gt = 0.0     # Dice(GT mask at 0, GT mask at T) as growth indicator

    assert len(val_set) == len(val_set.dataset)
    num_val_samples = min(config.max_validation_samples, len(val_set))
    plot_freq = max(num_val_samples // config.n_plot_per_epoch, 1)

    for iter_idx, batch in enumerate(tqdm(val_set)):
        if iter_idx > config.max_validation_samples:
            break

        shall_plot = (iter_idx % plot_freq == 0)

        # -------------------------
        # -------------------------
        images, masks, timestamps, vx, vy = batch
        x_list, t_list = convert_variables(images, timestamps, device)
        m_list, _ = convert_variables(masks, timestamps, device)

        x0, x1 = x_list
        m0, m1 = m_list
        t0, t1 = t_list[0], t_list[1]

        delta_t = (t1 - t0).view(1, 1)
        delta_t = torch.clamp(delta_t, min=1e-4)

        m0 = mask_to_logit(m0.float())
        m1 = mask_to_logit(m1.float())

        # -------------------------
        # predict at s=0 and integrate to T (single-step Euler over clinical Δt)
        # -------------------------
        s0 = torch.zeros_like(delta_t)

        m0 = m0.float()  # keep consistent with training (prob/float input)
        m0_prob = logit_to_prob(m0)
        v0 = model(x0, m0_prob, s0, delta_t)

        m1_pred = m0 + delta_t * v0
        m1_pred = ((m1_pred) > 0).float()

        m0_bin = (m0 > 0).float()
        m1_bin = (m1 > 0).float()

        m0_np, m1_np, m1_pred_np = numpy_variables(
            m0_bin, m1_bin, m1_pred
        )

        # pred is already 0/1 float
        val_seg_dice_xT += dice_coeff(m1_pred_np, m1_np)
        val_seg_hd_xT   += hausdorff(m1_pred_np, m1_np)
        val_seg_dice_gt += dice_coeff(m0_np, m1_np)

        if shall_plot:
            m1_pred_bin = (m1_pred > 0.).float()
            x0_np, x1_np = numpy_variables(x0, x1)
            m0_np, m1_np, m1_pred_np = numpy_variables(m0, m1, m1_pred_bin)
            u0 = (m1 - m0) / delta_t
            v_np, u_np = numpy_variables(v0, u0)

            save_path = (
                f"{config.save_folder}/val/"
                f"epoch{epoch_idx:03d}_iter{iter_idx:05d}.png"
            )

            plot_2x4_training_vis(
                x0_np, x1_np,
                m0_np, m1_np, m1_pred_np,
                v_np, u_np,
                save_path
            )
    # -------------------------
    # -------------------------
    val_seg_dice_xT /= num_val_samples
    val_seg_hd_xT   /= num_val_samples
    val_seg_dice_gt /= num_val_samples

    log(
        'Validation PDF [%s/%s] '
        'Dice(xT_true, xT_pred): %.3f, HD(xT_true, xT_pred): %.2f, '
        'Dice(x0_true, xT_true): %.3f'
        % (epoch_idx + 1, config.max_epochs,
           val_seg_dice_xT, val_seg_hd_xT,
           val_seg_dice_gt),
        filepath=config.log_dir,
        to_console=False
    )

    return val_pred_psnr, val_seg_dice_xT

def plot_side_by_side_PDF(
    t_list,
    x0_true,
    xT_true,
    v0_img,
    delta_img,
    xT_pred,
    save_path: str,
    x0_true_seg=None,
    xT_true_seg=None,
    xT_pred_seg=None,
    v0_mask=None,          # mask logit velocity at s=0
    delta_mask=None,       # Δt * v0_mask
    mask_from_v=None       # NEW: mask predicted purely from velocity (soft or binary)
) -> None:
    """
    Side-by-side visualization for Masked TFM (image + mask).

    Layout:
      Col 1: GT images (t=0 / t=T)
      Col 2: image velocity & Δt*v_img
      Col 3: mask velocity heatmaps
      Col 4: mask-from-velocity (row 1) / predicted image (row 2)
      Col 5: absolute diffs
      Col 6: GT images with GT masks overlay
      Col 7: raw masks (input / predicted)
    """

    fig_sbs = plt.figure(figsize=(30, 12))
    plt.rcParams['font.family'] = 'serif'

    aspect_ratio = x0_true.shape[0] / x0_true.shape[1]

    # -------------------------
    # convert to RGB if needed
    # -------------------------
    if len(x0_true.shape) == 2 or x0_true.shape[-1] == 1:
        x0_true, xT_true, v0_img, delta_img, xT_pred = \
            gray_to_rgb(x0_true, xT_true, v0_img, delta_img, xT_pred)

    # -------------------------
    # -------------------------
    def show_mask_heat(ax, m, title):
        m = np.squeeze(m)
        im = ax.imshow(np.abs(m), cmap='hot')
        ax.set_title(title)
        ax.set_axis_off()
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def show_mask_binary(ax, m, title):
        m = np.squeeze(m).astype(np.float32)
        ax.imshow(m, cmap='gray', vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.set_axis_off()
        ax.set_aspect(aspect_ratio)

    ncols = 7

    # ============================================================
    # Col 1: GT images
    # ============================================================
    ax = fig_sbs.add_subplot(2, ncols, 1)
    ax.imshow(x0_true)
    ax.set_title(
        'GT(t=0), time: %s\n[vs GT(t=T)]: PSNR=%.2f, SSIM=%.3f'
        % (t_list[0].item(), psnr(x0_true, xT_true), ssim(x0_true, xT_true))
    )
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)

    ax = fig_sbs.add_subplot(2, ncols, 1 + ncols)
    ax.imshow(xT_true)
    ax.set_title('GT(t=T), time: %s' % t_list[1].item())
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)

    # ============================================================
    # Col 2: image velocity & Δt*v_img
    # ============================================================
    ax = fig_sbs.add_subplot(2, ncols, 2)
    ax.imshow(v0_img)
    ax.set_title('Image velocity v_img(s=0)')
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)

    ax = fig_sbs.add_subplot(2, ncols, 2 + ncols)
    ax.imshow(delta_img)
    ax.set_title('Δt · v_img')
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)

    # ============================================================
    # Col 3: mask velocity heatmaps
    # ============================================================
    ax = fig_sbs.add_subplot(2, ncols, 3)
    if v0_mask is not None:
        show_mask_heat(ax, v0_mask, 'Mask velocity |v_l|(s=0)')
    else:
        ax.set_title('Mask velocity (N/A)')
        ax.set_axis_off()

    ax = fig_sbs.add_subplot(2, ncols, 3 + ncols)
    if (delta_mask is not None) and (v0_mask is not None):
        show_mask_heat(ax, delta_mask, 'Δt · |v_l|')
    else:
        ax.set_title('Δt · mask vel (N/A)')
        ax.set_axis_off()

    # ============================================================
    # Col 4 Row 1: mask predicted purely from velocity
    # ============================================================
    ax = fig_sbs.add_subplot(2, ncols, 4)
    if mask_from_v is not None:
        m = np.squeeze(mask_from_v).astype(np.float32)
        ax.imshow(m, cmap='gray', vmin=0.0, vmax=1.0)

        if xT_true_seg is not None:
            dsc = dice_coeff(m > 0.5, xT_true_seg)
            ax.set_title('Mask from v\nDSC=%.3f' % dsc)
        else:
            ax.set_title('Mask from v')
    else:
        ax.set_title('Mask from v (N/A)')
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)

    # ============================================================
    # Col 4 Row 2: predicted image
    # ============================================================
    ax = fig_sbs.add_subplot(2, ncols, 4 + ncols)
    ax.imshow(xT_pred)
    ax.set_title(
        'Pred(t=T), time: %s → %s\n[vs GT(t=T)]: PSNR=%.2f, SSIM=%.3f'
        % (t_list[0].item(), t_list[1].item(),
           psnr(xT_pred, xT_true), ssim(xT_pred, xT_true))
    )
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)

    # ============================================================
    # Col 5: absolute differences
    # ============================================================
    ax = fig_sbs.add_subplot(2, ncols, 5)
    ax.imshow(np.abs(x0_true - xT_true))
    ax.set_title('|GT(t=0) - GT(t=T)|\nMAE=%.4f, MSE=%.4f'
                 % (np.mean(np.abs(x0_true - xT_true)),
                    np.mean((x0_true - xT_true) ** 2)))
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)

    ax = fig_sbs.add_subplot(2, ncols, 5 + ncols)
    ax.imshow(np.abs(xT_pred - xT_true))
    ax.set_title('|Pred - GT|\nMAE=%.4f, MSE=%.4f'
                 % (np.mean(np.abs(xT_pred - xT_true)),
                    np.mean((xT_pred - xT_true) ** 2)))
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)

    # ============================================================
    # Col 6: GT images with GT masks overlay
    # ============================================================
    ax = fig_sbs.add_subplot(2, ncols, 6)
    if x0_true_seg is not None:
        plot_contour(x0_true, x0_true_seg)
    ax.imshow(x0_true)
    if (x0_true_seg is not None) and (xT_true_seg is not None):
        ax.set_title('GT(t=0)+mask\nDSC(x0,xT)=%.3f'
                     % dice_coeff(x0_true_seg, xT_true_seg))
    else:
        ax.set_title('GT(t=0)+mask')
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)

    ax = fig_sbs.add_subplot(2, ncols, 6 + ncols)
    if xT_true_seg is not None:
        plot_contour(xT_true, xT_true_seg)
    ax.imshow(xT_true)
    ax.set_title('GT(t=T)+mask')
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)

    # ============================================================
    # Col 7: raw masks (input / predicted)
    # ============================================================
    ax = fig_sbs.add_subplot(2, ncols, 7)
    if x0_true_seg is not None:
        show_mask_binary(ax, x0_true_seg, 'Mask(t=0) input')
    else:
        ax.set_title('Mask(t=0) N/A')
        ax.set_axis_off()

    ax = fig_sbs.add_subplot(2, ncols, 7 + ncols)
    if xT_pred_seg is not None:
        if xT_true_seg is not None:
            dsc = dice_coeff(xT_pred_seg, xT_true_seg)
            hdv = hausdorff(xT_pred_seg, xT_true_seg)
            show_mask_binary(
                ax,
                xT_pred_seg,
                f'Pred mask(t=T)\nDSC={dsc:.3f}, HD={hdv:.2f}'
            )
        else:
            show_mask_binary(ax, xT_pred_seg, 'Pred mask(t=T)')
    else:
        ax.set_title('Pred mask(t=T) N/A')
        ax.set_axis_off()

    fig_sbs.tight_layout()
    fig_sbs.savefig(save_path)
    plt.close(fig=fig_sbs)


@torch.no_grad()
def tfm_predict(model, x0, delta_t, steps=20):
    """
    Multi-step integration for TFM:
    dx/ds = v(x, s, delta_t)
    """
    x = x0
    for k in range(steps):
        s = torch.full(
            (x.shape[0], 1),
            (k + 0.5) / steps,
            device=x.device,
        )
        v = model(x, s, delta_t)
        x = x + v * delta_t * (1.0 / steps)   # ds ONLY
    return x

def estimate_flow(target, source, min_size=8, margin=5):
    """
    Safe TV-L1 flow estimation.
    If input is degenerate, return zero flow.
    """
    if target.ndim >2:
        target = target.squeeze()
    if source.ndim > 2:
        source = source.squeeze()

    assert target.ndim == 2 and source.ndim == 2, \
        f"expect 2D arrays, got {target.shape}, {source.shape}"

    v, u = optical_flow_tvl1(
        target.astype(np.float32),
        source.astype(np.float32),
    )


    return np.stack([v, u], axis=0)

def enforce_brain_boundary_flow(
    flow_u,
    brain_mask,
    width=5.0,
    eps=1e-6,
):
    """
    Enforce brain boundary conditions on flow:
      1) remove outward normal component
      2) attenuate magnitude near boundary

    Parameters
    ----------
    flow_u : np.ndarray (2, H, W)
        flow_u[0] = v (dy), flow_u[1] = u (dx)
    brain_mask : np.ndarray (H, W), bool
    width : float
        attenuation band width (pixels)
    eps : float

    Returns
    -------
    flow_u : np.ndarray (2, H, W)
    """
    v = flow_u[0]
    u = flow_u[1]

    # signed distance field
    d = (
        distance_transform_edt(brain_mask)
        - distance_transform_edt(~brain_mask)
    )

    # normal vectors (point outward)
    gy, gx = np.gradient(d)
    mag = np.sqrt(gx**2 + gy**2) + eps
    nx = gx / mag
    ny = gy / mag

    # ---- 1. remove outward component ----
    dot = u * nx + v * ny
    mask_out = dot > 0
    u = u - mask_out * dot * nx
    v = v - mask_out * dot * ny

    # ---- 2. attenuate near boundary ----
    d_in = np.clip(d, 0.0, None)
    w = np.clip(d_in / width, 0.0, 1.0)

    flow_u[0] = v * w
    flow_u[1] = u * w
    return flow_u


def flow_weight_from_u(u2_pred, gamma=1.0, eps=1e-6):
    """
    Weight flow magnitude by tumor concentration.

    Parameters
    ----------
    u2_pred : np.ndarray (H, W), in [0,1]
    gamma : float
        gamma > 1  -> emphasize core, suppress boundary
        gamma < 1  -> boundary moves more
    """
    w = np.clip(u2_pred, 0.0, 1.0)
    return (w + eps) ** gamma

def warp_with_flow_brain_strict(image, flow_vu, brain_mask, out_fill=0.0):
    """
    STRICT brain-boundary warp:

      - sampling coords clipped to image bounds
      - sampling coords must land inside brain_mask,
        otherwise fall back to identity mapping
      - output is zeroed outside brain_mask

    Parameters
    ----------
    image : np.ndarray (H, W)
    flow_vu : np.ndarray (2, H, W)
        flow_vu[0] = v (dy), flow_vu[1] = u (dx)
    brain_mask : np.ndarray (H, W), bool
    out_fill : float

    Returns
    -------
    warped : np.ndarray (H, W)
    """
    H, W = image.shape
    v, u = flow_vu

    yy, xx = np.meshgrid(
        np.arange(H),
        np.arange(W),
        indexing="ij"
    )

    # propose source coords
    ysrc = yy + v
    xsrc = xx + u

    # numeric bounds
    ysrc = np.clip(ysrc, 0, H - 1)
    xsrc = np.clip(xsrc, 0, W - 1)

    # check if proposed source lands inside brain
    y0 = np.round(ysrc).astype(np.int32)
    x0 = np.round(xsrc).astype(np.int32)
    valid_src = brain_mask[y0, x0]

    # fallback to identity mapping
    ysrc2 = ysrc.copy()
    xsrc2 = xsrc.copy()
    ysrc2[~valid_src] = yy[~valid_src]
    xsrc2[~valid_src] = xx[~valid_src]

    warped = map_coordinates(
        image,
        [ysrc2.ravel(), xsrc2.ravel()],
        order=1,
        mode="nearest",
    ).reshape(H, W)

    # HARD enforce: output outside brain is fill
    warped[~brain_mask] = out_fill

    return warped


def warp_from_mask(
    image1: torch.Tensor,
    mask1: torch.Tensor,
    mask2: torch.Tensor,
    tau=2.0,
    gamma=1.8,
    boundary_width=5.0,
    eps=1e-6,
):
    """
    mask1, mask2 : torch.Tensor (H,W) or (1,H,W)
    image1       : torch.Tensor (H,W) or (1,H,W)
    return       : torch.Tensor (1,1,H,W)
    """

    # --------------------------------------------------
    # 0. Torch -> numpy (ONCE)
    # --------------------------------------------------
    if mask1.dim() > 2:
        mask1 = mask1.squeeze()
    if mask2.dim() > 2:
        mask2 = mask2.squeeze()
    if image1.dim() > 2:
        image1 = image1.squeeze()
    device = image1.device

    mask1_np = mask1.detach().cpu().numpy()
    mask2_np = mask2.detach().cpu().numpy()
    image1_np = (image1.detach().cpu().numpy()+1)/2


    # brain mask: non-zero support
    brain_mask = image1_np > 0

    # --------------------------------------------------
    # 1. Soft tumor fields
    # --------------------------------------------------
    c1 = mask_to_soft(mask1_np, tau)
    c2 = mask_to_soft(mask2_np, tau)

    # --------------------------------------------------
    # 2. Estimate flow (target -> source)
    # --------------------------------------------------
    target_f = np.sqrt(c2 + eps)
    source_f = np.sqrt(c1 + eps)

    # flow_u: (2, H, W), [0]=v(dy), [1]=u(dx)
    flow_u = estimate_flow(target_f, source_f)

    # --------------------------------------------------
    # 3. Brain safety constraints
    # --------------------------------------------------
    flow_u = flow_u.copy()

    flow_u[0][~brain_mask] = 0.0
    flow_u[1][~brain_mask] = 0.0

    flow_u = enforce_brain_boundary_flow(flow_u, brain_mask, width=boundary_width)

    w = flow_weight_from_u(c2, gamma=gamma)

    flow_u[0] *= w
    flow_u[1] *= w

    # --------------------------------------------------
    # 4. Warp image
    # --------------------------------------------------
    image2_pred_np = warp_with_flow_brain_strict(
        image=image1_np,
        flow_vu=flow_u,
        brain_mask=brain_mask,
        out_fill=0.0,
    )

    image2_pred_np = np.clip(image2_pred_np, 0.0, 1.0)

    # --------------------------------------------------
    # 5. numpy -> torch (ONCE)
    # --------------------------------------------------
    image2_pred = (
        torch.from_numpy(image2_pred_np * 2 - 1)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    return image2_pred


def gray_to_rgb(*tensors: torch.Tensor) -> Tuple[np.array]:
    rgb_list = []
    for item in tensors:
        assert len(item.shape) in [2, 3]
        if len(item.shape) == 3:
            assert item.shape[-1] == 1
            rgb_list.append(np.repeat(item, 3, axis=-1))
        else:
            rgb_list.append(np.repeat(item[..., None], 3, axis=-1))

    return rgb_list

def plot_contour(image, label):
    true_contours, _hierarchy = cv2.findContours(np.uint8(label),
                                                 cv2.RETR_TREE,
                                                 cv2.CHAIN_APPROX_NONE)
    for contour in true_contours:
        cv2.drawContours(image, contour, -1, (0.0, 1.0, 0.0), 2)

def overlay_mask_contour(ax, img, mask, color='r', lw=2):
    ax.imshow(img, cmap='gray')
    mask = np.squeeze(mask)
    ax.contour(mask, levels=[0.5], colors=color, linewidths=lw)

def visualize_v(v_np):
    v_np = np.squeeze(v_np)
    if v_np.ndim == 3:  # multi-channel
        v_np = np.linalg.norm(v_np, axis=0)
    return v_np

def plot_2x5_test_vis(
    x1, x2, x2_pred,
    m1, m2, m2_pred,
    v,
    seg1, seg2, seg2_pred,
    save_path
):
    """
    x*: image [H,W] or [H,W,3]
    m*: GT / pred mask [H,W]
    seg*: segmentor mask [H,W]
    v: velocity [H,W] or [C,H,W]
    """

    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    plt.rcParams['font.family'] = 'serif'

    # =========================
    # Dice / HD (GT masks)
    # =========================
    dice_m12 = dice_coeff(m1 > 0.5, m2 > 0.5)
    hd_m12   = hausdorff(m1 > 0.5, m2 > 0.5)

    dice_mpred = dice_coeff(m2_pred > 0.5, m2 > 0.5)
    hd_mpred   = hausdorff(m2_pred > 0.5, m2 > 0.5)

    # =========================
    # Dice / HD (segmentor)
    # =========================
    dice_seg12 = dice_coeff(seg1, seg2)
    hd_seg12   = hausdorff(seg1, seg2)

    dice_segpred = dice_coeff(seg2_pred, seg2)
    hd_segpred   = hausdorff(seg2_pred, seg2)

    # =========================
    # -------- Row 1 ----------
    # =========================

    axes[0,0].imshow(x1, cmap='gray')
    axes[0,0].set_title("t1 image")

    axes[0,1].imshow(m1, cmap='gray', vmin=0, vmax=1)
    axes[0,1].set_title(
        f"t1 GT mask\nvs t2: Dice={dice_m12:.3f}, HD={hd_m12:.2f}"
    )

    v_vis = visualize_v(v)
    im = axes[0,2].imshow(v_vis, cmap='hot')
    axes[0,2].set_title("predicted v")
    plt.colorbar(im, ax=axes[0,2], fraction=0.046)

    axes[0,3].imshow(x2_pred, cmap='gray')
    axes[0,3].set_title("pred t2 image")

    overlay_mask_contour(axes[0,4], x1, seg1, 'cyan')
    axes[0,4].set_title(
        f"t1 img + seg\nvs t2 seg: Dice={dice_seg12:.3f}, HD={hd_seg12:.2f}"
    )

    # =========================
    # -------- Row 2 ----------
    # =========================

    axes[1,0].imshow(x2, cmap='gray')
    axes[1,0].set_title("t2 image")

    axes[1,1].imshow(m2, cmap='gray', vmin=0, vmax=1)
    axes[1,1].set_title("t2 GT mask")

    axes[1,2].imshow(m2_pred, cmap='gray', vmin=0, vmax=1)
    axes[1,2].set_title(
        f"pred t2 mask\nDice={dice_mpred:.3f}, HD={hd_mpred:.2f}"
    )

    overlay_mask_contour(axes[1,3], x2_pred, seg2_pred, 'lime')
    axes[1,3].set_title(
        f"pred t2 img + seg\nDice={dice_segpred:.3f}, HD={hd_segpred:.2f}"
    )

    overlay_mask_contour(axes[1,4], x2, seg2, 'lime')
    axes[1,4].set_title("t2 img + seg (GT)")

    # =========================
    # =========================
    for ax in axes.flat:
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

def lesion_stats(x, m, eps=1e-6):
    # x: [B,1,H,W], m: [B,1,H,W]
    mask = (m > 0.5)
    vals = x[mask]
    mean = vals.mean()
    std  = vals.std() + eps
    return mean, std


import torch.nn.functional as F
# ---------- gaussian / DoG ----------
def _gaussian_blur(x, sigma):
    if sigma <= 0:
        return x
    k = int(6 * sigma + 1)
    if k % 2 == 0:
        k += 1

    device = x.device
    coords = torch.arange(k, device=device).float() - k // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / (g.sum() + 1e-12)

    C = x.shape[1]
    gx = g.view(1, 1, 1, k).repeat(C, 1, 1, 1)
    gy = g.view(1, 1, k, 1).repeat(C, 1, 1, 1)

    x = F.conv2d(x, gx, padding=(0, k // 2), groups=C)
    x = F.conv2d(x, gy, padding=(k // 2, 0), groups=C)
    return x


def _dog_hp(x, sigma1=1.0, sigma2=2.0):
    return _gaussian_blur(x, sigma1) - _gaussian_blur(x, sigma2)

# ---------- boundary weight ----------
def _make_boundary_weight(m, max_dist_px=8):
    m = (m > 0.5).float()

    def erode(x):
        return -F.max_pool2d(-x, 3, 1, 1)

    w = torch.zeros_like(m)
    prev = m
    steps = max(1, int(max_dist_px))
    for i in range(steps):
        nxt = erode(prev)
        ring = (prev - nxt).clamp(0, 1)
        wi = 1.0 - i / max(1, steps - 1)
        w = w + wi * ring
        prev = nxt

    return w.clamp(0, 1)


# ---------- LF texture matching ----------
def texture_project_mean_std(x, m, ref_mean, ref_std, eps=1e-6):
    mask = (m > 0.5).expand_as(x)
    if mask.sum() == 0:
        return x

    vals = x[mask]
    cur_mean = vals.mean()
    cur_std = vals.std() + eps

    vals = (vals - cur_mean) / cur_std
    vals = vals * (ref_std + eps) + ref_mean

    x_new = x.clone()
    x_new[mask] = vals
    return x_new


@torch.no_grad()
def rfm_edit_heun_texture_hf_2(
    model,
    x0: torch.Tensor,
    m_apply: torch.Tensor,   # t2 mask
    m_ref: torch.Tensor,     # t1 mask
    ref_mean: torch.Tensor,
    ref_std: torch.Tensor,
    steps: int = 50,
    delta_max: float = 0.05,
    clamp_min: float = -1.0,
    clamp_max: float = 1.0,
    sigma1: float = 1.0,
    sigma2: float = 2.0,
    alpha_hf: float = 0.25,
    boundary_width: int = 8,
):
    """
    Flow + LF(mean/std) + DIRECT HF injection from x0
    """

    x = x0
    B = x.shape[0]
    s_grid = torch.linspace(0.0, 1.0, steps + 1, device=x.device)

    # ---------- precompute HIGH-FREQ from x0 ----------
    hp0 = _dog_hp(x0, sigma1, sigma2)          # [B,C,H,W]
    hp0 = hp0 * (m_ref > 0.5).float()           # Keep only high-frequency details inside the t1 lesion.

    # boundary weight on target mask
    w = _make_boundary_weight(m_apply, boundary_width)

    for k in range(steps):
        s_k = s_grid[k]
        s_k1 = s_grid[k + 1]
        ds = s_k1 - s_k

        s_in  = torch.full((B, 1), float(s_k), device=x.device)
        s_in1 = torch.full((B, 1), float(s_k1), device=x.device)

        # ---------- flow (Heun) ----------
        v0 = model(x, m_apply, s_in)
        x_euler = x + ds * (m_apply * v0)

        v1 = model(x_euler, m_apply, s_in1)
        delta = 0.5 * ds * (m_apply * (v0 + v1))

        if delta_max is not None:
            delta = torch.clamp(delta, -delta_max, delta_max)

        x = x + delta

        # ---------- LF texture lock ----------
        x = texture_project_mean_std(x, m_apply, ref_mean, ref_std)

        # ---------- HF DIRECT injection ----------
        x = x + alpha_hf * w * hp0

        # ---------- clamp ----------
        x = torch.where(
            m_apply > 0,
            torch.clamp(x, clamp_min, clamp_max),
            x
        )

    return x


@torch.no_grad()
def test(config: AttributeHashmap):
    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')
    train_set, val_set, test_set, num_image_channel, max_t = \
        prepare_dataset(config=config)
    
    # Build the model
    model_registry = {'PDF_morph': PDF_morph}
    if config.model not in model_registry:
        raise ValueError(f'Unsupported PDF model: {config.model}')

    model = model_registry[config.model](
        device=device,
        num_filters=config.num_filters,
        depth=config.depth,
        in_channels=num_image_channel,
    )

    model.to(device)

    if os.path.isfile(config.segmentor_ckpt):
        segmentor = torch.nn.Sequential(
            monai.networks.nets.DynUNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                kernel_size=[5, 5, 5, 5],
                filters=[16, 32, 64, 128],
                strides=[1, 1, 1, 1],
                upsample_kernel_size=[1, 1, 1, 1]),
            torch.nn.Sigmoid()).to(device)
        segmentor.load_state_dict(torch.load(config.segmentor_ckpt, map_location=device))
        segmentor.eval()
    else:
        print('Using identity segmentor fallback.')
        segmentor = torch.nn.Identity()

    # Only relevant to ODE
    config.t_multiplier = config.ode_max_t / max_t

    assert len(test_set) == len(test_set.dataset)
    num_test_samples = min(config.max_testing_samples, len(test_set))

    for best_type in ['last', 'seg_dice']:
        if best_type == 'last':

            model.load_weights(config.model_save_path.replace('.pty', '_last.pty'), device=device)
            log('%s: Model weights successfully loaded.' % config.model,
                to_console=True)

            save_path_fig_summary = '%s/results_last/summary.png' % config.save_folder
            os.makedirs(os.path.dirname(save_path_fig_summary), exist_ok=True)

        elif best_type == 'pred_psnr':

            model.load_weights(config.model_save_path.replace('.pty', '_best_pred_psnr.pty'), device=device)
            log('%s: Model weights successfully loaded.' % config.model,
                to_console=True)

            save_path_fig_summary = '%s/results_best_pred_psnr/summary.png' % config.save_folder
            os.makedirs(os.path.dirname(save_path_fig_summary), exist_ok=True)

        elif best_type == 'seg_dice':

            model.load_weights(config.model_save_path.replace('.pty', '_best_seg_dice.pty'), device=device)
            log('%s: Model weights successfully loaded.' % config.model,
                to_console=True)

            save_path_fig_summary = '%s/results_best_seg_dice/summary.png' % config.save_folder
            os.makedirs(os.path.dirname(save_path_fig_summary), exist_ok=True)

        mse_loss = torch.nn.MSELoss()

        deltaT_list, psnr_list, ssim_list = [], [], []
        test_loss, test_recon_psnr, test_recon_ssim, test_pred_psnr, test_pred_ssim = 0, 0, 0, [], []
        test_seg_dice, test_seg_hd, test_residual_mae, test_residual_mae_lesion, test_residual_mae_healthy, test_residual_mse = [], [], [], [], [], []
        test_dm_l1_full, test_dm_l1_lesion, test_dm_l1_healthy = [], [], []
        test_seg_dice_gt = []

        test_mask_dice_pred = []
        test_mask_hd_pred   = []
        test_mask_dice_gt   = []

        for iter_idx, batch in enumerate(tqdm(test_set)):

            if iter_idx > config.max_testing_samples:
                break

            # ==========================================================
            # 1. Unpack batch (DIFFERENT for PDF)
            # ==========================================================
            if 'PDF' in config.model:
                images, masks, timestamps, vx, vy = batch
                x_list, t_list = convert_variables(images, timestamps, device)
                m_list, _      = convert_variables(masks, timestamps, device)

                x_start, x_end = x_list        # [1,C,H,W]
                m0, m1         = m_list        # [1,1,H,W]

            assert t_list.shape[0] == 2

            # ----------------------------------------------------------
            if 'PDF' in config.model:
                warp = True #False #
                intensity = True #False #

                if warp == True and intensity == False:
                    delta_t = torch.diff(t_list).view(1, 1).clamp(min=1e-4)

                    # mask condition (USE t0 only)
                    m0_prob = m0.float()

                    s0 = torch.zeros_like(delta_t)
                    v0_l = model(x_start, m0_prob, s0, delta_t)

                    # -------- multi-step integration --------
                    x = x_start
                    l = mask_to_logit(m0.float()) 

                    multi_steps = False #True #
                    if multi_steps:
                        steps = getattr(config, "tfm_steps", 20)

                        for k in range(steps):
                            s = torch.full(
                                (x.shape[0], 1),
                                (k + 0.5) / steps,
                                device=device
                            )
                            m_prob = torch.sigmoid(l)  
                            v_l = model(x, m_prob, s, delta_t)
                            l = l + v_l   * (delta_t / steps)
                            mask1 = (m_prob > 0.01).float()
                            mask2 = (torch.sigmoid(l) > 0.01).float()
                            x = warp_from_mask(x, mask1, mask2)

                    else:
                        m_prob = torch.sigmoid(l)  
                        s = torch.full(
                                (x.shape[0], 1),
                                0,
                                device=device
                            )
                        v_l = model(x, m_prob, s, delta_t)
                        l = l + v_l * delta_t

                        mask1 = m0.float()
                        mask2 = (torch.sigmoid(l) > 0.001).float()
                        x = warp_from_mask(x, mask1, mask2)
                        
                    x_end_pred = x
                    m_end_pred = (torch.sigmoid(l) > 0.001).float()
                    x_start_recon = x_start
                    x_end_recon   = x_end

                    s0   = torch.zeros_like(delta_t)
                    v0   = model(x_start, m0_prob, s0, delta_t)
                    disp = x_end_pred - x_start
                
                elif warp == True and intensity == True:
                    delta_t = torch.diff(t_list).view(1, 1).clamp(min=1e-4)

                    # mask condition (USE t0 only)
                    m0_prob = m0.float()

                    # -------- multi-step integration --------
                    x = x_start
                    l = mask_to_logit(m0.float()) 

                    
                    m_prob = torch.sigmoid(l)  
                    s = torch.full(
                            (x.shape[0], 1),
                            0,
                            device=device
                        )
                    v_l = model(x, m_prob, s, delta_t)
                    l = l + v_l * delta_t

                    mask1 = m0.float()
                    mask2 = (torch.sigmoid(l) > 0.001).float()
                    x = warp_from_mask(x, mask1, mask2)

                    model_edit = PDF_intensity(
                        device=device,
                        in_channels=num_image_channel,
                        num_filters=config.num_filters,
                        depth=config.depth,
                    )
                    model_edit.to(device)
                    model_edit.eval()

                    if not getattr(config, "intensity_model_ckpt", None):
                        raise ValueError("--intensity-model-ckpt is required when intensity editing is enabled.")
                    model_edit.load_weights(config.intensity_model_ckpt, device=device)

                    log(f'{config.model}: Model weights successfully loaded.',
                        to_console=True)
                    
                    ref_mean, ref_std = lesion_stats(x, mask1)
                    




                    steps = getattr(config, "rfm_edit_steps", 1)

                    x = rfm_edit_heun_texture_hf_2(
                        model_edit,
                        x,
                        m_apply=mask2,   # t2
                        m_ref=mask1,          # t1
                        ref_mean=ref_mean,
                        ref_std=ref_std,
                        steps=steps,
                        alpha_hf=0.05,
                        boundary_width=4
                    )


                        
                    x_end_pred = x
                    m_end_pred = (torch.sigmoid(l) > 0.001).float()
                    x_start_recon = x_start
                    x_end_recon   = x_end

                    s0   = torch.zeros_like(delta_t)
                    v0   = model(x_start, m0_prob, s0, delta_t)
                    disp = x_end_pred - x_start

                elif warp == False and intensity == True:
                    delta_t = torch.diff(t_list).view(1, 1).clamp(min=1e-4)

                    # mask condition (USE t0 only)
                    m0_prob = m0.float()

                    # -------- multi-step integration --------
                    x = x_start
                    l = mask_to_logit(m0.float()) 

                    
                    m_prob = torch.sigmoid(l)  
                    s = torch.full(
                            (x.shape[0], 1),
                            0,
                            device=device
                        )
                    v_l = model(x, m_prob, s, delta_t)
                    l = l + v_l * delta_t

                    mask1 = m0.float()
                    mask2 = (torch.sigmoid(l) > 0.01).float()

                    model_edit = PDF_intensity(
                        device=device,
                        in_channels=num_image_channel,
                        num_filters=config.num_filters,
                        depth=config.depth,
                    )
                    model_edit.to(device)
                    model_edit.eval()

                    if not getattr(config, "intensity_model_ckpt", None):
                        raise ValueError("--intensity-model-ckpt is required when intensity editing is enabled.")
                    model_edit.load_weights(config.intensity_model_ckpt, device=device)
                    log(f'{config.model}: Model weights successfully loaded.',
                        to_console=True)
                    
                    ref_mean, ref_std = lesion_stats(x, mask1)

                    steps = getattr(config, "rfm_edit_steps", 1)

                    x = rfm_edit_heun_texture_hf_2(
                        model_edit,
                        x,
                        m_apply=mask2,   # t2
                        m_ref=mask1,          # t1
                        ref_mean=ref_mean,
                        ref_std=ref_std,
                        steps=steps,
                        alpha_hf=0.25,
                        boundary_width=8
                    )
                        
                    x_end_pred = x
                    m_end_pred = (torch.sigmoid(l) > 0.01).float()
                    x_start_recon = x_start
                    x_end_recon   = x_end

                    s0   = torch.zeros_like(delta_t)
                    v0   = model(x_start, m0_prob, s0, delta_t)
                    disp = x_end_pred - x_start


            # ==========================================================
            # 3. (metrics, segmentation, plots, csv)
            # ==========================================================
            
            loss_recon = mse_loss(x_start, x_start_recon) + mse_loss(x_end, x_end_recon)
            loss_pred = mse_loss(x_end, x_end_pred)

            loss = loss_recon + loss_pred
            test_loss += loss.item()

            x_start_seg = segmentor(x_start) > 0.5
            x_end_seg = segmentor(x_end) > 0.5
            x_end_pred_seg = segmentor(x_end_pred) > 0.5

            x0_true, x0_recon, xT_true, xT_recon, xT_pred, x0_seg, xT_seg, xT_pred_seg = \
                numpy_variables(x_start, x_start_recon, x_end, x_end_recon, x_end_pred,
                                x_start_seg, x_end_seg, x_end_pred_seg)

            # NOTE: Convert to image with normal dynamic range.
            x0_true, x0_recon, xT_true, xT_recon, xT_pred = \
                cast_to_0to1(x0_true, x0_recon, xT_true, xT_recon, xT_pred)

            # Ground-truth annotation masks from the dataset, not the segmentor output.
            m0_gt_np, m1_gt_np, m_end_pred_np = numpy_variables(m0, m1, m_end_pred)
            m0_gt_bin = m0_gt_np > 0.5
            m1_gt_bin = m1_gt_np > 0.5
            
            # -----------------------------------------
            # GT mask & PDE-pred mask metrics
            # -----------------------------------------
            m_pred_bin = m_end_pred_np > 0.5


            test_recon_psnr += psnr(x0_true, x0_recon) / 2 + psnr(xT_true, xT_recon) / 2
            test_recon_ssim += ssim(x0_true, x0_recon) / 2 + ssim(xT_true, xT_recon) / 2
            test_pred_psnr.append(psnr(xT_true, xT_pred))
            test_pred_ssim.append(ssim(xT_true, xT_pred))

            if 'PDF' in config.model:
                test_seg_dice.append(dice_coeff(xT_pred_seg, xT_seg))
                test_seg_hd.append(hausdorff(xT_pred_seg, xT_seg))
                test_residual_mae.append(np.mean(np.abs(xT_pred - xT_true)))
                lesion_mae, healthy_mae = region_mae_from_seg(xT_pred, xT_true, m1_gt_bin)
                test_residual_mae_lesion.append(lesion_mae)
                test_residual_mae_healthy.append(healthy_mae)
                dm_full, dm_lesion, dm_healthy = difference_map_l1_from_seg(
                    x0_true, xT_true, xT_pred, m1_gt_bin
                )
                test_dm_l1_full.append(dm_full)
                test_dm_l1_lesion.append(dm_lesion)
                test_dm_l1_healthy.append(dm_healthy)
                test_residual_mse.append(np.mean((xT_pred - xT_true)**2))
                test_seg_dice_gt.append(dice_coeff(x0_seg, xT_seg))


                test_mask_dice_pred.append(dice_coeff(m_pred_bin, m1_gt_bin))
                test_mask_hd_pred.append(hausdorff(m_pred_bin, m1_gt_bin))
                test_mask_dice_gt.append(dice_coeff(m0_gt_bin, m1_gt_bin))

            else:
                test_seg_dice.append(dice_coeff(xT_pred_seg, xT_seg))
                test_seg_hd.append(hausdorff(xT_pred_seg, xT_seg))
                test_residual_mae.append(np.mean(np.abs(xT_pred - xT_true)))
                lesion_mae, healthy_mae = region_mae_from_seg(xT_pred, xT_true, m1_gt_bin)
                test_residual_mae_lesion.append(lesion_mae)
                test_residual_mae_healthy.append(healthy_mae)
                dm_full, dm_lesion, dm_healthy = difference_map_l1_from_seg(
                    x0_true, xT_true, xT_pred, m1_gt_bin
                )
                test_dm_l1_full.append(dm_full)
                test_dm_l1_lesion.append(dm_lesion)
                test_dm_l1_healthy.append(dm_healthy)
                test_residual_mse.append(np.mean((xT_pred - xT_true)**2))
                test_seg_dice_gt.append(dice_coeff(x0_seg, xT_seg))

            # Plot an overall scattering plot.
            deltaT_list.append(0)
            psnr_list.append(psnr(x0_true, x0_recon))
            ssim_list.append(ssim(x0_true, x0_recon))
            deltaT_list.append(0)
            psnr_list.append(psnr(xT_true, xT_recon))
            ssim_list.append(ssim(xT_true, xT_recon))
            deltaT_list.append((t_list[1] - t_list[0]).item())
            psnr_list.append(psnr(xT_true, xT_pred))
            ssim_list.append(ssim(xT_true, xT_pred))

            fig_summary = plt.figure(figsize=(12, 8))
            ax = fig_summary.add_subplot(2, 1, 1)
            ax.spines[['right', 'top']].set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.scatter(deltaT_list, psnr_list, color='black', s=50, alpha=0.5)
            ax.set_xlabel('Time difference', fontsize=20)
            ax.set_ylabel('PSNR', fontsize=20)
            ax = fig_summary.add_subplot(2, 1, 2)
            ax.spines[['right', 'top']].set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.scatter(deltaT_list, ssim_list, color='crimson', s=50, alpha=0.5)
            ax.set_xlabel('Time difference', fontsize=20)
            ax.set_ylabel('SSIM', fontsize=20)
            fig_summary.tight_layout()
            fig_summary.savefig(save_path_fig_summary)
            plt.close(fig=fig_summary)

            # Plot the side-by-side figures.
            save_path_fig_sbs = '%s/figure_%s.png' % (
                os.path.dirname(save_path_fig_summary), str(iter_idx + 1).zfill(5))
            
            if 'PDF' in config.model:
                vl_np = numpy_variables(v_l)

                plot_2x5_test_vis(
                x1=x0_true,
                x2=xT_true,
                x2_pred=xT_pred,
                m1=m0_gt_np,
                m2=m1_gt_np,
                m2_pred=m_end_pred_np,
                v=vl_np,
                seg1=x0_seg,
                seg2=xT_seg,
                seg2_pred=xT_pred_seg,
                save_path=save_path_fig_sbs
                )
            else:
                plot_side_by_side(t_list, x0_true, xT_true, x0_recon, xT_recon, xT_pred, save_path_fig_sbs,
                                x0_true_seg=x0_seg, xT_pred_seg=xT_pred_seg, xT_true_seg=xT_seg)

        test_loss = test_loss / num_test_samples
        test_recon_psnr = test_recon_psnr / num_test_samples
        test_recon_ssim = test_recon_ssim / num_test_samples

        test_pred_psnr = np.array(test_pred_psnr)
        test_pred_ssim = np.array(test_pred_ssim)
        test_seg_dice = np.array(test_seg_dice)
        test_seg_hd = np.array(test_seg_hd)
        test_residual_mae = np.array(test_residual_mae)
        test_residual_mae_lesion = np.array(test_residual_mae_lesion)
        test_residual_mae_healthy = np.array(test_residual_mae_healthy)
        test_dm_l1_full = np.array(test_dm_l1_full)
        test_dm_l1_lesion = np.array(test_dm_l1_lesion)
        test_dm_l1_healthy = np.array(test_dm_l1_healthy)
        test_residual_mse = np.array(test_residual_mse)
        test_seg_dice_gt = np.array(test_seg_dice_gt)

        growth_dice_thr = 0.9

        test_pred_psnr_minor_growth = test_pred_psnr[test_seg_dice_gt > growth_dice_thr]
        test_pred_ssim_minor_growth = test_pred_ssim[test_seg_dice_gt > growth_dice_thr]
        test_seg_dice_minor_growth = test_seg_dice[test_seg_dice_gt > growth_dice_thr]
        test_seg_hd_minor_growth = test_seg_hd[test_seg_dice_gt > growth_dice_thr]
        test_residual_mae_minor_growth = test_residual_mae[test_seg_dice_gt > growth_dice_thr]
        test_residual_mae_lesion_minor_growth = test_residual_mae_lesion[test_seg_dice_gt > growth_dice_thr]
        test_residual_mae_healthy_minor_growth = test_residual_mae_healthy[test_seg_dice_gt > growth_dice_thr]
        test_residual_mse_minor_growth = test_residual_mse[test_seg_dice_gt > growth_dice_thr]

        test_pred_psnr_major_growth = test_pred_psnr[test_seg_dice_gt <= growth_dice_thr]
        test_pred_ssim_major_growth = test_pred_ssim[test_seg_dice_gt <= growth_dice_thr]
        test_seg_dice_major_growth = test_seg_dice[test_seg_dice_gt <= growth_dice_thr]
        test_seg_hd_major_growth = test_seg_hd[test_seg_dice_gt <= growth_dice_thr]
        test_residual_mae_major_growth = test_residual_mae[test_seg_dice_gt <= growth_dice_thr]
        test_residual_mae_lesion_major_growth = test_residual_mae_lesion[test_seg_dice_gt <= growth_dice_thr]
        test_residual_mae_healthy_major_growth = test_residual_mae_healthy[test_seg_dice_gt <= growth_dice_thr]
        test_residual_mse_major_growth = test_residual_mse[test_seg_dice_gt <= growth_dice_thr]

        log('[Best %s] Test loss: %.3f, PSNR (recon): %.3f, SSIM (recon): %.3f, PSNR (pred): %.3f \u00B1 %.3f, SSIM (pred): %.3f \u00B1 %.3f'
            % (best_type, test_loss, test_recon_psnr, test_recon_ssim,
               *mean_and_sem(test_pred_psnr),
               *mean_and_sem(test_pred_ssim)) + \
            ' MAE (pred): %.3f \u00B1 %.3f, L1 lesion: %.3f \u00B1 %.3f, L1 healthy: %.3f \u00B1 %.3f, MSE (pred): %.3f \u00B1 %.3f, DSC (pred): %.3f \u00B1 %.3f, HD (pred): %.3f \u00B1 %.3f'
            % (*mean_and_sem(test_residual_mae),
               *mean_and_sem(test_residual_mae_lesion),
               *mean_and_sem(test_residual_mae_healthy),
               *mean_and_sem(test_residual_mse),
               *mean_and_sem(test_seg_dice),
               *mean_and_sem(test_seg_hd)),
            filepath=config.log_dir,
            to_console=True)

        log('Difference-map L1 [full/lesion/healthy]: %.3f \u00B1 %.3f / %.3f \u00B1 %.3f / %.3f \u00B1 %.3f'
            % (*mean_and_sem(test_dm_l1_full),
               *mean_and_sem(test_dm_l1_lesion),
               *mean_and_sem(test_dm_l1_healthy)),
            filepath=config.log_dir,
            to_console=True)

        log('Minor growth (GT dice > %s) PSNR (pred): %.3f \u00B1 %.3f, SSIM (pred): %.3f \u00B1 %.3f'
            % (growth_dice_thr,
               *mean_and_sem(test_pred_psnr_minor_growth),
               *mean_and_sem(test_pred_ssim_minor_growth)) + \
            ' MAE (pred): %.3f \u00B1 %.3f, MSE (pred): %.3f \u00B1 %.3f, DSC (pred): %.3f \u00B1 %.3f, HD (pred): %.3f \u00B1 %.3f'
            % (*mean_and_sem(test_residual_mae_minor_growth),
               *mean_and_sem(test_residual_mse_minor_growth),
               *mean_and_sem(test_seg_dice_minor_growth),
               *mean_and_sem(test_seg_hd_minor_growth)),
            filepath=config.log_dir,
            to_console=True)

        log('Major growth (GT dice <= %s) PSNR (pred): %.3f \u00B1 %.3f, SSIM (pred): %.3f \u00B1 %.3f'
            % (growth_dice_thr,
               *mean_and_sem(test_pred_psnr_major_growth),
               *mean_and_sem(test_pred_ssim_major_growth)) + \
            ' MAE (pred): %.3f \u00B1 %.3f, MSE (pred): %.3f \u00B1 %.3f, DSC (pred): %.3f \u00B1 %.3f, HD (pred): %.3f \u00B1 %.3f'
            % (*mean_and_sem(test_residual_mae_major_growth),
               *mean_and_sem(test_residual_mse_major_growth),
               *mean_and_sem(test_seg_dice_major_growth),
               *mean_and_sem(test_seg_hd_major_growth)),
            filepath=config.log_dir,
            to_console=True)

        # Save to csv.
        results_df = pd.DataFrame({
            'DICE(xT_true_seg, x0_true_seg)': test_seg_dice_gt,
            'PSNR(xT_true, xT_pred)': test_pred_psnr,
            'SSIM(xT_true, xT_pred)': test_pred_ssim,
            'DICE(xT_true_seg, xT_pred_seg)': test_seg_dice,
            'HD(xT_true_seg, xT_pred_seg)': test_seg_hd,
            'MAE(xT_true, xT_pred)': test_residual_mae,
            'L1_lesion(xT_true_seg)': test_residual_mae_lesion,
            'L1_healthy(xT_true_seg)': test_residual_mae_healthy,
            'DM_L1_full(xT_true-x0_true)': test_dm_l1_full,
            'DM_L1_lesion(xT_true-x0_true)': test_dm_l1_lesion,
            'DM_L1_healthy(xT_true-x0_true)': test_dm_l1_healthy,
            'MSE(xT_true, xT_pred)': test_residual_mse,
        })
        results_df.to_csv(config.log_dir.replace('log.txt', best_type + '.csv'))

    return


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


def region_mae_from_seg(x_pred: np.ndarray,
                        x_true: np.ndarray,
                        seg: np.ndarray) -> Tuple[float, float]:
    '''
    Compute per-pixel MAE inside lesion and healthy regions.

    The lesion region is defined by the ground-truth annotation mask.
    Empty regions fall back to 0.0 so evaluation summaries stay finite.
    '''
    diff = np.abs(x_pred - x_true)
    if diff.ndim == 3:
        if diff.shape[-1] == 1:
            diff = diff[..., 0]
        else:
            diff = np.mean(diff, axis=-1)
    mask = np.squeeze(seg).astype(bool)
    lesion_diff = diff[mask]
    healthy_diff = diff[~mask]
    lesion_mae = float(np.mean(lesion_diff)) if lesion_diff.size > 0 else 0.0
    healthy_mae = float(np.mean(healthy_diff)) if healthy_diff.size > 0 else 0.0
    return lesion_mae, healthy_mae


def difference_map_l1_from_seg(x_start: np.ndarray,
                               x_true: np.ndarray,
                               x_pred: np.ndarray,
                               seg: np.ndarray) -> Tuple[float, float, float]:
    """
    L1 on difference maps:
      gt_delta = x_true - x_start
      pred_delta = x_pred - x_start

    Returns full-image, lesion-region, and healthy-region L1.
    """
    gt_delta = x_true - x_start
    pred_delta = x_pred - x_start
    diff = np.abs(pred_delta - gt_delta)
    if diff.ndim == 3:
        if diff.shape[-1] == 1:
            diff = diff[..., 0]
        else:
            diff = np.mean(diff, axis=-1)
    mask = np.squeeze(seg).astype(bool)
    full_l1 = float(np.mean(diff)) if diff.size > 0 else 0.0
    lesion_diff = diff[mask]
    healthy_diff = diff[~mask]
    lesion_l1 = float(np.mean(lesion_diff)) if lesion_diff.size > 0 else 0.0
    healthy_l1 = float(np.mean(healthy_diff)) if healthy_diff.size > 0 else 0.0
    return full_l1, lesion_l1, healthy_l1


def mean_and_sem(values: np.ndarray) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values) / np.sqrt(values.size))

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
    parser.add_argument('--gpu-id', help='Index of GPU device', default=0, type=int)
    parser.add_argument('--run-count', default=None, type=int)

    parser.add_argument('--dataset-name', default='brain_ucsf_growth', type=str)
    parser.add_argument('--target-dim', default='(256, 256)', type=ast.literal_eval)
    parser.add_argument('--output-save-folder', default='$ROOT/results/', type=str)
    parser.add_argument('--segmentor-ckpt', default='$ROOT/checkpoints/segment_retinaUCSF_seed1.pty', type=str)
    parser.add_argument('--intensity-model-ckpt', default=None, type=str)

    parser.add_argument('--model', default='PDF_morph', type=str)
    parser.add_argument('--random-seed', default=1, type=int)
    parser.add_argument('--learning-rate', default=1e-4, type=float)
    parser.add_argument('--max-epochs', default=120, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--ode-max-t', default=5.0, type=float)
    parser.add_argument('--depth', default=5, type=int)                # only relevant to simple unet
    parser.add_argument('--num-filters', default=64, type=int)         # only relevant to simple unet
    parser.add_argument('--num-workers', default=8, type=int)
    parser.add_argument('--train-val-test-ratio', default='6:2:2', type=str)
    parser.add_argument('--max-training-samples', default=2048, type=int)
    parser.add_argument('--max-validation-samples', default=256, type=int)
    parser.add_argument('--max-testing-samples', default=500, type=int)
    parser.add_argument('--n-plot-per-epoch', default=4, type=int)

    parser.add_argument('--coeff-latent', default=0, type=float)
    parser.add_argument('--coeff-contrastive', default=0, type=float)
    parser.add_argument('--coeff-invariance', default=0, type=float)

    args = vars(parser.parse_args())
    config = AttributeHashmap(args)
    config = parse_settings(config, log_settings=config.mode == 'train', run_count=config.run_count)

    assert config.mode in ['train', 'test']

    seed_everything(config.random_seed)

    if config.mode == 'train':
        train(config=config)
    elif config.mode == 'test':
        test(config=config)
