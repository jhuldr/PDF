import argparse
import csv
import os
import re

import cv2
import monai
import numpy as np
import torch

from data_utils.prepare_dataset import prepare_dataset
from utils.attribute_hashmap import AttributeHashmap
from utils.metrics import dice_coeff, hausdorff, psnr, ssim
from utils.seed import seed_everything
from nn.pdf import PDF_intensity, PDF_morph


def parse_target_dim(value):
    if isinstance(value, (tuple, list)):
        return tuple(int(v) for v in value)
    match = re.findall(r"\d+", str(value))
    if len(match) != 2:
        raise argparse.ArgumentTypeError("target dim must contain height and width")
    return int(match[0]), int(match[1])


def extract_time(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    match = re.search(r"(?:time[_-]?)?(\d+(?:\.\d+)?)", stem)
    if match is None:
        raise ValueError(f"Could not parse a time token from {path}")
    return float(match.group(1))


def load_image(path, device, target_dim):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, target_dim)
    img = img.astype(np.float32) / 255.0
    img = img * 2 - 1
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)


def load_mask(path, device, target_dim):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    mask = cv2.resize(mask, target_dim, interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.float32)
    return torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)


def to_numpy(x):
    x = x.squeeze().detach().cpu().numpy()
    return np.clip((x + 1) / 2, 0, 1)


def save_image(img, path):
    cv2.imwrite(path, (img * 255).astype(np.uint8))


def draw_contour_on_image(image, seg_mask):
    img = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(seg_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    return img


def build_morph_model(config, device, num_image_channel):
    model = PDF_morph(
        device=device,
        num_filters=config.num_filters,
        depth=config.depth,
        in_channels=num_image_channel,
    ).to(device)
    model.load_weights(config.model_ckpt, device=device)
    model.eval()
    return model


def build_intensity_model(config, device, num_image_channel):
    if not config.intensity_model_ckpt:
        return None
    model = PDF_intensity(
        device=device,
        in_channels=num_image_channel,
        num_filters=config.num_filters,
        depth=config.depth,
    ).to(device)
    model.load_weights(config.intensity_model_ckpt, device=device)
    model.eval()
    return model


def build_segmentor(config, device):
    segmentor = torch.nn.Sequential(
        monai.networks.nets.DynUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            kernel_size=[5, 5, 5, 5],
            filters=[16, 32, 64, 128],
            strides=[1, 1, 1, 1],
            upsample_kernel_size=[1, 1, 1, 1],
        ),
        torch.nn.Sigmoid(),
    ).to(device)
    segmentor.load_state_dict(torch.load(config.segmentor_ckpt, map_location=device))
    segmentor.eval()
    return segmentor


def predict_pdf(config, model, model_edit, x_start, m_start, delta_t):
    from pdf_stages.morph import lesion_stats, mask_to_logit, rfm_edit_heun_texture_hf_2, warp_from_mask

    delta_t = delta_t.view(1, 1).clamp(min=1e-4)
    x = x_start
    l = mask_to_logit(m_start.float())
    m_prob = torch.sigmoid(l)
    s = torch.zeros((x.shape[0], 1), device=x.device)
    v_l = model(x, m_prob, s, delta_t)
    l = l + v_l * delta_t

    mask_start = m_start.float()
    mask_pred = (torch.sigmoid(l) > config.mask_threshold).float()
    x_pred = warp_from_mask(x, mask_start, mask_pred)

    if model_edit is not None:
        ref_mean, ref_std = lesion_stats(x_pred, mask_start)
        x_pred = rfm_edit_heun_texture_hf_2(
            model_edit,
            x_pred,
            m_apply=mask_pred,
            m_ref=mask_start,
            ref_mean=ref_mean,
            ref_std=ref_std,
            steps=config.rfm_edit_steps,
            alpha_hf=config.alpha_hf,
            boundary_width=config.boundary_width,
        )

    return x_pred, mask_pred


def compute_metrics(x_start, x_end, x_pred, mask_start, mask_end, mask_pred, segmentor):
    x_start_np = to_numpy(x_start)
    x_end_np = to_numpy(x_end)
    x_pred_np = to_numpy(x_pred)

    x_start_seg = (segmentor(x_start) > 0.5).cpu().numpy().squeeze()
    x_end_seg = (segmentor(x_end) > 0.5).cpu().numpy().squeeze()
    x_pred_seg = (segmentor(x_pred) > 0.5).cpu().numpy().squeeze()

    mask_start_np = mask_start.cpu().numpy().squeeze() > 0.5
    mask_end_np = mask_end.cpu().numpy().squeeze() > 0.5
    mask_pred_np = mask_pred.cpu().numpy().squeeze() > 0.5

    return {
        "t1_t2_psnr": psnr(x_start_np, x_end_np),
        "t1_t2_ssim": ssim(x_start_np, x_end_np),
        "t1_t2_dice": dice_coeff(x_start_seg, x_end_seg),
        "t1_t2_hd": hausdorff(x_start_seg, x_end_seg),
        "pred_t2_psnr": psnr(x_pred_np, x_end_np),
        "pred_t2_ssim": ssim(x_pred_np, x_end_np),
        "pred_t2_dice": dice_coeff(x_pred_seg, x_end_seg),
        "pred_t2_hd": hausdorff(x_pred_seg, x_end_seg),
        "mask_t1_t2_dice": dice_coeff(mask_start_np, mask_end_np),
        "mask_pred_t2_dice": dice_coeff(mask_pred_np, mask_end_np),
        "mask_pred_t2_hd": hausdorff(mask_pred_np, mask_end_np),
    }


def write_metrics_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    metric_names = [k for k in rows[0].keys() if k != "sample_idx"]
    print("\n=== Dataset Metrics ===")
    for name in metric_names:
        values = np.array([row[name] for row in rows], dtype=np.float32)
        print(f"{name}: {values.mean():.4f} +/- {values.std():.4f}")


@torch.no_grad()
def run_dataset_test(config):
    device = torch.device(f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu")
    _, _, test_set, num_image_channel, max_t = prepare_dataset(config=config)
    config.t_multiplier = config.ode_max_t / max_t

    model = build_morph_model(config, device, num_image_channel)
    model_edit = build_intensity_model(config, device, num_image_channel)
    segmentor = build_segmentor(config, device)

    from pdf_stages.morph import convert_variables

    rows = []
    max_samples = min(config.max_testing_samples, len(test_set))
    for sample_idx, batch in enumerate(test_set):
        if sample_idx >= max_samples:
            break
        images, masks, timestamps, _, _ = batch
        x_list, t_list = convert_variables(images, timestamps, device)
        m_list, _ = convert_variables(masks, timestamps, device)
        x_start, x_end = x_list
        mask_start, mask_end = m_list
        delta_t = torch.diff(t_list).view(1, 1).clamp(min=1e-4)

        x_pred, mask_pred = predict_pdf(config, model, model_edit, x_start, mask_start, delta_t)
        metrics = compute_metrics(x_start, x_end, x_pred, mask_start, mask_end, mask_pred, segmentor)
        rows.append({"sample_idx": sample_idx, **metrics})

    if not rows:
        raise RuntimeError("No test samples were evaluated.")

    output_csv = os.path.join(config.output_folder, config.dataset_name, "metrics.csv")
    write_metrics_csv(rows, output_csv)
    print_summary(rows)
    print(f"\nSaved metrics to {output_csv}")


@torch.no_grad()
def run_pair_test(config):
    missing = [
        name for name in ("t1_path", "t2_path", "m1_path", "m2_path")
        if not getattr(config, name, None)
    ]
    if missing:
        args = ", ".join("--" + name.replace("_", "-") for name in missing)
        raise ValueError(f"Pair mode requires {args}.")

    device = torch.device(f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu")
    _, _, _, num_image_channel, max_t = prepare_dataset(config=config)
    config.t_multiplier = config.ode_max_t / max_t

    model = build_morph_model(config, device, num_image_channel)
    model_edit = build_intensity_model(config, device, num_image_channel)
    segmentor = build_segmentor(config, device)

    t1 = extract_time(config.t1_path)
    t2 = extract_time(config.t2_path)
    x1 = load_image(config.t1_path, device, config.target_dim)
    x2 = load_image(config.t2_path, device, config.target_dim)
    m1 = load_mask(config.m1_path, device, config.target_dim)
    m2 = load_mask(config.m2_path, device, config.target_dim)

    x_pred, mask_pred = predict_pdf(config, model, model_edit, x1, m1, torch.tensor([t2 - t1], device=device))

    metrics = compute_metrics(x1, x2, x_pred, m1, m2, mask_pred, segmentor)
    print("\n=== Pair Metrics ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    image_name = os.path.splitext(os.path.basename(config.t1_path))[0]
    save_dir = os.path.join(config.output_folder, "PDF_morph", image_name)
    os.makedirs(save_dir, exist_ok=True)

    x1_np = to_numpy(x1)
    x2_np = to_numpy(x2)
    x_pred_np = to_numpy(x_pred)
    x1_seg = (segmentor(x1) > 0.5).cpu().numpy().squeeze()
    x2_seg = (segmentor(x2) > 0.5).cpu().numpy().squeeze()
    x_pred_seg = (segmentor(x_pred) > 0.5).cpu().numpy().squeeze()

    save_image(x1_np, os.path.join(save_dir, "t1.png"))
    save_image(x2_np, os.path.join(save_dir, "t2.png"))
    save_image(x_pred_np, os.path.join(save_dir, "pred.png"))
    cv2.imwrite(os.path.join(save_dir, "t1_seg_overlay.png"), draw_contour_on_image(x1_np, x1_seg))
    cv2.imwrite(os.path.join(save_dir, "t2_seg_overlay.png"), draw_contour_on_image(x2_np, x2_seg))
    cv2.imwrite(os.path.join(save_dir, "pred_seg_overlay.png"), draw_contour_on_image(x_pred_np, x_pred_seg))


def main():
    parser = argparse.ArgumentParser(description="Run PDF dataset evaluation or pair inference.")
    parser.add_argument("--eval-mode", choices=("dataset", "pair"), default="dataset")
    parser.add_argument("--t1-path", default=None)
    parser.add_argument("--t2-path", default=None)
    parser.add_argument("--m1-path", default=None)
    parser.add_argument("--m2-path", default=None)

    parser.add_argument("--dataset-name", default="brain_ucsf_growth")
    parser.add_argument("--train-val-test-ratio", default="6:2:2", type=str)
    parser.add_argument("--target-dim", default=(256, 256), type=parse_target_dim)
    parser.add_argument("--random-seed", default=1, type=int)
    parser.add_argument("--num-workers", default=8, type=int)
    parser.add_argument("--max-testing-samples", default=100000, type=int)

    parser.add_argument("--model-ckpt", required=True)
    parser.add_argument("--intensity-model-ckpt", default=None)
    parser.add_argument("--segmentor-ckpt", required=True)

    parser.add_argument("--output-folder", default="./results_inference")
    parser.add_argument("--gpu-id", default=0, type=int)
    parser.add_argument("--num-filters", default=64, type=int)
    parser.add_argument("--depth", default=5, type=int)
    parser.add_argument("--ode-max-t", default=5.0, type=float)
    parser.add_argument("--mask-threshold", default=0.05, type=float)
    parser.add_argument("--rfm-edit-steps", default=1, type=int)
    parser.add_argument("--alpha-hf", default=0.25, type=float)
    parser.add_argument("--boundary-width", default=8, type=int)

    config = AttributeHashmap(vars(parser.parse_args()))
    config.target_dim = parse_target_dim(config.target_dim)
    seed_everything(config.random_seed)

    if config.eval_mode == "dataset":
        run_dataset_test(config)
    else:
        run_pair_test(config)


if __name__ == "__main__":
    main()
