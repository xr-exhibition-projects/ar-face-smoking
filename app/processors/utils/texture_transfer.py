import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F

from app.processors.utils import faceutil


def load_reference_image(path: Path, device: torch.device) -> Optional[torch.Tensor]:
    """Load a reference image from disk into a (C, H, W) tensor on the given device."""
    if not path or not path.exists():
        return None

    img = cv2.imread(str(path))
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img).permute(2, 0, 1).float().to(device)
    return tensor


def warp_triangle(img, src_tri, dst_tri, size):
    r1 = cv2.boundingRect(np.float32([src_tri]))
    r2 = cv2.boundingRect(np.float32([dst_tri]))

    src_tri_cropped = []
    dst_tri_cropped = []
    for idx in range(3):
        src_tri_cropped.append(((src_tri[idx][0] - r1[0]), (src_tri[idx][1] - r1[1])))
        dst_tri_cropped.append(((dst_tri[idx][0] - r2[0]), (dst_tri[idx][1] - r2[1])))

    img_cropped = img[r1[1]: r1[1] + r1[3], r1[0]: r1[0] + r1[2]]
    warp_mat = cv2.getAffineTransform(np.float32(src_tri_cropped), np.float32(dst_tri_cropped))
    dst_cropped = cv2.warpAffine(
        img_cropped,
        warp_mat,
        (r2[2], r2[3]),
        None,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(dst_tri_cropped), (1.0, 1.0, 1.0), 16, 0)
    dst_cropped = dst_cropped * mask
    return dst_cropped, mask, r2


def align_reference_to_target(ref_img: torch.Tensor, ref_kps: np.ndarray, target_size: int = 128) -> torch.Tensor:
    """Align reference image to standard arcface template of given size."""
    dst_kps = faceutil.get_arcface_template(image_size=target_size, mode='arcface128')
    dst_kps = np.squeeze(dst_kps)

    img_np = ref_img.permute(1, 2, 0).detach().cpu().numpy()
    if img_np.dtype != np.uint8:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)

    h = w = target_size
    img_warped = np.zeros((h, w, 3), dtype=np.float32)

    rh, rw = img_np.shape[:2]
    ref_points = ref_kps.tolist()
    dst_points = dst_kps.tolist()
    corners_ref = [
        [0, 0], [rw - 1, 0], [0, rh - 1], [rw - 1, rh - 1],
        [rw // 2, 0], [rw // 2, rh - 1], [0, rh // 2], [rw - 1, rh // 2]
    ]
    corners_dst = [
        [0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1],
        [w // 2, 0], [w // 2, h - 1], [0, h // 2], [w - 1, h // 2]
    ]
    ref_points.extend(corners_ref)
    dst_points.extend(corners_dst)

    ref_points = np.array(ref_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)

    rect = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(rect)
    for pt in dst_points:
        subdiv.insert((float(pt[0]), float(pt[1])))

    triangles = subdiv.getTriangleList()

    for tri in triangles:
        pts_dst = [(tri[0], tri[1]), (tri[2], tri[3]), (tri[4], tri[5])]
        idxs = []
        for pt in pts_dst:
            for idx, dp in enumerate(dst_points):
                if abs(dp[0] - pt[0]) < 1.0 and abs(dp[1] - pt[1]) < 1.0:
                    idxs.append(idx)
                    break
        if len(idxs) != 3:
            continue
        src_tri = [ref_points[i] for i in idxs]
        dst_tri = [dst_points[i] for i in idxs]

        warped_patch, mask, r2 = warp_triangle(img_np, src_tri, dst_tri, (w, h))
        x, y, w_r, h_r = r2
        if x < 0 or y < 0 or x + w_r > w or y + h_r > h:
            continue

        out_area = img_warped[y:y + h_r, x:x + w_r]
        mask_inv = 1.0 - mask
        img_warped[y:y + h_r, x:x + w_r] = out_area * mask_inv + warped_patch

    warped_tensor = torch.from_numpy(img_warped).permute(2, 0, 1).float().to(ref_img.device)
    return warped_tensor


def transfer_color_reinhard(source: torch.Tensor, target: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """
    Transfer color statistics from source to target using a Reinhard-style method.

    Args:
        source: reference tensor (C, H, W) in [0, 255].
        target: swap tensor (C, H, W) in [0, 255].
        strength: 0 disables, 1 matches source statistics, values > 1 additionally
            bleed the reference palette (low-frequency colors) for a stronger tint.
    """
    if strength <= 0:
        return target

    strength = float(strength)
    base_strength = min(strength, 1.0)
    overdrive = max(strength - 1.0, 0.0)

    src = source / 255.0
    tgt = target / 255.0

    src_lab = faceutil.rgb_to_lab(src, normalize=False)
    tgt_lab = faceutil.rgb_to_lab(tgt, normalize=False)

    src_mean = torch.mean(src_lab, dim=(1, 2), keepdim=True)
    src_std = torch.std(src_lab, dim=(1, 2), keepdim=True)
    tgt_mean = torch.mean(tgt_lab, dim=(1, 2), keepdim=True)
    tgt_std = torch.std(tgt_lab, dim=(1, 2), keepdim=True)

    tgt_lab_new = (tgt_lab - tgt_mean) / (tgt_std + 1e-6) * src_std + src_mean
    matched = faceutil.lab_to_rgb(tgt_lab_new, normalize=False) * 255.0

    blended = torch.lerp(target, matched, base_strength)

    if overdrive > 0:
        kernel = int(7 + overdrive * 8)
        kernel += 1 - kernel % 2  # ensure odd
        sigma = max(1.5, kernel / 4.0)
        low_freq = F.gaussian_blur(source, kernel_size=[kernel, kernel], sigma=[sigma, sigma])
        blended = torch.lerp(blended, low_freq, min(overdrive, 1.0))

    return torch.clamp(blended, 0, 255)


def transfer_texture_highpass(source: torch.Tensor, target: torch.Tensor, strength: float = 1.0, blur_radius: int = 5) -> torch.Tensor:
    """Blend high-frequency texture from source onto target."""
    if strength <= 0:
        return target

    src = source
    if src.shape[1:] != target.shape[1:]:
        src = F.resize(src, target.shape[-2:])

    kernel = blur_radius * 2 + 1
    src_blur = F.gaussian_blur(src, kernel_size=[kernel, kernel], sigma=[blur_radius, blur_radius])
    high_pass = src - src_blur
    result = target + high_pass * strength
    return torch.clamp(result, 0, 255)

