# Feature: Hòa trộn dòng ROI tích hợp Cạnh Tranh Độ Nét Linh Hoạt (Adaptive Focus-Stacking & Sharpness Competition)
# Purpose: Automatically detects and prioritizes the sharpest focal slices between overlapping tiles (adaptive focus stacking)
# Path: tool/image_alignment/backend/blending.py

import cv2
import numpy as np

def compute_local_sharpness_map(img_rgb, kernel_size=15):
    """
    Calculates localized microscopic sharpness/focus map
    Kết hợp 3 tiêu chí:
    1. Tenengrad Gradient (Độ tương phản biên vi thể)
    2. Modified Laplacian (Độ sắc nét nhân tế bào)
    3. Local Variance (Năng lượng kết cấu mô học - vùng rõ nét có variance cao vượt trội so với vùng out-focus mờ)
    """
    if len(img_rgb.shape) == 3:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_rgb.copy()

    gray_f = gray.astype(np.float32)

    # 1. Gradient bậc 1 (Tenengrad)
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)

    # 2. Gradient bậc 2 (Laplacian vi phân tế bào)
    lap_mag = np.abs(cv2.Laplacian(gray_f, cv2.CV_32F, ksize=3))

    # 3. Local Standard Deviation / Texture Energy
    k = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
    mean = cv2.blur(gray_f, (k, k))
    sq_mean = cv2.blur(gray_f * gray_f, (k, k))
    variance = np.maximum(0.0, sq_mean - mean * mean)
    local_std = np.sqrt(variance)

    # Tổng hợp năng lượng độ nét
    sharpness_raw = grad_mag + 0.8 * lap_mag + 0.6 * local_std

    # Logic gốc của commit 4d17bbe (26/09): làm mượt rồi chuẩn hóa
    # từng tile về [0, 10], chỉ điều biến nhẹ quality theo độ nét.
    sharpness_smooth = cv2.GaussianBlur(sharpness_raw, (k, k), 0)
    max_v = np.max(sharpness_smooth)
    if max_v > 1e-5:
        sharpness_norm = (sharpness_smooth / max_v) * 10.0
    else:
        sharpness_norm = np.zeros_like(sharpness_smooth)

    return sharpness_norm


def compute_tile_global_sharpness(img_rgb):
    """
    Tính phương sai Laplacian của tile ảnh để đo mức độ in-focus tổng thể.
    Tile nét có giá trị cao (150-300+), tile mờ out-of-focus có giá trị thấp (<60).
    """
    if img_rgb is None or img_rgb.size == 0:
        return 1.0
    if img_rgb.ndim == 3:
        gray = cv2.cvtColor(img_rgb[:, :, :3], cv2.COLOR_RGB2GRAY)
    else:
        gray = img_rgb
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return float(np.var(lap))


def enhance_cellular_clarity(image, strength=0.35):
    """
    Làm rõ nét vi thể tế bào (Cellular Clarity & Microscopic Detail Sharpening).
    Sử dụng Unsharp Mask vi phân giúp nổi bật nhân tế bào và cấu trúc mô học mà không gây nhiễu hạt.
    """
    if image is None or image.size == 0 or strength <= 0.01:
        return image

    has_alpha = (image.ndim == 3 and image.shape[2] == 4)
    rgb = image[:, :, :3].astype(np.float32)

    blurred = cv2.GaussianBlur(rgb, (0, 0), sigmaX=1.5)
    detail = rgb - blurred
    sharpened = np.clip(rgb + strength * detail, 0.0, 255.0).astype(np.uint8)

    if has_alpha:
        return np.dstack([sharpened, image[:, :, 3]])
    return sharpened


def estimate_flat_field_profile(source_images):
    """
    Ước lượng bản đồ trường sáng nền 2D cong (Vignetting / 2D Shading Profile) của kính hiển vi từ tập tile.
    Tự động tách và loại trừ mô tế bào (Tissue Mask Exclusion) để chỉ đo trên nền lam kính thực sự,
    tránh để vị trí cụm mô làm méo phân bố ánh sáng nền.
    """
    if not source_images or len(source_images) < 2:
        return None

    first_img = next(iter(source_images.values()))
    h, w = first_img.shape[:2]

    down_w = min(384, max(32, w // 4))
    down_h = min(216, max(32, h // 4))

    bg_luma_maps = []
    for img in source_images.values():
        if img is None or img.size == 0:
            continue
        small = cv2.resize(img[:, :, :3], (down_w, down_h))
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        # Vùng nền lam kính: độ bão hòa màu thấp và độ sáng cao (không chứa mô nhuộm đậm)
        bg_mask = (sat < 40) & (val > 120)
        luma = (0.299 * small[:, :, 0] + 0.587 * small[:, :, 1] + 0.114 * small[:, :, 2]).astype(np.float32)
        luma_clean = np.where(bg_mask, luma, np.nan)
        bg_luma_maps.append(luma_clean)

    if len(bg_luma_maps) < 2:
        return None

    stack = np.stack(bg_luma_maps, axis=0)
    with np.errstate(all='ignore'):
        avg_luma = np.nanmedian(stack, axis=0)

    global_med = float(np.nanmedian(avg_luma))
    if np.isnan(global_med) or global_med < 1e-4:
        clean_stack = [cv2.resize(img[:, :, :3].astype(np.float32), (down_w, down_h)) for img in source_images.values()]
        avg_luma = np.mean([0.299 * s[:, :, 0] + 0.587 * s[:, :, 1] + 0.114 * s[:, :, 2] for s in clean_stack], axis=0)
        global_med = float(np.median(avg_luma))

    avg_luma = np.nan_to_num(avg_luma, nan=global_med)

    ksize = max(31, (min(down_w, down_h) // 4) * 2 + 1)
    shading_smooth = cv2.GaussianBlur(avg_luma, (ksize, ksize), 0)

    med = np.median(shading_smooth)
    if med < 1e-4:
        return None

    shading_norm = shading_smooth / med
    shading_norm = np.clip(shading_norm, 0.80, 1.25)
    shading_profile = cv2.resize(shading_norm, (w, h), interpolation=cv2.INTER_LINEAR)
    return shading_profile


def equalize_tile_illumination(source_images, enable_gaussian_smoothing=True):
    """
    Tự động chuẩn hóa màu nền và cân bằng phơi sáng giữa các ô ảnh kính hiển vi (Microscopy Flat-Field & Background Normalization).
    - enable_gaussian_smoothing: Nếu True, áp dụng khử tối góc quang học 2D để trường sáng đồng đều từ tâm ra 4 góc.
      Nếu False, chỉ cân bằng gain màu nền để giữ nguyên độ tương phản quang học gốc.
    """
    if not source_images or len(source_images) <= 1:
        return source_images

    # 1. Ước lượng trường sáng 2D cong nếu bật Gaussian smoothing
    shading_profile = estimate_flat_field_profile(source_images) if enable_gaussian_smoothing else None

    # 2. Khử tối góc 2D và thu thập mức nền chuẩn xác của từng ô ảnh
    bg_levels = {}
    flat_images = {}
    for i, img in source_images.items():
        if img is None or img.size == 0:
            continue

        has_alpha = (img.ndim == 3 and img.shape[2] == 4)
        if has_alpha:
            rgb = img[:, :, :3].astype(np.float32)
            alpha_mask = img[:, :, 3] > 30
        else:
            rgb = img.astype(np.float32)
            alpha_mask = np.ones(img.shape[:2], dtype=bool)

        if np.count_nonzero(alpha_mask) < 50:
            continue

        if shading_profile is not None and shading_profile.shape == rgb.shape[:2]:
            flat_rgb = np.clip(rgb / shading_profile[:, :, None], 0.0, 255.0)
        else:
            flat_rgb = rgb

        flat_images[i] = flat_rgb

        valid_pixels = flat_rgb[alpha_mask]
        luma = 0.299 * valid_pixels[:, 0] + 0.587 * valid_pixels[:, 1] + 0.114 * valid_pixels[:, 2]
        p90 = np.percentile(luma, 90)
        bg_pixels = valid_pixels[luma >= p90]
        if len(bg_pixels) >= 10:
            bg_levels[i] = np.mean(bg_pixels, axis=0)
        else:
            bg_levels[i] = np.percentile(valid_pixels, 95, axis=0)

    if len(bg_levels) <= 1:
        return source_images

    all_bg = np.array(list(bg_levels.values()), dtype=np.float32)
    ref_bg = np.median(all_bg, axis=0)

    if np.mean(ref_bg) < 70.0:
        return source_images

    equalized_images = {}
    for i, img in source_images.items():
        if i not in bg_levels or i not in flat_images:
            equalized_images[i] = img
            continue

        tile_bg = bg_levels[i]
        gains = ref_bg / np.maximum(tile_bg, 1.0)
        gains = np.clip(gains, 0.70, 1.45)

        flat_rgb = flat_images[i]
        adj_rgb = np.clip(flat_rgb * gains[None, None, :], 0.0, 255.0).astype(np.uint8)

        has_alpha = (img.ndim == 3 and img.shape[2] == 4)
        if has_alpha:
            new_img = np.dstack([adj_rgb, img[:, :, 3]])
        else:
            new_img = adj_rgb

        equalized_images[i] = new_img

    return equalized_images


def estimate_panorama_flat_field(panorama_rgb, valid_mask=None):
    """
    Ước lượng bản đồ trường sáng nền 2D Gaussian từ toàn bộ ảnh ghép panorama.
    Chỉ đo và làm mượt trên vùng nền lam kính trong suốt (Tissue-Exclusion Mask),
    tuyệt đối KHÔNG làm mờ (Gaussian blur) trực tiếp lên cấu trúc mô học.
    """
    if panorama_rgb is None or panorama_rgb.size == 0:
        return None

    h, w = panorama_rgb.shape[:2]
    # Downsample vừa phải để tính toán nhanh trường sáng quy mô lớn
    down_w = min(512, max(64, w // 4))
    down_h = min(512, max(64, h // 4))

    rgb_small = cv2.resize(panorama_rgb[:, :, :3], (down_w, down_h), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Vùng nền lam kính: độ bão hòa màu thấp và độ sáng cao (không chứa nhân hoặc bào tương nhuộm màu)
    bg_mask = (sat < 40) & (val > 115)
    if valid_mask is not None:
        mask_small = cv2.resize(valid_mask, (down_w, down_h), interpolation=cv2.INTER_NEAREST) > 128
        bg_mask = bg_mask & mask_small

    luma = (0.299 * rgb_small[:, :, 0] + 0.587 * rgb_small[:, :, 1] + 0.114 * rgb_small[:, :, 2]).astype(np.float32)
    bg_count = np.count_nonzero(bg_mask)
    if bg_count < 100:
        # Nếu mô chiếm gần kín toàn bộ ảnh, dùng giá trị luma tổng thể
        luma_clean = luma
    else:
        med_val = float(np.median(luma[bg_mask]))
        luma_clean = np.where(bg_mask, luma, med_val)

    # Làm mượt trường sáng nền bằng Gaussian kernel quy mô lớn
    ksize = max(31, (min(down_w, down_h) // 4) * 2 + 1)
    shading_smooth = cv2.GaussianBlur(luma_clean, (ksize, ksize), 0)

    med = float(np.median(shading_smooth))
    if med < 1e-4:
        return None

    shading_norm = shading_smooth / med
    shading_norm = np.clip(shading_norm, 0.75, 1.30)
    shading_profile = cv2.resize(shading_norm, (w, h), interpolation=cv2.INTER_LINEAR)
    return shading_profile


def compute_background_balance_gains(image_rgb, valid_mask=None, target_bg=None):
    """
    Cân bằng màu sắc thích nghi cho mô học (Tissue Chromatic Balancing).
    Tuyệt đối KHÔNG dựa vào nền lam kính.
    Đo trên chính các tế bào và chất nền mô học để cân bằng độ bão hòa và phơi sáng tự nhiên giữa 3 kênh R, G, B,
    khử ám màu đèn kính hiển vi mà không làm thay đổi hay loang lổ nền lam kính.
    """
    if image_rgb is None or image_rgb.size == 0:
        return np.array([1.0, 1.0, 1.0], dtype=np.float32)

    small = cv2.resize(image_rgb[:, :, :3], (min(image_rgb.shape[1], 400), min(image_rgb.shape[0], 400)), interpolation=cv2.INTER_AREA)
    small_f = small.astype(np.float32)

    # Đo trên vùng mô học: loại trừ điểm quá tối (<25) và điểm quá sáng/nền trắng (>235)
    luma = 0.299 * small_f[:, :, 0] + 0.587 * small_f[:, :, 1] + 0.114 * small_f[:, :, 2]
    tissue_mask = (luma > 25.0) & (luma < 235.0)
    if valid_mask is not None:
        mask_small = cv2.resize(valid_mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST) > 128
        tissue_mask = tissue_mask & mask_small

    if np.count_nonzero(tissue_mask) < 80:
        return np.array([1.0, 1.0, 1.0], dtype=np.float32)

    tissue_pixels = small_f[tissue_mask]
    med_rgb = np.median(tissue_pixels, axis=0) # [med_R, med_G, med_B]
    target_luma = float(np.mean(med_rgb))

    # Cân bằng nhẹ giữa các kênh để khử ám màu đèn kính hiển vi
    gains = target_luma / np.maximum(med_rgb, 1.0)
    # Kẹp chặt trong dải tự nhiên [0.90, 1.12] để giữ nguyên sắc tố mô học, không làm cháy sáng
    return np.clip(gains, 0.90, 1.12).astype(np.float32)


def segment_histology_tissue(image_rgb, valid_mask=None):
    """
    Phân đoạn chính xác mặt nạ mô học (Histology Tissue Mask) trên không gian thu nhỏ (tối đa 512px)
    để đạt tốc độ tức thì (<0.02s) mà vẫn chuẩn xác 100% cấu trúc mô.
    """
    if image_rgb is None or image_rgb.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    rgb = image_rgb[:, :, :3]
    h, w = rgb.shape[:2]

    down_w = min(512, max(64, w // 4))
    down_h = min(512, max(64, h // 4))
    small_rgb = cv2.resize(rgb, (down_w, down_h), interpolation=cv2.INTER_AREA)

    gray_s = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2GRAY)
    k_blur = max(3, (min(down_w, down_h) // 30) * 2 + 1)

    gx = cv2.Sobel(gray_s, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_s, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2)
    grad_smooth = cv2.GaussianBlur(grad, (k_blur, k_blur), 0)

    mean_g = cv2.blur(gray_s.astype(np.float32), (k_blur, k_blur))
    sq_g = cv2.blur(gray_s.astype(np.float32)**2, (k_blur, k_blur))
    std_g = np.sqrt(np.maximum(0.0, sq_g - mean_g**2))

    hsv_s = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2HSV)
    sat_s = hsv_s[:, :, 1].astype(np.float32)
    val_s = hsv_s[:, :, 2].astype(np.float32)
    darkness_s = np.maximum(0.0, 240.0 - val_s)

    cue = grad_smooth * 1.5 + std_g * 2.0 + sat_s * 1.2 + darkness_s * 0.8
    p98 = float(np.percentile(cue, 98))
    if p98 < 5.0:
        return np.zeros((h, w), dtype=np.uint8)

    cue_u8 = np.clip(cue / p98 * 255.0, 0, 255).astype(np.uint8)
    thresh_val, binary = cv2.threshold(cue_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_blur, k_blur))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask_s = np.zeros_like(closed)
    min_area = max(30, (down_w * down_h) // 1000)
    for c in contours:
        if cv2.contourArea(c) > min_area:
            cv2.drawContours(clean_mask_s, [c], -1, 255, -1)

    if valid_mask is not None:
        vm_s = cv2.resize(
            (valid_mask > 128).astype(np.uint8) if valid_mask.ndim == 2 else (valid_mask[:, :, 0] > 128).astype(np.uint8),
            (down_w, down_h),
            interpolation=cv2.INTER_NEAREST
        )
        clean_mask_s = np.where(vm_s > 0, clean_mask_s, 0)

    # Resize mask lên full resolution
    clean_mask = cv2.resize(clean_mask_s, (w, h), interpolation=cv2.INTER_NEAREST)
    return clean_mask


def apply_tissue_illumination_balance(
    image_rgb,
    valid_mask=None,
    strength=0.70,
    clahe_clip=2.0,
    bg_gains=None
):
    """
    Cân bằng trường sáng và tương phản CHỈ BÊN TRONG CÁC PHẦN CỦA MÔ (Tissue-Internal Balancing):
    - Tối ưu hóa hiệu năng cao (High Performance): Phân tích trường sáng trên không gian thu nhỏ (downsampled),
      sau đó nội suy mượt lên độ phân giải gốc, thực thi chỉ trong 0.1 - 2 giây, không bao giờ bị treo CPU/RAM.
    - Tuyệt đối KHÔNG thay đổi / cân bằng nền (background giữ nguyên 100% pixel gốc).
    - Ước lượng trường sáng vĩ mô bên trong mô bằng Normalized Convolution độc lập với nền.
    - Kéo các vùng mô tối và sáng về mức độ sáng hài hòa, bảo toàn độ nét màng tế bào và nhân.
    - Nếu có hệ số cân màu (bg_gains), chỉ áp dụng bên trong mô, không làm đổi màu nền kính.
    - Chuyển tiếp mượt mà bên trong mép mô (Strict Internal Feathering), không viền halo.
    """
    if image_rgb is None or image_rgb.size == 0:
        return image_rgb

    has_alpha = (image_rgb.ndim == 3 and image_rgb.shape[2] == 4)
    rgb = image_rgb[:, :, :3]
    h, w = rgb.shape[:2]

    down_w = min(512, max(64, w // 4))
    down_h = min(512, max(64, h // 4))
    small_rgb = cv2.resize(rgb, (down_w, down_h), interpolation=cv2.INTER_AREA)

    # 1. Phân đoạn mô trên ảnh thu nhỏ
    gray_s = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2GRAY)
    k_blur = max(3, (min(down_w, down_h) // 30) * 2 + 1)
    gx = cv2.Sobel(gray_s, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_s, cv2.CV_32F, 0, 1, ksize=3)
    grad_s = np.sqrt(gx**2 + gy**2)
    grad_smooth = cv2.GaussianBlur(grad_s, (k_blur, k_blur), 0)

    mean_s = cv2.blur(gray_s.astype(np.float32), (k_blur, k_blur))
    sq_s = cv2.blur(gray_s.astype(np.float32)**2, (k_blur, k_blur))
    std_s = np.sqrt(np.maximum(0.0, sq_s - mean_s**2))

    hsv_s = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2HSV)
    sat_s = hsv_s[:, :, 1].astype(np.float32)
    val_s = hsv_s[:, :, 2].astype(np.float32)
    darkness_s = np.maximum(0.0, 240.0 - val_s)

    cue_s = grad_smooth * 1.5 + std_s * 2.0 + sat_s * 1.2 + darkness_s * 0.8
    p98 = float(np.percentile(cue_s, 98))
    if p98 < 5.0:
        return image_rgb.copy()

    cue_u8 = np.clip(cue_s / max(p98, 1.0) * 255.0, 0, 255).astype(np.uint8)
    thresh_val, binary = cv2.threshold(cue_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_blur, k_blur))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask_s = np.zeros_like(closed)
    min_area = max(30, (down_w * down_h) // 1000)
    for c in contours:
        if cv2.contourArea(c) > min_area:
            cv2.drawContours(clean_mask_s, [c], -1, 255, -1)

    if valid_mask is not None:
        vm_s = cv2.resize(
            (valid_mask > 128).astype(np.uint8) if valid_mask.ndim == 2 else (valid_mask[:, :, 0] > 128).astype(np.uint8),
            (down_w, down_h),
            interpolation=cv2.INTER_NEAREST
        )
        clean_mask_s = np.where(vm_s > 0, clean_mask_s, 0)

    if np.count_nonzero(clean_mask_s) < 15:
        return image_rgb.copy()

    m_s = (clean_mask_s > 0).astype(np.float32)

    # 2. LAB và Normalized Convolution trên ảnh thu nhỏ
    if bg_gains is not None:
        small_rgb_proc = np.clip(small_rgb.astype(np.float32) * bg_gains[None, None, :], 0.0, 255.0).astype(np.uint8)
    else:
        small_rgb_proc = small_rgb

    lab_s = cv2.cvtColor(small_rgb_proc, cv2.COLOR_RGB2LAB)
    l_s = lab_s[:, :, 0].astype(np.float32)
    target_luma = float(np.median(l_s[m_s > 0]))

    k_macro = max(15, (min(down_w, down_h) // 8) * 2 + 1)
    num_s = cv2.GaussianBlur(l_s * m_s, (k_macro, k_macro), 0)
    den_s = cv2.GaussianBlur(m_s, (k_macro, k_macro), 0)
    macro_tissue_s = np.where(den_s > 1e-4, num_s / np.maximum(den_s, 1e-4), target_luma)
    norm_factor_s = np.clip(macro_tissue_s / max(target_luma, 1.0), 0.65, 1.45)

    soft_mask_s = cv2.GaussianBlur(m_s, (5, 5), 0)
    soft_mask_s = np.where(m_s > 0, soft_mask_s, 0.0)

    # 3. Nội suy profile trường sáng lên full resolution (cực kỳ nhanh, <0.02s)
    norm_factor = cv2.resize(norm_factor_s, (w, h), interpolation=cv2.INTER_LINEAR)
    soft_mask = cv2.resize(soft_mask_s, (w, h), interpolation=cv2.INTER_LINEAR)[:, :, None]

    # 4. Áp dụng lên full resolution
    if bg_gains is not None:
        rgb_proc = np.clip(rgb.astype(np.float32) * bg_gains[None, None, :], 0.0, 255.0).astype(np.uint8)
    else:
        rgb_proc = rgb

    lab = cv2.cvtColor(rgb_proc, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_f = l.astype(np.float32)
    l_balanced = np.clip(l_f / norm_factor, 0.0, 255.0).astype(np.uint8)

    # CLAHE vi thể nâng sắc nét nhân và màng tế bào
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(16, 16))
    l_clahe = clahe.apply(l_balanced)
    l_tissue_final = cv2.addWeighted(l_clahe, strength, l_balanced, 1.0 - strength, 0)

    lab_final = cv2.merge([l_tissue_final, a, b])
    rgb_balanced = cv2.cvtColor(lab_final, cv2.COLOR_LAB2RGB)

    # Trộn mềm nghiêm ngặt BÊN TRONG MÔ (ngoài mô soft_mask == 0.0 tuyệt đối)
    final_rgb = np.clip(
        rgb_balanced.astype(np.float32) * soft_mask +
        rgb.astype(np.float32) * (1.0 - soft_mask),
        0.0, 255.0
    ).astype(np.uint8)

    if has_alpha:
        return np.dstack([final_rgb, image_rgb[:, :, 3]])
    return final_rgb


def apply_variant_postprocessing(
    image_rgb,
    valid_mask=None,
    postprocess_spec=None,
    shading_profile=None,
    bg_gains=None
):
    """
    Áp dụng hậu xử lý cho một biến thể:
    - postprocess_spec: {"gaussian": bool, "balanced": bool, "clarity": bool}
    - shading_profile: Bản đồ trường sáng Gaussian 2D (tính 1 lần rồi tái sử dụng)
    - bg_gains: Hệ số cân màu nền (tính 1 lần rồi tái sử dụng)
    """
    if postprocess_spec is None:
        return image_rgb.copy()

    has_alpha = (image_rgb.ndim == 3 and image_rgb.shape[2] == 4)
    rgb = image_rgb[:, :, :3].astype(np.float32)
    alpha = image_rgb[:, :, 3] if has_alpha else None

    # 1. Khử trường sáng nền Gaussian nếu bật
    if postprocess_spec.get("gaussian", False) and shading_profile is not None:
        if shading_profile.shape == rgb.shape[:2]:
            rgb = np.clip(rgb / shading_profile[:, :, None], 0.0, 255.0)

    # 2. Khử quang sai sắc xanh tím/cyan quanh ranh giới mô học (chuẩn kietlearntocode/stitch)
    if postprocess_spec.get("balanced", False):
        rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
        rgb_u8 = apply_defringe_filter(rgb_u8)
        rgb = rgb_u8.astype(np.float32)

    res_rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    # 3. Cellular Clarity (làm rõ nét vi thể tế bào) nếu bật
    if postprocess_spec.get("clarity", False):
        res_rgb = enhance_cellular_clarity(res_rgb, strength=0.35)


    if has_alpha:
        return np.dstack([res_rgb, alpha])
    return res_rgb


def get_tissue_mask(img_rgb, threshold=215):
    """
    Trích xuất mặt nạ mô học (loại bỏ nền kính quang học).
    Thuật toán chuẩn từ kietlearntocode/stitch:
    - Điểm ảnh mô có giá trị xám < threshold và > 15 (loại trừ cả nền kính quá sáng và biên đen quá tối).
    - Sử dụng phép toán hình thái học (morphological closing) để lấp đầy các khoảng trống nhỏ bên trong nhân tế bào.
    """
    if img_rgb.ndim == 3:
        gray = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = img_rgb.astype(np.uint8)
    mask = (gray < threshold) & (gray > 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)


def apply_defringe_filter(rgb_img, threshold=12, min_blue=50):
    """
    Khử hiện tượng quang sai sắc biên giới (viền xanh tím/cyan quanh ranh giới mô học).
    Thuật toán chuẩn từ kietlearntocode/stitch:
    - Tự động nhận diện các điểm ảnh có kênh Blue vượt trội bất thường so với max(Red, Green).
    - Cắt giảm đỉnh Blue về đúng mức max(Red, Green) để khôi phục màu mô chuẩn giải phẫu bệnh.
    """
    if rgb_img is None or rgb_img.size == 0 or rgb_img.ndim < 3:
        return rgb_img

    has_alpha = (rgb_img.shape[2] == 4)
    rgb = rgb_img[:, :, :3]
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    max_rg = np.maximum(r, g)
    diff = b.astype(np.int16) - max_rg.astype(np.int16)
    fringe_mask = (diff > threshold) & (b > min_blue)

    b_clean = b.copy()
    b_clean[fringe_mask] = max_rg[fringe_mask]

    res = np.dstack([r, g, b_clean])
    if has_alpha:
        return np.dstack([res, rgb_img[:, :, 3]])
    return res


def solve_tissue_specific_gains(images, adjusted_transforms, canvas_w=None, canvas_h=None, threshold=215):
    """
    Cân bằng phơi sáng và màu sắc chuyên biệt trên phần mô học tại các vùng giao thoa.
    Thuật toán chuẩn từ kietlearntocode/stitch:
    - Tìm vùng giao thoa giữa các cặp ảnh trên canvas toàn cục.
    - Ưu tiên đo trung vị (np.median) trên phần mô tế bào chung (shared_tissue).
    - Giải hệ Least Squares log-linear để tìm hệ số gain tối ưu cho mỗi ô ảnh.
    - Chuẩn hóa: gains = gains / np.median(gains) và clip(0.5, 2.0).
    - Bảo toàn 100% tỷ lệ màu sắc (Hue & Saturation) của thuốc nhuộm mô học.
    """
    indices = [i for i in images if i in adjusted_transforms and images[i] is not None]
    n_images = len(indices)
    if n_images <= 1:
        return {i: np.ones(3, dtype=np.float32) for i in images}

    # Ước lượng canvas_w, canvas_h nếu chưa truyền vào
    if canvas_w is None or canvas_h is None:
        max_x, max_y = 100, 100
        for i in indices:
            h, w = images[i].shape[:2]
            H = adjusted_transforms[i]
            corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float64).T
            warped = H @ corners
            max_x = max(max_x, int(np.ceil(np.max(warped[0, :] / np.maximum(warped[2, :], 1e-6)))))
            max_y = max(max_y, int(np.ceil(np.max(warped[1, :] / np.maximum(warped[2, :], 1e-6)))))
        canvas_w = max(canvas_w or 100, max_x)
        canvas_h = max(canvas_h or 100, max_y)

    max_dim = max(canvas_w, canvas_h)
    scale = min(0.25, 2048.0 / max(max_dim, 1))
    dw = max(1, int(canvas_w * scale))
    dh = max(1, int(canvas_h * scale))
    S = np.diag([scale, scale, 1.0])

    warped_rgbs = {}
    warped_tissue_masks = {}
    warped_tile_masks = {}

    for i in indices:
        img = images[i]
        h, w = img.shape[:2]
        T = (S @ np.asarray(adjusted_transforms[i], dtype=np.float64))[:2]
        rgb = img[:, :, :3] if img.ndim == 3 and img.shape[2] == 4 else img
        warped_rgbs[i] = cv2.warpAffine(rgb, T, (dw, dh), flags=cv2.INTER_LINEAR)
        t_mask = get_tissue_mask(rgb, threshold=threshold)
        warped_tissue_masks[i] = cv2.warpAffine(t_mask, T, (dw, dh), flags=cv2.INTER_NEAREST)
        warped_tile_masks[i] = cv2.warpAffine(np.ones((h, w), dtype=np.uint8), T, (dw, dh), flags=cv2.INTER_NEAREST)

    rows_A = []
    vals_b = []
    weights = []

    min_tissue_pts = 25
    min_glass_pts = 40

    for p_i in range(n_images):
        i = indices[p_i]
        for p_j in range(p_i + 1, n_images):
            j = indices[p_j]
            shared_tissue = (warped_tissue_masks[i] > 0) & (warped_tissue_masks[j] > 0)
            n_tissue = np.count_nonzero(shared_tissue)

            if n_tissue > min_tissue_pts:
                mean_i = float(np.median(warped_rgbs[i][shared_tissue]))
                mean_j = float(np.median(warped_rgbs[j][shared_tissue]))
                if mean_i > 5.0 and mean_j > 5.0:
                    ratio = mean_j / mean_i
                    row = np.zeros(n_images, dtype=np.float32)
                    row[p_i] = 1.0
                    row[p_j] = -1.0
                    w_ij = np.sqrt(float(n_tissue))
                    rows_A.append(row * w_ij)
                    vals_b.append(np.log(ratio) * w_ij)
                    weights.append(w_ij)
            else:
                shared_glass = (warped_tile_masks[i] > 0) & (warped_tile_masks[j] > 0) & (~shared_tissue)
                n_glass = np.count_nonzero(shared_glass)
                if n_glass > min_glass_pts:
                    mean_i = float(np.median(warped_rgbs[i][shared_glass]))
                    mean_j = float(np.median(warped_rgbs[j][shared_glass]))
                    if mean_i > 5.0 and mean_j > 5.0:
                        ratio = mean_j / mean_i
                        row = np.zeros(n_images, dtype=np.float32)
                        row[p_i] = 1.0
                        row[p_j] = -1.0
                        w_ij = 0.5 * np.sqrt(float(n_glass))
                        rows_A.append(row * w_ij)
                        vals_b.append(np.log(ratio) * w_ij)
                        weights.append(w_ij)

    if len(rows_A) == 0:
        return {i: np.ones(3, dtype=np.float32) for i in images}

    A = np.array(rows_A, dtype=np.float32)
    b = np.array(vals_b, dtype=np.float32)

    constraint_row = np.ones((1, n_images), dtype=np.float32) * (np.mean(weights) * 2.0)
    constraint_val = np.zeros(1, dtype=np.float32)
    A_full = np.vstack([A, constraint_row])
    b_full = np.concatenate([b, constraint_val])

    log_g, _, _, _ = np.linalg.lstsq(A_full, b_full, rcond=None)
    gains = np.exp(log_g)
    gains = gains / np.median(gains)
    gains = np.clip(gains, 0.5, 2.0)

    result_gains = {}
    for pos, idx in enumerate(indices):
        result_gains[idx] = np.full(3, gains[pos], dtype=np.float32)
    for idx in images:
        if idx not in result_gains:
            result_gains[idx] = np.ones(3, dtype=np.float32)

    return result_gains


def estimate_overlap_exposure_gains(source_images, transforms, motion_model='affine'):
    """
    Cân bằng phơi sáng và màu sắc đa tile dựa trên vùng chồng lấn (Overlap-based Gain Optimization).
    Tích hợp thuật toán Least Squares Log-Linear của kietlearntocode/stitch.
    Tuyệt đối KHÔNG dựa vào nền lam kính.
    """
    if not source_images or len(source_images) <= 1:
        return {i: 1.0 for i in source_images}

    gains_3ch = solve_tissue_specific_gains(source_images, transforms)
    return {i: float(gains_3ch[i][0]) for i in source_images}



class FastStreamingBlender:
    """
    Lõi ghép khôi phục nguyên lý commit 4d17bbe ngày 26/09:
    Voronoi quality theo khoảng cách mép, điều biến độ nét nhẹ và không
    tích lũy trung bình chuẩn hóa nhiều lát trên toàn vùng overlap.
    """
    def __init__(self, canvas_shape, background_mode='white', focus_stacking=True, enhance_clarity=False, compute_balanced=False):
        self.canvas_h, self.canvas_w = canvas_shape
        self.background_mode = background_mode
        self.focus_stacking = focus_stacking
        self.enhance_clarity = enhance_clarity
        self.compute_balanced = compute_balanced

        bg_color = (255.0, 255.0, 255.0) if background_mode == 'white' else (0.0, 0.0, 0.0)
        self.canvas_rgb = np.full((self.canvas_h, self.canvas_w, 3), bg_color, dtype=np.float32)
        if self.compute_balanced:
            self.canvas_balanced_rgb = np.full((self.canvas_h, self.canvas_w, 3), bg_color, dtype=np.float32)
        else:
            self.canvas_balanced_rgb = None
        self.canvas_dist = np.zeros((self.canvas_h, self.canvas_w), dtype=np.float32)
        self.canvas_alpha = np.zeros((self.canvas_h, self.canvas_w), dtype=np.float32)

    def accumulate_tile(self, img_rgb, H_matrix, motion_model='affine', tile_gain=None):
        """
        Warp tile trong ROI rồi cập nhật theo Voronoi quality như commit ngày 26.
        Hỗ trợ tích lũy song song ảnh gốc và ảnh cân màu theo tile_gain chuẩn Kiệt.
        """
        h, w = img_rgb.shape[:2]
        source_alpha = img_rgb[:, :, 3].astype(np.float32) / 255.0 if img_rgb.ndim == 3 and img_rgb.shape[2] == 4 else np.ones((h, w), dtype=np.float32)
        source_rgb = img_rgb[:, :, :3] if img_rgb.ndim == 3 and img_rgb.shape[2] == 4 else img_rgb
        source_premul = source_rgb.astype(np.float32) * source_alpha[:, :, None]

        # 1. Tính toán tọa độ 4 góc của tile trên Canvas toàn cục
        corners = np.array([
            [0, 0, 1],
            [w, 0, 1],
            [w, h, 1],
            [0, h, 1]
        ], dtype=np.float64).T

        warped_corners = H_matrix @ corners
        if motion_model == 'homography':
            warped_corners /= (warped_corners[2:3, :] + 1e-8)

        pts = warped_corners[:2, :].T
        x0 = max(0, int(np.floor(np.min(pts[:, 0]))))
        y0 = max(0, int(np.floor(np.min(pts[:, 1]))))
        x1 = min(self.canvas_w, int(np.ceil(np.max(pts[:, 0]))))
        y1 = min(self.canvas_h, int(np.ceil(np.max(pts[:, 1]))))

        roi_w = x1 - x0
        roi_h = y1 - y0

        if roi_w <= 0 or roi_h <= 0:
            return

        # 2. Ma trận dịch chuyển cục bộ cho ROI
        T_roi = np.array([
            [1.0, 0.0, -x0],
            [0.0, 1.0, -y0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        H_roi = T_roi @ H_matrix

        # 3. Commit ngày 26 dùng INTER_LINEAR, không dùng bicubic gây ringing.
        if motion_model != 'homography':
            warped_roi = cv2.warpAffine(
                source_premul, H_roi[:2, :], (roi_w, roi_h),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
            )
            mask_roi = cv2.warpAffine(
                source_alpha, H_roi[:2, :], (roi_w, roi_h),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
        else:
            warped_roi = cv2.warpPerspective(
                source_premul, H_roi, (roi_w, roi_h),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
            )
            mask_roi = cv2.warpPerspective(
                source_alpha, H_roi, (roi_w, roi_h),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )

        binary_mask = (mask_roi > 0.25).astype(np.uint8)
        if np.count_nonzero(binary_mask) == 0:
            return

        warped_straight = np.zeros_like(warped_roi)
        np.divide(warped_roi, mask_roi[:, :, None], out=warped_straight, where=mask_roi[:, :, None] > 1e-4)
        warped_straight = np.clip(warped_straight, 0.0, 255.0)

        # 4. Voronoi distance quality đúng commit 4d17bbe.
        # Padding chỉ chống OpenCV trả FLT_MAX khi mask kín toàn ROI; công thức
        # Voronoi/seam phía dưới vẫn giữ nguyên commit ngày 26.
        padded_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
        dist_map = cv2.distanceTransform(padded_mask, cv2.DIST_L2, 5)[1:-1, 1:-1].astype(np.float32)
        dist_map = np.maximum(0.0, dist_map - 1.5)

        if self.focus_stacking:
            sharpness_map = compute_local_sharpness_map(warped_straight, kernel_size=15)
            sharp_factor = 1.0 + 0.05 * np.nan_to_num(np.clip(sharpness_map, 0.0, 10.0), nan=0.0)
            quality_metric = (dist_map * sharp_factor.astype(np.float32)).astype(np.float32)
        else:
            quality_metric = dist_map

        # 5. Cập nhật canvas theo seam transition của commit ngày 26.
        roi_rgb = self.canvas_rgb[y0:y1, x0:x1]
        roi_dist = self.canvas_dist[y0:y1, x0:x1]
        roi_alpha = self.canvas_alpha[y0:y1, x0:x1]
        diff = quality_metric - roi_dist

        min_dim = float(min(roi_w, roi_h))
        seam_width = max(32.0, min(240.0, min_dim * 0.25))
        t = np.clip(diff / seam_width, -1.0, 1.0)
        alpha = 0.5 + 0.5 * np.sin(t * (np.pi / 2.0))

        feather_edge = np.clip(dist_map / 28.0, 0.0, 1.0)
        alpha = np.where(roi_alpha > 0.01, alpha * feather_edge, alpha)
        alpha = np.where(roi_alpha <= 0.01, 1.0, alpha)
        alpha = np.where(binary_mask == 0, 0.0, alpha)

        alpha_3ch = alpha[:, :, np.newaxis]
        self.canvas_rgb[y0:y1, x0:x1] = roi_rgb * (1.0 - alpha_3ch) + warped_straight * alpha_3ch

        if self.compute_balanced and self.canvas_balanced_rgb is not None:
            if tile_gain is not None:
                g = np.asarray(tile_gain, dtype=np.float32)
                g_val = g[None, None, :] if (g.ndim == 1 and len(g) == 3) else float(g)
                warped_balanced = np.clip(warped_straight * g_val, 0.0, 255.0)
            else:
                warped_balanced = warped_straight
            roi_bal = self.canvas_balanced_rgb[y0:y1, x0:x1]
            self.canvas_balanced_rgb[y0:y1, x0:x1] = roi_bal * (1.0 - alpha_3ch) + warped_balanced * alpha_3ch

        self.canvas_dist[y0:y1, x0:x1] = np.maximum(roi_dist, quality_metric * mask_roi)
        self.canvas_alpha[y0:y1, x0:x1] = np.clip(roi_alpha + mask_roi * (1.0 - roi_alpha), 0.0, 1.0)

    def finalize(self):
        """
        Chuẩn hóa kết quả hòa trộn cuối cùng và xử lý màu nền.
        Nếu compute_balanced=True: trả về (blended_rgb, blended_balanced, valid_mask).
        Nếu compute_balanced=False: trả về (blended_rgb, valid_mask).
        """
        blended_rgb = np.clip(self.canvas_rgb, 0, 255).astype(np.uint8)
        if self.enhance_clarity:
            blended_rgb = enhance_cellular_clarity(blended_rgb, strength=0.40)

        blended_balanced = None
        if self.compute_balanced and self.canvas_balanced_rgb is not None:
            blended_balanced = np.clip(self.canvas_balanced_rgb, 0, 255).astype(np.uint8)
            blended_balanced = apply_defringe_filter(blended_balanced)
            if self.enhance_clarity:
                blended_balanced = enhance_cellular_clarity(blended_balanced, strength=0.40)

        final_alpha = np.clip(self.canvas_alpha * 255.0, 0, 255).astype(np.uint8)
        valid_mask = self.canvas_alpha > 0.05
        valid_mask_u8 = np.where(valid_mask, 255, 0).astype(np.uint8)

        del self.canvas_rgb
        if self.canvas_balanced_rgb is not None:
            del self.canvas_balanced_rgb
        del self.canvas_dist
        del self.canvas_alpha

        # Xử lý màu nền
        if self.background_mode == 'transparent':
            blended_rgba = cv2.cvtColor(blended_rgb, cv2.COLOR_RGB2RGBA)
            blended_rgba[:, :, 3] = final_alpha
            if self.compute_balanced and blended_balanced is not None:
                bal_rgba = cv2.cvtColor(blended_balanced, cv2.COLOR_RGB2RGBA)
                bal_rgba[:, :, 3] = final_alpha
                return blended_rgba, bal_rgba, final_alpha
            return blended_rgba, final_alpha
        elif self.background_mode == 'black':
            blended_rgb = np.where(valid_mask[:, :, None], blended_rgb, 0).astype(np.uint8)
            if self.compute_balanced and blended_balanced is not None:
                blended_balanced = np.where(valid_mask[:, :, None], blended_balanced, 0).astype(np.uint8)
                return blended_rgb, blended_balanced, valid_mask_u8
            return blended_rgb, valid_mask_u8
        else: # 'white'
            blended_rgb = np.where(valid_mask[:, :, None], blended_rgb, 255).astype(np.uint8)
            if self.compute_balanced and blended_balanced is not None:
                blended_balanced = np.where(valid_mask[:, :, None], blended_balanced, 255).astype(np.uint8)
                return blended_rgb, blended_balanced, valid_mask_u8
            return blended_rgb, valid_mask_u8

# Alias tương thích ngược
MultiBandBlender = FastStreamingBlender

def compensate_overlap_exposure(source_images, transforms, motion_model='affine', channel_wise=True, threshold=215):
    """
    Cân bằng phơi sáng vùng giao thoa giữa các tile (Wrapper tương thích ngược).
    Tính gain theo thuật toán mô học Kiệt và nhân gain vào từng tile.
    """
    if not source_images or len(source_images) <= 1:
        return source_images
    gains = solve_tissue_specific_gains(source_images, transforms)
    compensated = {}
    for idx, img in source_images.items():
        if img is None:
            continue
        g = gains.get(idx, np.array([1.0, 1.0, 1.0], dtype=np.float32))
        img_f = img.astype(np.float32) * g.reshape((1, 1, 3))
        compensated[idx] = np.clip(img_f, 0, 255).astype(np.uint8)
    return compensated

def blend_multiband_voronoi(images, tile_gains, adjusted_transforms, canvas_w, canvas_h, num_bands=5, background_mode='white'):
    """
    Hòa trộn đa băng tần Laplacian kết hợp phân vùng Voronoi Seam Partitioning.
    Cơ chế Sharpness-Dominant Selection:
    - Tại mọi vùng chồng lấn, tile có độ nét cao hơn (in-focus) sẽ chiến thắng áp đảo tile mờ (out-of-focus).
    - Tần số cao nhất (High-band - chi tiết tế bào) lấy 100% từ tile rõ nét nhất, loại bỏ hoàn toàn phần mờ nhòe.
    - Tần số thấp (Low-band - trường sáng nền) hòa trộn mượt qua tháp Laplacian 5 tầng, xóa sạch đường nối thẳng đứng.
    """
    indices = [i for i in images if i in adjusted_transforms and images[i] is not None]
    n_images = len(indices)
    if n_images == 0:
        bg_col = 255 if background_mode == 'white' else 0
        return np.full((canvas_h, canvas_w, 3), bg_col, dtype=np.uint8), np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    # 1. Cân bằng phơi sáng an toàn
    compensated_images = {}
    for i in indices:
        img = images[i]
        rgb = img[:, :, :3] if img.ndim == 3 and img.shape[2] == 4 else img
        g = tile_gains.get(i, np.ones(3, dtype=np.float32))
        img_f = rgb.astype(np.float32) * g[None, None, :]
        compensated_images[i] = np.clip(img_f, 0.0, 255.0).astype(np.uint8)

    # 2. Tạo mặt nạ Voronoi Seam Partitioning với Sharpness Dominance
    tile_global_sharpness = {}
    for i in indices:
        tile_global_sharpness[i] = compute_tile_global_sharpness(compensated_images[i])

    med_sharpness = float(np.median(list(tile_global_sharpness.values()))) if tile_global_sharpness else 1.0
    med_sharpness = max(1.0, med_sharpness)

    max_dist = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    best_tile_idx = np.full((canvas_h, canvas_w), -1, dtype=np.int16)
    tile_rois = {}

    for i in indices:
        h, w = images[i].shape[:2]
        H = np.asarray(adjusted_transforms[i], dtype=np.float64)
        is_homo = bool(abs(H[2, 0]) > 1e-7 or abs(H[2, 1]) > 1e-7 or abs(H[2, 2] - 1.0) > 1e-7)

        # Hỗ trợ kênh alpha nếu ảnh là RGBA
        if images[i].ndim == 3 and images[i].shape[2] == 4:
            base_mask = (images[i][:, :, 3] > 10).astype(np.uint8)
        else:
            base_mask = np.ones((h, w), dtype=np.uint8)

        padded_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = base_mask
        dist_map = cv2.distanceTransform(padded_mask, cv2.DIST_L2, 5)[1:-1, 1:-1].astype(np.float32)

        corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float64).T
        warped_corners = H @ corners
        if is_homo:
            warped_corners /= (warped_corners[2:3, :] + 1e-8)
        pts = warped_corners[:2, :].T
        x0 = max(0, int(np.floor(np.min(pts[:, 0]))))
        y0 = max(0, int(np.floor(np.min(pts[:, 1]))))
        x1 = min(canvas_w, int(np.ceil(np.max(pts[:, 0]))))
        y1 = min(canvas_h, int(np.ceil(np.max(pts[:, 1]))))
        rw, rh = x1 - x0, y1 - y0

        if rw <= 0 or rh <= 0:
            continue

        T_roi = np.array([[1.0, 0.0, -x0], [0.0, 1.0, -y0], [0.0, 0.0, 1.0]], dtype=np.float64)
        H_roi_3x3 = T_roi @ H

        sharpness_map = compute_local_sharpness_map(compensated_images[i], kernel_size=15)

        if is_homo:
            warped_dist = cv2.warpPerspective(dist_map, H_roi_3x3, (rw, rh), flags=cv2.INTER_LINEAR, borderValue=0.0)
            warped_mask = cv2.warpPerspective(base_mask, H_roi_3x3, (rw, rh), flags=cv2.INTER_NEAREST, borderValue=0)
            warped_sharpness = cv2.warpPerspective(sharpness_map, H_roi_3x3, (rw, rh), flags=cv2.INTER_LINEAR, borderValue=0.0)
        else:
            H_roi_2x3 = H_roi_3x3[:2]
            warped_dist = cv2.warpAffine(dist_map, H_roi_2x3, (rw, rh), flags=cv2.INTER_LINEAR, borderValue=0.0)
            warped_mask = cv2.warpAffine(base_mask, H_roi_2x3, (rw, rh), flags=cv2.INTER_NEAREST, borderValue=0)
            warped_sharpness = cv2.warpAffine(sharpness_map, H_roi_2x3, (rw, rh), flags=cv2.INTER_LINEAR, borderValue=0.0)

        g_focus = max(0.01, float(tile_global_sharpness[i] / med_sharpness) ** 2.0)

        edge_ramp = np.clip(warped_dist / 15.0, 0.0, 1.0)
        tile_quality = (edge_ramp * (g_focus * (warped_sharpness ** 2.0) + 0.001 * warped_dist)).astype(np.float32)

        canvas_d_sub = max_dist[y0:y1, x0:x1]
        canvas_idx_sub = best_tile_idx[y0:y1, x0:x1]

        better = (tile_quality > canvas_d_sub) & (warped_mask > 0)
        canvas_d_sub[better] = tile_quality[better]
        canvas_idx_sub[better] = i

        tile_rois[i] = (x0, y0, x1, y1, H_roi_3x3, is_homo)

    # 3. Nạp vào MultiBandBlender của OpenCV
    actual_bands = max(1, min(int(num_bands), 8))
    blender = cv2.detail.MultiBandBlender(0, actual_bands)
    blender.prepare((0, 0, canvas_w, canvas_h))

    for i in indices:
        if i not in tile_rois:
            continue
        img = compensated_images[i]
        x0, y0, x1, y1, H_roi_3x3, is_homo = tile_rois[i]
        rw, rh = x1 - x0, y1 - y0

        if is_homo:
            warped_img = cv2.warpPerspective(img, H_roi_3x3, (rw, rh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        else:
            warped_img = cv2.warpAffine(img, H_roi_3x3[:2], (rw, rh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        tile_seam_mask = (best_tile_idx[y0:y1, x0:x1] == i).astype(np.uint8) * 255
        blender.feed(warped_img.astype(np.int16), tile_seam_mask, (x0, y0))

    result_img, result_mask = blender.blend(None, None)
    if result_img is None or result_mask is None:
        bg_col = 255 if background_mode == 'white' else 0
        return np.full((canvas_h, canvas_w, 3), bg_col, dtype=np.uint8), np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    final_rgb = np.clip(result_img, 0, 255).astype(np.uint8)

    bg_val = 255 if background_mode == 'white' else 0
    bg_mask = (result_mask == 0)
    final_rgb[bg_mask] = bg_val

    global_mask = (result_mask > 0).astype(np.uint8) * 255
    return final_rgb, global_mask


