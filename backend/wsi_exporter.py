# Feature: Exports high-resolution mosaics and generates DeepZoom pyramids (DZI)
# Purpose: Bảo toàn đối xứng định dạng và phục vụ xem ảnh WSI khổng lồ mượt mà trên trình duyệt
# Path: tool/image_alignment/backend/wsi_exporter.py

import os
import math
import mmap
import cv2
import numpy as np
import tifffile
import tempfile
import shutil
import json
import contextlib
import errno
import threading
import time
from PIL import Image

_PUBLICATION_THREAD_LOCK = threading.RLock()


@contextlib.contextmanager
def _publication_lock(output_dir, timeout=30.0):
    """Prevent same-directory artifact sets from interleaving across processes."""
    lock_path = os.path.join(output_dir, ".image_alignment.publish.lock")
    lock_file = open(lock_path, "a+b")
    acquired = False
    try:
        lock_file.seek(0, os.SEEK_END)
        if lock_file.tell() == 0:
            lock_file.write(b"0")
            lock_file.flush()
        lock_file.seek(0)
        if os.name == "nt":
            import msvcrt
            deadline = time.monotonic() + timeout
            while True:
                try:
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                    acquired = True
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EDEADLK, errno.EAGAIN) or time.monotonic() >= deadline:
                        raise TimeoutError("Unable to lock publication directory") from exc
                    time.sleep(0.05)
        else:
            import fcntl
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            acquired = True
        yield
    finally:
        try:
            if acquired:
                lock_file.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()
            try:
                os.remove(lock_path)
            except Exception:
                pass

def export_wsi_image(file_path, image_data, target_format=None, quality=95):
    """
    Xuất ảnh kết quả WSI.
    - file_path: Đường dẫn lưu trữ (ví dụ: output/result.tif)
    - image_data: numpy array (H, W, 3) hoặc (H, W, 4) uint8
    - target_format: 'tif', 'tiff', 'png', 'jpg', 'jpeg', 'bmp' hoặc None (tự suy ra từ file_path)
    """
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    if target_format is None:
        target_format = os.path.splitext(file_path)[1].lstrip('.').lower()
    else:
        target_format = target_format.lower()

    if target_format in ['tif', 'tiff']:
        is_rgba = (image_data.ndim == 3 and image_data.shape[2] == 4)
        photometric = 'rgb'
        use_bigtiff = (image_data.nbytes > 2 * 1024 * 1024 * 1024)
        try:
            tifffile.imwrite(
                file_path,
                image_data,
                photometric=photometric,
                bigtiff=use_bigtiff,
                extrasamples=['unassalpha'] if is_rgba else None
            )
        except Exception:
            try:
                pil_img = Image.fromarray(image_data)
                pil_img.save(file_path, format='TIFF')
            except Exception:
                if is_rgba:
                    bgr = cv2.cvtColor(image_data, cv2.COLOR_RGBA2BGRA)
                else:
                    bgr = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
                cv2.imwrite(file_path, bgr)
    elif target_format in ['png']:
        # OpenCV nhận BGR hoặc BGRA
        if image_data.ndim == 3 and image_data.shape[2] == 4:
            bgr = cv2.cvtColor(image_data, cv2.COLOR_RGBA2BGRA)
        else:
            bgr = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_path, bgr, [cv2.IMWRITE_PNG_COMPRESSION, 4])
    elif target_format in ['jpg', 'jpeg']:
        if image_data.ndim == 3 and image_data.shape[2] == 4:
            # Drop alpha cho JPEG
            rgb = image_data[:, :, :3]
        else:
            rgb = image_data
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_path, bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    elif target_format in ['bmp']:
        bgr = cv2.cvtColor(image_data[:, :, :3], cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_path, bgr)
    else:
        # Fallback PIL
        pil_img = Image.fromarray(image_data)
        pil_img.save(file_path)

    return file_path

def _downsample_rgba(image_data, size):
    alpha = image_data[:, :, 3:4].astype(np.float32) / 255.0
    premul = image_data[:, :, :3].astype(np.float32) * alpha
    alpha_small = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)
    if alpha_small.ndim == 2:
        alpha_small = alpha_small[:, :, None]
    premul_small = cv2.resize(premul, size, interpolation=cv2.INTER_AREA)
    rgb_small = np.zeros_like(premul_small)
    np.divide(premul_small, alpha_small, out=rgb_small, where=alpha_small > 1e-8)
    return np.dstack((np.clip(rgb_small, 0, 255), np.clip(alpha_small[:, :, 0] * 255.0, 0, 255))).astype(np.uint8)


def _publish_replace_dir(src, dst):
    """Thay thế thư mục đích an toàn trên mọi hệ điều hành (tránh WinError 5 trên Windows NTFS)."""
    import gc
    import time

    if not os.path.exists(src):
        return

    # Nếu dst đã tồn tại, dọn dẹp hoặc dời dst trước (trên Windows os.replace cấm replace directory)
    if os.path.exists(dst):
        if os.path.isdir(dst):
            temp_trash = f"{dst}.old.{time.time_ns()}"
            moved_old = False
            for attempt in range(5):
                try:
                    os.rename(dst, temp_trash)
                    moved_old = True
                    break
                except (PermissionError, OSError):
                    gc.collect()
                    time.sleep(0.05 * (attempt + 1))
            if moved_old:
                shutil.rmtree(temp_trash, ignore_errors=True)
            else:
                # Nếu không thể rename dst (do file đang được server đọc), copy đè trực tiếp
                shutil.copytree(src, dst, dirs_exist_ok=True)
                shutil.rmtree(src, ignore_errors=True)
                return
        else:
            try:
                os.remove(dst)
            except OSError:
                pass

    # dst không còn tồn tại, thử rename src sang dst
    for attempt in range(5):
        try:
            os.rename(src, dst)
            return
        except (PermissionError, OSError):
            gc.collect()
            time.sleep(0.05 * (attempt + 1))

    # Fallback cuối cùng nếu Windows vẫn khóa: copytree đè và dọn dẹp src
    shutil.copytree(src, dst, dirs_exist_ok=True)
    shutil.rmtree(src, ignore_errors=True)


def _publish_replace(src, dst):
    if os.path.isdir(src) or (os.path.exists(dst) and os.path.isdir(dst)):
        _publish_replace_dir(src, dst)
    else:
        os.replace(src, dst)


def _publish_staged_artifacts(stage_root, publications):
    backups = []
    published = []
    output_dir = os.path.dirname(os.path.abspath(publications[0][1]))
    with _PUBLICATION_THREAD_LOCK, _publication_lock(output_dir):
        try:
            for staged, destination in publications:
                backup = None
                if os.path.exists(destination):
                    backup = os.path.join(stage_root, f"backup_{len(backups)}")
                    _publish_replace(destination, backup)
                backups.append((backup, destination))
                _publish_replace(staged, destination)
                published.append(destination)
        except Exception:
            for destination in reversed(published):
                if os.path.isdir(destination):
                    shutil.rmtree(destination, ignore_errors=True)
                elif os.path.exists(destination):
                    try:
                        os.remove(destination)
                    except OSError:
                        pass
            for backup, destination in reversed(backups):
                if backup and os.path.exists(backup):
                    try:
                        _publish_replace(backup, destination)
                    except Exception:
                        pass
            raise


def _reclaim_memmap_pages(array):
    """Flush/reclaim resident mapped pages when the operating system supports it."""
    if isinstance(array, np.memmap):
        array.flush()
        if hasattr(array._mmap, 'madvise') and hasattr(mmap, 'MADV_DONTNEED'):
            array._mmap.madvise(mmap.MADV_DONTNEED)


def write_tiled_tiff(array, file_path, tile_size=512, is_mask=False, release_pages=False):
    """Stream tiles preserving dtype, including uint32 source-ownership maps."""
    height, width = array.shape[:2]
    channels = 1 if array.ndim == 2 else array.shape[2]
    tile_size = max(16, int(math.ceil(tile_size / 16.0)) * 16)

    def tiles():
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                tile_shape = (tile_size, tile_size) if channels == 1 else (tile_size, tile_size, channels)
                tile = np.zeros(tile_shape, dtype=array.dtype)
                source = array[y:min(y + tile_size, height), x:min(x + tile_size, width)]
                tile[:source.shape[0], :source.shape[1]] = source
                yield tile
            if release_pages:
                _reclaim_memmap_pages(array)

    kwargs = {
        "shape": (height, width) if channels == 1 else (height, width, channels),
        "dtype": array.dtype,
        "tile": (tile_size, tile_size),
        "photometric": 'minisblack' if is_mask else 'rgb',
        "metadata": None,
    }
    if channels == 4:
        kwargs["extrasamples"] = ['unassalpha']
    try:
        with tifffile.TiffWriter(file_path, bigtiff=True) as writer:
            writer.write(tiles(), **kwargs)
    finally:
        # TIFF writers may consume exactly N tiles without exhausting the
        # generator, so reclamation after its final yield is not sufficient.
        if release_pages:
            _reclaim_memmap_pages(array)


def generate_dzi_pyramid_bounded(
    image_data,
    output_dzi_path,
    work_dir,
    tile_size=254,
    tile_overlap=1,
    chunk_size=512,
):
    """Generate DZI while keeping only one tile/downsample chunk resident in memory."""
    if tile_size < 1 or tile_size > 4096 or tile_overlap < 0 or tile_overlap > 16:
        raise ValueError("Thông số DZI is invalid")
    height, width = image_data.shape[:2]
    channels = image_data.shape[2] if image_data.ndim == 3 else 1
    tile_format = 'png' if channels == 4 else 'jpg'
    base_name = os.path.splitext(os.path.basename(output_dzi_path))[0]
    out_dir = os.path.dirname(os.path.abspath(output_dzi_path))
    tiles_dir = os.path.join(out_dir, f"{base_name}_files")
    os.makedirs(tiles_dir, exist_ok=True)
    max_level = int(math.ceil(math.log2(max(width, height))))
    dzi_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Image xmlns="http://schemas.microsoft.com/deepzoom/2008" Format="{tile_format}" Overlap="{tile_overlap}" TileSize="{tile_size}">
    <Size Width="{width}" Height="{height}"/>
</Image>'''
    with open(output_dzi_path, 'w', encoding='utf-8') as stream:
        stream.write(dzi_xml)

    current = image_data
    owned_path = None
    pending = None
    try:
        for level in range(max_level, -1, -1):
            cur_h, cur_w = current.shape[:2]
            level_dir = os.path.join(tiles_dir, str(level))
            os.makedirs(level_dir, exist_ok=True)
            cols = int(math.ceil(cur_w / tile_size))
            rows = int(math.ceil(cur_h / tile_size))
            for row in range(rows):
                for col in range(cols):
                    x1 = max(0, col * tile_size - (tile_overlap if col else 0))
                    y1 = max(0, row * tile_size - (tile_overlap if row else 0))
                    x2 = min(cur_w, (col + 1) * tile_size + (tile_overlap if col < cols - 1 else 0))
                    y2 = min(cur_h, (row + 1) * tile_size + (tile_overlap if row < rows - 1 else 0))
                    tile = np.asarray(current[y1:y2, x1:x2])
                    tile_path = os.path.join(level_dir, f"{col}_{row}.{tile_format}")
                    encoded = cv2.cvtColor(tile, cv2.COLOR_RGBA2BGRA if channels == 4 else cv2.COLOR_RGB2BGR)
                    params = [cv2.IMWRITE_PNG_COMPRESSION, 4] if channels == 4 else [cv2.IMWRITE_JPEG_QUALITY, 88]
                    if not cv2.imwrite(tile_path, encoded, params):
                        raise RuntimeError("Không thể ghi DZI tile")
            if cur_w == 1 and cur_h == 1:
                break

            next_w = max(1, int(math.ceil(cur_w / 2)))
            next_h = max(1, int(math.ceil(cur_h / 2)))
            next_path = os.path.join(work_dir, f"dzi_level_{level - 1}.dat")
            next_shape = (next_h, next_w, channels)
            next_level = np.memmap(next_path, mode='w+', dtype=np.uint8, shape=next_shape)
            pending = next_level
            for out_y in range(0, next_h, chunk_size):
                for out_x in range(0, next_w, chunk_size):
                    out_y2 = min(next_h, out_y + chunk_size)
                    out_x2 = min(next_w, out_x + chunk_size)
                    source = np.asarray(current[out_y * 2:min(cur_h, out_y2 * 2), out_x * 2:min(cur_w, out_x2 * 2)])
                    size = (out_x2 - out_x, out_y2 - out_y)
                    reduced = _downsample_rgba(source, size) if channels == 4 else cv2.resize(source, size, interpolation=cv2.INTER_AREA)
                    next_level[out_y:out_y2, out_x:out_x2] = reduced
            next_level.flush()
            if owned_path:
                current._mmap.close()
                try:
                    os.remove(owned_path)
                except FileNotFoundError:
                    pass
            current = next_level
            owned_path = next_path
            pending = None
    finally:
        if pending is not None and pending is not current:
            pending._mmap.close()
        if owned_path and isinstance(current, np.memmap):
            current._mmap.close()
            try:
                os.remove(owned_path)
            except FileNotFoundError:
                pass
    return output_dzi_path


def generate_dzi_pyramid(image_data, output_dzi_path, tile_size=254, tile_overlap=1, tile_format='jpg'):
    """
    Tạo cấu trúc DeepZoom Image (.dzi) và thư mục tiles pyramid để xem bằng OpenSeadragon.
    - output_dzi_path: ví dụ 'aligned/stitched_dzi/wsi.dzi'
    """
    out_dir = os.path.dirname(os.path.abspath(output_dzi_path))
    base_name = os.path.splitext(os.path.basename(output_dzi_path))[0]
    tiles_dir = os.path.join(out_dir, f"{base_name}_files")
    os.makedirs(tiles_dir, exist_ok=True)
    if image_data.ndim == 3 and image_data.shape[2] == 4:
        tile_format = 'png'

    h, w = image_data.shape[:2]
    max_dim = max(w, h)
    max_level = int(math.ceil(math.log2(max_dim)))

    # Tạo file descriptor .dzi XML theo chuẩn Microsoft DeepZoom
    dzi_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"
       Format="{tile_format}"
       Overlap="{tile_overlap}"
       TileSize="{tile_size}">
    <Size Width="{w}" Height="{h}"/>
</Image>"""
    with open(output_dzi_path, 'w', encoding='utf-8') as f:
        f.write(dzi_xml)

    # Sinh từng level pyramid từ max_level xuống 0
    current_img = image_data
    for level in range(max_level, -1, -1):
        level_dir = os.path.join(tiles_dir, str(level))
        os.makedirs(level_dir, exist_ok=True)

        cur_h, cur_w = current_img.shape[:2]
        cols = int(math.ceil(cur_w / tile_size))
        rows = int(math.ceil(cur_h / tile_size))

        for row in range(rows):
            for col in range(cols):
                # Tính vùng cắt có tính đến overlap
                src_x = max(0, col * tile_size - (tile_overlap if col > 0 else 0))
                src_y = max(0, row * tile_size - (tile_overlap if row > 0 else 0))
                src_w = min(cur_w - src_x, tile_size + (tile_overlap * 2 if col > 0 and col < cols - 1 else tile_overlap))
                src_h = min(cur_h - src_y, tile_size + (tile_overlap * 2 if row > 0 and row < rows - 1 else tile_overlap))

                tile_crop = current_img[src_y:src_y+src_h, src_x:src_x+src_w]
                tile_filename = os.path.join(level_dir, f"{col}_{row}.{tile_format}")

                if tile_format == 'jpg':
                    if tile_crop.ndim == 3 and tile_crop.shape[2] == 4:
                        tile_crop = tile_crop[:, :, :3]
                    bgr_tile = cv2.cvtColor(tile_crop, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(tile_filename, bgr_tile, [cv2.IMWRITE_JPEG_QUALITY, 88])
                else:
                    if tile_crop.ndim == 3 and tile_crop.shape[2] == 4:
                        bgr_tile = cv2.cvtColor(tile_crop, cv2.COLOR_RGBA2BGRA)
                    else:
                        bgr_tile = cv2.cvtColor(tile_crop, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(tile_filename, bgr_tile)

        # Thu nhỏ 1/2 cho level tiếp theo
        next_w = max(1, int(math.ceil(cur_w / 2)))
        next_h = max(1, int(math.ceil(cur_h / 2)))
        if cur_w == 1 and cur_h == 1:
            break
        if current_img.ndim == 3 and current_img.shape[2] == 4:
            current_img = _downsample_rgba(current_img, (next_w, next_h))
        else:
            current_img = cv2.resize(current_img, (next_w, next_h), interpolation=cv2.INTER_AREA)

    return output_dzi_path

def export_wsi_multiformat(image_data, output_dir, folder_name="stitched_wsi", target_ext="tif", tile_size=254, valid_mask=None, metadata_dict=None, export_dzi=True, export_mask=True):
    """
    Xuất ảnh WSI theo định dạng và tự động sinh DeepZoom Tiles Pyramid, Companion Training Mask và Metadata JSON.
    Có thể tắt export_dzi và export_mask để chỉ lưu duy nhất file ảnh (ví dụ .tif) mà không tạo thêm thư mục phụ.
    """
    os.makedirs(output_dir, exist_ok=True)
    if not folder_name or os.path.basename(folder_name) != folder_name:
        raise ValueError("Invalid output file name")
    target_ext = target_ext.lower().lstrip('.')
    if target_ext == 'dzi':
        target_ext = 'tif'
    if target_ext not in ('tif', 'tiff', 'png', 'jpg', 'jpeg', 'bmp'):
        raise ValueError("Invalid output format")
    if valid_mask is not None and valid_mask.shape[:2] != image_data.shape[:2]:
        raise ValueError("Companion mask must match output dimensions")
    if target_ext == 'png' and image_data.shape[0] * image_data.shape[1] > 268_435_456:
        raise ValueError("PNG output exceeds pixel threshold; use BigTIFF")

    file_name = f"{folder_name}.{target_ext}"
    output_path = os.path.join(output_dir, file_name)
    mask_path = os.path.join(output_dir, f"{folder_name}_valid_mask.png") if (valid_mask is not None and export_mask) else None
    meta_path = os.path.join(output_dir, f"{folder_name}_metadata.json") if metadata_dict is not None else None
    dzi_dir = os.path.join(output_dir, f"{folder_name}_dzi") if export_dzi else None
    dzi_path = os.path.join(dzi_dir, f"{folder_name}.dzi") if export_dzi else None

    stage_root = tempfile.mkdtemp(prefix=f".{folder_name}.publish.", dir=output_dir)
    staged_output = os.path.join(stage_root, file_name)
    staged_mask = os.path.join(stage_root, os.path.basename(mask_path)) if mask_path else None
    staged_meta = os.path.join(stage_root, os.path.basename(meta_path)) if meta_path else None
    staged_dzi_dir = os.path.join(stage_root, f"{folder_name}_dzi") if export_dzi else None
    staged_dzi_path = os.path.join(staged_dzi_dir, f"{folder_name}.dzi") if export_dzi else None
    try:
        export_wsi_image(staged_output, image_data, target_format=target_ext)
        if staged_mask and not cv2.imwrite(staged_mask, valid_mask, [cv2.IMWRITE_PNG_COMPRESSION, 4]):
            raise RuntimeError("Không thể ghi companion mask")
        if staged_meta:
            with open(staged_meta, 'w', encoding='utf-8') as f:
                json.dump(metadata_dict, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
        if export_dzi:
            os.makedirs(staged_dzi_dir, exist_ok=True)
            tile_fmt = 'png' if image_data.ndim == 3 and image_data.shape[2] == 4 else 'jpg'
            generate_dzi_pyramid(image_data, staged_dzi_path, tile_size=tile_size, tile_format=tile_fmt)

        publications = [(staged_output, output_path)]
        if staged_mask:
            publications.append((staged_mask, mask_path))
        if staged_meta:
            publications.append((staged_meta, meta_path))
        if export_dzi and staged_dzi_dir:
            publications.append((staged_dzi_dir, dzi_dir))
        _publish_staged_artifacts(stage_root, publications)
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)
    
    return {
        "output_path": output_path,
        "file_name": file_name,
        "dzi_path": dzi_path,
        "mask_path": mask_path,
        "metadata_path": meta_path
    }
