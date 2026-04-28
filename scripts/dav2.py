#!/usr/bin/env python3
"""
dav2_v2.py — 1024切片→512 + DAv2生成reldepth
流程：
  1. 读取原始dom/ndsm（1024x1024）
  2. 切成512x512，30%重叠度
  3. 保存到dom_512/和ndsm_512/
  4. 用DAv2对dom_512生成reldepth
  5. 生成新的train.txt/val.txt
"""
import sys
import os
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import torch
sys.path.append('/root/work/depth_anything_v2')

# =====================================================================
#  1. 切片模块：1024 → 512, 30%重叠
# =====================================================================

def tile_image(src_path, tile_size=512, overlap=0.3):
    """
    将大图切成tile
    返回: list of (tile_array, transform, row, col, x_offset, y_offset)
    """
    tiles = []
    stride = int(tile_size * (1 - overlap))  # 512 * 0.7 = 358

    with rasterio.open(src_path) as src:
        height = src.height
        width = src.width
        crs = src.crs
        transform = src.transform
        count = src.count
        dtype = src.dtypes[0]

        row = 0
        while row < height:
            col = 0
            while col < width:
                # 确保tile不超出边界
                h = min(tile_size, height - row)
                w = min(tile_size, width - col)

                # 如果tile太小，调整起始位置
                if h < tile_size and row > 0:
                    row = height - tile_size
                    h = min(tile_size, height - row)
                if w < tile_size and col > 0:
                    col = width - tile_size
                    w = min(tile_size, width - col)

                window = Window(col, row, w, h)
                data = src.read(window=window)

                # 如果大小不够，pad到tile_size
                if data.shape[1] != tile_size or data.shape[2] != tile_size:
                    padded = np.zeros((count, tile_size, tile_size), dtype=dtype)
                    padded[:, :data.shape[1], :data.shape[2]] = data
                    data = padded

                tile_transform = rasterio.windows.transform(window, transform)
                tiles.append({
                    'data': data,
                    'transform': tile_transform,
                    'row': row, 'col': col,
                    'height': h, 'width': w
                })

                col += stride
                if col + tile_size > width and col != width - tile_size:
                    break
            row += stride
            if row + tile_size > height and row != height - tile_size:
                break

    return tiles


def process_city_tiling(city_folder, tile_size=512, overlap=0.3):
    dom_dir = os.path.join(city_folder, 'dom')
    ndsm_dir = os.path.join(city_folder, 'ndsm')
    dom_512_dir = os.path.join(city_folder, 'dom_512')
    ndsm_512_dir = os.path.join(city_folder, 'ndsm_512')

    os.makedirs(dom_512_dir, exist_ok=True)
    os.makedirs(ndsm_512_dir, exist_ok=True)

    # ★ 只处理原始txt，跳过 _512.txt
    txts = [f for f in os.listdir(city_folder)
            if f.endswith('.txt') and '_512' not in f]
    if not txts:
        print(f"  未找到原始txt文件")
        return

    for txt_file in txts:
        txt_path = os.path.join(city_folder, txt_file)
        new_lines = []

        with open(txt_path) as f:
            lines = f.readlines()

        cn = os.path.basename(city_folder.rstrip('/'))
        print(f"\n  切片 {cn}/{txt_file} ({len(lines)} 个原始tile)")

        total_tiles = 0
        for line in tqdm(lines, desc=f"  切片"):
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            dom_abs = parts[0].replace('\\', '/')
            gt_abs = parts[1].replace('\\', '/')

            fn = dom_abs.split('dom/')[-1] if 'dom/' in dom_abs else os.path.basename(dom_abs)
            if not fn.endswith('.tif'):
                fn = fn + '.tif'

            dom_full = os.path.join(dom_dir, fn)
            gt_full = os.path.join(ndsm_dir, fn)

            if not os.path.exists(dom_full) or not os.path.exists(gt_full):
                continue

            try:
                dom_tiles = tile_image(dom_full, tile_size, overlap)
                ndsm_tiles = tile_image(gt_full, tile_size, overlap)

                n_tiles = min(len(dom_tiles), len(ndsm_tiles))

                for i in range(n_tiles):
                    base_name = os.path.splitext(fn)[0]
                    new_fn = f"{base_name}_t{i}.tif"

                    dom_save = os.path.join(dom_512_dir, new_fn)
                    ndsm_save = os.path.join(ndsm_512_dir, new_fn)

                    if os.path.exists(dom_save) and os.path.exists(ndsm_save):
                        new_lines.append(f"dom_512/{new_fn}\tndsm_512/{new_fn}")
                        total_tiles += 1
                        continue

                    dt = dom_tiles[i]
                    dom_profile = {
                        'driver': 'GTiff',
                        'dtype': dt['data'].dtype,
                        'count': dt['data'].shape[0],
                        'height': tile_size,
                        'width': tile_size,
                        'crs': rasterio.open(dom_full).crs,
                        'transform': dt['transform'],
                    }
                    with rasterio.open(dom_save, 'w', **dom_profile) as dst:
                        dst.write(dt['data'])

                    nt = ndsm_tiles[i]
                    ndsm_profile = {
                        'driver': 'GTiff',
                        'dtype': nt['data'].dtype,
                        'count': nt['data'].shape[0],
                        'height': tile_size,
                        'width': tile_size,
                        'crs': rasterio.open(gt_full).crs,
                        'transform': nt['transform'],
                    }
                    with rasterio.open(ndsm_save, 'w', **ndsm_profile) as dst:
                        dst.write(nt['data'])

                    new_lines.append(f"dom_512/{new_fn}\tndsm_512/{new_fn}")
                    total_tiles += 1

            except Exception as e:
                print(f"  切片失败 {fn}: {e}")

        if not new_lines:
            print(f"  {txt_file}: 无切片结果")
            continue

        # ★ train/val划分（80/20）
        np.random.seed(42)
        np.random.shuffle(new_lines)
        split = int(len(new_lines) * 0.8)

        train_lines = new_lines[:split]
        val_lines = new_lines[split:]

        train_txt = os.path.join(city_folder, 'train_512.txt')
        val_txt = os.path.join(city_folder, 'val_512.txt')

        with open(train_txt, 'w') as f:
            f.write('\n'.join(train_lines) + '\n')
        with open(val_txt, 'w') as f:
            f.write('\n'.join(val_lines) + '\n')

        print(f"  切片完成: {total_tiles} 个tile")
        print(f"  train_512.txt: {len(train_lines)}")
        print(f"  val_512.txt: {len(val_lines)}")

# =====================================================================
#  2. DAv2模块：生成reldepth
# =====================================================================

def load_model(ckpt_path, device='cuda'):
    from depth_anything_v2.dpt import DepthAnythingV2
    model = DepthAnythingV2(encoder='vitl')
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def save_geotiff(path, array, profile):
    profile.update(dtype=rasterio.float32, count=1, compress='lzw')
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(array.astype(np.float32), 1)


def generate_reldepth(city_folder, model, device='cuda'):
    """
    对dom_512生成reldepth
    输出到 reldepth_512/
    """
    dom_512_dir = os.path.join(city_folder, 'dom_512')
    reldepth_dir = os.path.join(city_folder, 'reldepth_512')

    os.makedirs(reldepth_dir, exist_ok=True)

    # 找所有512的dom
    dom_files = sorted([f for f in os.listdir(dom_512_dir) if f.endswith('.tif')])

    if not dom_files:
        print(f"  dom_512为空，跳过")
        return

    cn = os.path.basename(city_folder.rstrip('/'))
    print(f"\n  生成reldepth: {cn} ({len(dom_files)} 个tile)")

    skipped = 0
    for fn in tqdm(dom_files, desc=f"  DAv2"):
        save_path = os.path.join(reldepth_dir, fn)
        if os.path.exists(save_path):
            skipped += 1
            continue

        dom_path = os.path.join(dom_512_dir, fn)

        try:
            with rasterio.open(dom_path) as src:
                profile = src.profile
                img_data = src.read([1, 2, 3]).transpose(1, 2, 0)

                if img_data.dtype != np.uint8:
                    img_data = ((img_data - img_data.min()) /
                                (img_data.max() - img_data.min() + 1e-8) * 255).astype(np.uint8)

                with torch.no_grad():
                    patch_depth = model.infer_image(img_data, input_size=518)

                p_min, p_max = patch_depth.min(), patch_depth.max()
                if p_max > p_min:
                    patch_depth = (patch_depth - p_min) / (p_max - p_min)

                save_geotiff(save_path, patch_depth, profile)

        except Exception as e:
            print(f"  失败 {fn}: {e}")

    print(f"  完成 (跳过{skipped}个已有文件)")


# =====================================================================
#  3. 更新txt路径（加入reldepth列）
# =====================================================================

def update_txt_with_reldepth(city_folder):
    """
    给txt文件加入reldepth列
    格式: dom_512/xxx.tif  ndsm_512/xxx.tif  reldepth_512/xxx.tif
    """
    reldepth_dir = os.path.join(city_folder, 'reldepth_512')

    for txt_name in ['train_512.txt', 'val_512.txt']:
        txt_path = os.path.join(city_folder, txt_name)
        if not os.path.exists(txt_path):
            continue

        new_lines = []
        with open(txt_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                dom_rel = parts[0]
                ndsm_rel = parts[1]

                # 从dom路径提取文件名
                fn = os.path.basename(dom_rel)
                rd_rel = f"reldepth_512/{fn}"

                new_lines.append(f"{dom_rel}\t{ndsm_rel}\t{rd_rel}")

        with open(txt_path, 'w') as f:
            f.write('\n'.join(new_lines) + '\n')

        print(f"  更新 {txt_name}: {len(new_lines)} 行 (含reldepth)")


# =====================================================================
#  主函数
# =====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--city_roots', nargs='+', required=True,
                        help='城市数据目录，如 /root/autodl-tmp/train/NYC')
    parser.add_argument('--ckpt', type=str,
                        default='/root/autodl-tmp/checkpoints/depth_anything_v2_vitl.pth')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--skip_tiling', action='store_true',
                        help='跳过切片，直接生成reldepth')
    parser.add_argument('--tile_size', type=int, default=512)
    parser.add_argument('--overlap', type=float, default=0.3)
    args = parser.parse_args()

    city_roots = args.city_roots

    # Step 1: 切片
    if not args.skip_tiling:
        print("=" * 60)
        print("Step 1: 切片 1024 → 512")
        print("=" * 60)
        for city in city_roots:
            print(f"\n处理: {city}")
            process_city_tiling(city, args.tile_size, args.overlap)

    # Step 2: 生成reldepth
    print("\n" + "=" * 60)
    print("Step 2: DAv2生成reldepth")
    print("=" * 60)
    model = load_model(args.ckpt, args.device)

    for city in city_roots:
        generate_reldepth(city, model, args.device)

    # Step 3: 更新txt
    print("\n" + "=" * 60)
    print("Step 3: 更新txt文件（加入reldepth列）")
    print("=" * 60)
    for city in city_roots:
        update_txt_with_reldepth(city)

    print("\n全部完成！")
    print("后续训练用 train_512.txt / val_512.txt")


if __name__ == "__main__":
    main()
