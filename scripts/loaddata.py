"""
loaddata.py - 多城市nDSM数据加载器
数据目录结构:
  /root/autodl-tmp/train/
  ├── LA/
  │   ├── dom_512/        (RGB 3ch)
  │   ├── ndsm_512/       (GT nDSM 1ch)
  │   ├── reldepth_512/   (相对深度 1ch)
  │   ├── train_512.txt   (训练列表)
  │   └── val_512.txt     (验证列表)
  ├── NYC/
  │   ├── dom_512/
  │   ├── ndsm_512/
  │   ├── reldepth_512/
  │   ├── train_512.txt
  │   └── val_512.txt
  ├── lagpkg              (LA建筑矢量)
  └── nycgpkg             (NYC建筑矢量)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
from rasterio.features import geometry_mask
import fiona
from shapely.geometry import shape
import warnings
warnings.filterwarnings('ignore')


def read_tif(path):
    """读取单波段或多波段tif，返回numpy array (C,H,W)"""
    with rasterio.open(path) as src:
        data = src.read()  # (C, H, W)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
    return data.astype(np.float32), transform, crs, nodata


def load_building_mask_from_gpkg(gpkg_path, transform, crs, height, width):
    """
    从GPKG文件生成建筑二值mask
    Args:
        gpkg_path: GPKG文件路径
        transform: rasterio transform（从对应的tif获取）
        crs: 坐标参考系统
        height, width: 目标mask的高宽
    Returns:
        mask: np.ndarray (H, W), 1=建筑, 0=背景
    """
    if not os.path.exists(gpkg_path):
        print(f"警告: GPKG文件不存在 {gpkg_path}，使用全1 mask")
        return np.ones((height, width), dtype=np.float32)

    mask = np.zeros((height, width), dtype=np.uint8)

    try:
        with fiona.open(gpkg_path, 'r') as src:
            if src.crs is None:
                print(f"警告: GPKG无CRS信息，跳过mask生成")
                return np.ones((height, width), dtype=np.float32)

            # 将GPKG中的几何体栅格化到tile的坐标系
            shapes = []
            for feature in src:
                geom = feature['geometry']
                if geom is not None:
                    shapes.append(shape(geom))

            if len(shapes) == 0:
                print(f"警告: GPKG中无几何体，使用全1 mask")
                return np.ones((height, width), dtype=np.float32)

            # 使用rasterio栅格化
            burned = geometry_mask(
                shapes,
                out_shape=(height, width),
                transform=transform,
                invert=True  # True=建筑区域为1
            )
            mask = burned.astype(np.uint8)

    except Exception as e:
        print(f"警告: 读取GPKG失败 ({e})，使用全1 mask")
        return np.ones((height, width), dtype=np.float32)

    return mask.astype(np.float32)


def generate_mask_from_ndsm(ndsm_data, building_threshold=1.0):
    """
    备用方案：从nDSM生成建筑mask
    nDSM中高于阈值的区域视为建筑
    Args:
        ndsm_data: np.ndarray (1, H, W) 或 (H, W)
        building_threshold: 高度阈值(米)
    Returns:
        mask: np.ndarray (H, W)
    """
    if ndsm_data.ndim == 3:
        ndsm_2d = ndsm_data[0]
    else:
        ndsm_2d = ndsm_data
    mask = (ndsm_2d > building_threshold).astype(np.float32)
    return mask


def parse_txt_line(line):
    """
    解析txt文件中的一行
    支持格式:
      - "dom_path\tndsm_path\treldepth_path" (tab分隔)
      - "dom_path ndsm_path reldepth_path"   (空格分隔)
      - "dom_path,ndsm_path,reldepth_path"   (逗号分隔)
    """
    line = line.strip()
    if not line:
        return None

    # 尝试tab分隔
    if '\t' in line:
        parts = line.split('\t')
    # 尝试逗号分隔
    elif ',' in line:
        parts = line.split(',')
    # 尝试空格分隔（连续空格合并）
    else:
        parts = line.split()

    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) == 3:
        return parts[0], parts[1], parts[2]  # dom, ndsm, reldepth
    elif len(parts) == 2:
        # 只有两列：reldepth 和 ndsm（dom用同名文件推断）
        return None, parts[1], parts[0]
    else:
        return None


class CityHeightDataset(Dataset):
    """
    单城市nDSM数据集
    读取txt列表，加载 dom(RGB), ndsm(GT), reldepth 三类tif
    """
    def __init__(self, city_root, txt_file, gpkg_path=None, building_threshold=1.0,
                 use_gpkg_mask=True, height_normalize=True):
        """
        Args:
            city_root: 城市数据根目录 e.g. /root/autodl-tmp/train/NYC
            txt_file: train_512.txt 或 val_512.txt
            gpkg_path: GPKG文件路径（可选）
            building_threshold: 从nDSM生成mask的高度阈值
            use_gpkg_mask: 是否优先使用GPKG生成mask
            height_normalize: 是否对nDSM做z-score归一化
        """
        self.city_root = city_root
        self.gpkg_path = gpkg_path
        self.building_threshold = building_threshold
        self.use_gpkg_mask = use_gpkg_mask
        self.height_normalize = height_normalize

        self.samples = []  # [(dom_path, ndsm_path, reldepth_path), ...]

        txt_full = os.path.join(city_root, txt_file)
        if not os.path.exists(txt_full):
            print(f"警告: 列表文件不存在 {txt_full}")
            return

        with open(txt_full, 'r', encoding='utf-8') as f:
            for line in f:
                result = parse_txt_line(line)
                if result is None:
                    continue
                dom_path, ndsm_path, reldepth_path = result

                # 拼接完整路径
                dom_full = self._resolve_path(dom_path)
                ndsm_full = self._resolve_path(ndsm_path)
                reldepth_full = self._resolve_path(reldepth_path)

                # 验证文件存在
                if os.path.exists(dom_full) and os.path.exists(ndsm_full) and os.path.exists(reldepth_full):
                    self.samples.append((dom_full, ndsm_full, reldepth_full))
                else:
                    missing = []
                    if not os.path.exists(dom_full): missing.append(f"dom:{dom_full}")
                    if not os.path.exists(ndsm_full): missing.append(f"ndsm:{ndsm_full}")
                    if not os.path.exists(reldepth_full): missing.append(f"reldepth:{reldepth_full}")
                    # 静默跳过缺失文件（不打印，避免刷屏）

        print(f"  {os.path.basename(city_root)}: 加载 {len(self.samples)} 个有效样本 (来自 {txt_file})")

    def _resolve_path(self, path):
        """
        解析路径：如果是相对路径则拼接city_root
        支持 "dom_512/xxx.tif" 和 "/absolute/path/xxx.tif" 两种格式
        """
        path = path.strip()
        if os.path.isabs(path):
            return path
        return os.path.join(self.city_root, path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dom_path, ndsm_path, reldepth_path = self.samples[idx]
    
        dom_data, transform, crs, _ = read_tif(dom_path)
        if dom_data.shape[0] > 3:
            dom_data = dom_data[:3]
        elif dom_data.shape[0] == 1:
            dom_data = np.repeat(dom_data, 3, axis=0)
    
        ndsm_data, _, _, ndsm_nodata = read_tif(ndsm_path)
        if ndsm_data.ndim == 3:
            ndsm_data = ndsm_data[:1]
        else:
            ndsm_data = ndsm_data[np.newaxis, :, :]
    
        H, W = ndsm_data.shape[1], ndsm_data.shape[2]
    
        reldepth_data, _, _, _ = read_tif(reldepth_path)
        if reldepth_data.ndim == 3:
            reldepth_data = reldepth_data[:1]
        else:
            reldepth_data = reldepth_data[np.newaxis, :, :]
    
        # 尺寸对齐
        if dom_data.shape[1] != H or dom_data.shape[2] != W:
            from scipy.ndimage import zoom
            dom_data = zoom(dom_data, (1, H / dom_data.shape[1], W / dom_data.shape[2]), order=1)
        if reldepth_data.shape[1] != H or reldepth_data.shape[2] != W:
            from scipy.ndimage import zoom
            reldepth_data = zoom(reldepth_data, (1, H / reldepth_data.shape[1], W / reldepth_data.shape[2]), order=1)
    
        # ===== 关键修复：彻底清洗数据 =====
        ndsm_2d = ndsm_data[0].copy()
    
        # 清洗nDSM
        if ndsm_nodata is not None:
            ndsm_data[ndsm_data == ndsm_nodata] = 0.0
        ndsm_data = np.nan_to_num(ndsm_data, nan=0.0, posinf=0.0, neginf=0.0)
        ndsm_data = np.clip(ndsm_data, 0.0, 500.0)  # nDSM合理范围
        ndsm_2d = ndsm_data[0]
    
        # 清洗reldepth
        reldepth_data = np.nan_to_num(reldepth_data, nan=0.0, posinf=0.0, neginf=0.0)
        reldepth_data = np.clip(reldepth_data, -500.0, 500.0)
    
        # 清洗RGB
        dom_data = np.nan_to_num(dom_data, nan=0.0, posinf=1.0, neginf=0.0)
    
        # 生成mask
        mask = np.ones((H, W), dtype=np.float32)
        mask[np.isnan(ndsm_2d)] = 0
        mask[np.isinf(ndsm_2d)] = 0
        mask[ndsm_2d <= 0] = 0  # 非正高度视为无效
    
        # GPKG mask
        if self.use_gpkg_mask and self.gpkg_path and os.path.exists(self.gpkg_path):
            try:
                gpkg_mask = load_building_mask_from_gpkg(self.gpkg_path, transform, crs, H, W)
                mask = mask * gpkg_mask
            except Exception:
                threshold_mask = generate_mask_from_ndsm(ndsm_data, self.building_threshold)
                mask = mask * threshold_mask
        else:
            threshold_mask = generate_mask_from_ndsm(ndsm_data, self.building_threshold)
            mask = mask * threshold_mask
    
        # RGB归一化
        if dom_data.max() > 1.0:
            dom_data = dom_data / 255.0
        dom_data = np.clip(dom_data, 0.0, 1.0)
    
        # 归一化nDSM和reldepth
        height_mean, height_std = 0.0, 1.0
        valid_mask = mask > 0
        if valid_mask.sum() > 10:
            valid_ndsm = ndsm_2d[valid_mask]
            height_mean = float(np.mean(valid_ndsm))
            height_std = float(np.std(valid_ndsm))
            if height_std < 1e-3:
                height_std = 1.0  # 防止除零
            ndsm_data = (ndsm_data - height_mean) / height_std
    
            valid_reldepth = reldepth_data[0][valid_mask]
            rd_mean = float(np.mean(valid_reldepth))
            rd_std = float(np.std(valid_reldepth))
            if rd_std < 1e-3:
                rd_std = 1.0
            reldepth_data = (reldepth_data - rd_mean) / rd_std
    
        # 最终安全检查：确保输出无异常值
        ndsm_data = np.clip(ndsm_data, -10.0, 10.0)
        reldepth_data = np.clip(reldepth_data, -10.0, 10.0)
    
        sample = {
            'image': torch.from_numpy(dom_data.copy()).float(),
            'depth': torch.from_numpy(ndsm_data.copy()).float(),
            'rel_depth': torch.from_numpy(reldepth_data.copy()).float(),
            'mask': torch.from_numpy(mask.copy()).float(),
            'args_height': (height_mean, height_std),
        }
        return sample


class MultiCityDataset(Dataset):
    """
    多城市合并数据集
    将LA和NYC的数据合并为一个统一的Dataset
    """
    def __init__(self, city_configs, txt_type='train', building_threshold=1.0,
                 use_gpkg_mask=True, height_normalize=True):
        """
        Args:
            city_configs: list of dict, 每个dict包含:
                {
                    'root': '/root/autodl-tmp/train/NYC',
                    'gpkg': '/root/autodl-tmp/train/nycgpkg',
                    'txt_prefix': 'train'  或 'val'
                }
            txt_type: 'train' 或 'val'，决定读取哪个txt
            building_threshold: nDSM高度阈值
            use_gpkg_mask: 是否使用GPKG
            height_normalize: 是否归一化
        """
        self.datasets = []
        self.cumulative_lengths = []
        total = 0

        for cfg in city_configs:
            city_root = cfg['root']
            gpkg_path = cfg.get('gpkg', None)
            city_name = os.path.basename(city_root)

            # 自动查找txt文件
            txt_file = f"{txt_type}_512.txt"
            txt_full = os.path.join(city_root, txt_file)
            if not os.path.exists(txt_full):
                # 尝试其他可能的命名
                for candidate in [f"{txt_type}.txt", f"{city_name}_{txt_type}_512.txt"]:
                    if os.path.exists(os.path.join(city_root, candidate)):
                        txt_file = candidate
                        break

            ds = CityHeightDataset(
                city_root=city_root,
                txt_file=txt_file,
                gpkg_path=gpkg_path,
                building_threshold=building_threshold,
                use_gpkg_mask=use_gpkg_mask,
                height_normalize=height_normalize
            )
            self.datasets.append(ds)
            total += len(ds)
            self.cumulative_lengths.append(total)

        print(f"  合并数据集 ({txt_type}): 共 {total} 个样本")

    def __len__(self):
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def __getitem__(self, idx):
        # 确定idx属于哪个子数据集
        dataset_idx = 0
        for i, cum_len in enumerate(self.cumulative_lengths):
            if idx < cum_len:
                dataset_idx = i
                break

        if dataset_idx > 0:
            local_idx = idx - self.cumulative_lengths[dataset_idx - 1]
        else:
            local_idx = idx

        return self.datasets[dataset_idx][local_idx]


def collate_fn(batch):
    """自定义collate，处理mask维度不一致的情况"""
    result = {}
    for key in batch[0]:
        if key in ['dom_path', 'ndsm_path', 'reldepth_path', 'args_height']:
            result[key] = [b[key] for b in batch]
        elif key == 'mask':
            masks = [b[key] for b in batch]
            # 确保mask是(H, W)格式
            masks = [m if m.ndim == 2 else m.squeeze(0) for m in masks]
            result[key] = torch.stack(masks, dim=0)  # (B, H, W)
        else:
            tensors = [b[key] for b in batch]
            result[key] = torch.stack(tensors, dim=0)
    return result


def getTrainingData(batch_size, train_pred=None, train_depth=None, train_gt=None,
                    city_roots=None, gpkg_paths=None, num_workers=4):
    """
    创建训练集DataLoader
    两种调用方式:
      1. 旧接口: getTrainingData(batch_size, train_pred, train_depth, train_gt)
         直接传入路径列表
      2. 新接口: getTrainingData(batch_size, city_roots=[...], gpkg_paths=[...])
         从城市目录自动加载
    """
    if city_roots is not None:
        # 新接口：从城市目录加载
        city_configs = []
        for root, gpkg in zip(city_roots, gpkg_paths or [None] * len(city_roots)):
            city_configs.append({
                'root': root,
                'gpkg': gpkg,
            })
        dataset = MultiCityDataset(city_configs, txt_type='train',
                                   building_threshold=10.0,
                                   use_gpkg_mask=True,
                                   height_normalize=True)
    elif train_pred is not None and train_gt is not None:
        # 旧接口：直接传路径列表（兼容原代码）
        dataset = SimpleListDataset(train_pred, train_depth, train_gt)
    else:
        raise ValueError("必须提供 city_roots 或 (train_pred, train_gt)")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=True if num_workers > 0 else False
    )
    return loader


def getTestingData(batch_size, val_pred=None, val_depth=None, val_gt=None,
                   city_roots=None, gpkg_paths=None, num_workers=4):
    """创建验证集DataLoader，接口同getTrainingData"""
    if city_roots is not None:
        city_configs = []
        for root, gpkg in zip(city_roots, gpkg_paths or [None] * len(city_roots)):
            city_configs.append({
                'root': root,
                'gpkg': gpkg,
            })
        dataset = MultiCityDataset(city_configs, txt_type='val',
                                   building_threshold=10.0,
                                   use_gpkg_mask=True,
                                   height_normalize=True)
    elif val_pred is not None and val_gt is not None:
        dataset = SimpleListDataset(val_pred, val_depth, val_gt)
    else:
        raise ValueError("必须提供 city_roots 或 (val_pred, val_gt)")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=True if num_workers > 0 else False
    )
    return loader


class SimpleListDataset(Dataset):
    """旧接口兼容：直接从路径列表加载"""
    def __init__(self, pred_list, depth_list, gt_list):
        self.pred_list = pred_list
        self.depth_list = depth_list if depth_list is not None else pred_list
        self.gt_list = gt_list

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):
        image_data, transform, crs, _ = read_tif(self.pred_list[idx])
        depth_data, _, _, _ = read_tif(self.depth_list[idx])
        gt_data, _, _, ndsm_nodata = read_tif(self.gt_list[idx])

        if image_data.shape[0] > 3:
            image_data = image_data[:3]
        elif image_data.shape[0] == 1:
            image_data = np.repeat(image_data, 3, axis=0)

        if depth_data.ndim == 3:
            depth_data = depth_data[:1]
        else:
            depth_data = depth_data[np.newaxis, :, :]

        if gt_data.ndim == 3:
            gt_data = gt_data[:1]
        else:
            gt_data = gt_data[np.newaxis, :, :]

        gt_2d = gt_data[0]
        mask = np.ones_like(gt_2d, dtype=np.float32)
        if ndsm_nodata is not None:
            mask[gt_2d == ndsm_nodata] = 0
        mask[np.isnan(gt_2d)] = 0

        gt_data = np.nan_to_num(gt_data, nan=0.0, posinf=0.0, neginf=0.0)
        depth_data = np.nan_to_num(depth_data, nan=0.0, posinf=0.0, neginf=0.0)

        if image_data.max() > 1.0:
            image_data = image_data / 255.0
        image_data = np.clip(image_data, 0.0, 1.0)

        height_mean, height_std = 0.0, 1.0
        valid = mask > 0
        if valid.sum() > 0:
            height_mean = float(gt_2d[valid].mean())
            height_std = float(gt_2d[valid].std()) + 1e-6
            gt_data = (gt_data - height_mean) / height_std

        return {
            'image': torch.from_numpy(image_data.copy()).float(),
            'depth': torch.from_numpy(gt_data.copy()).float(),
            'rel_depth': torch.from_numpy(depth_data.copy()).float(),
            'mask': torch.from_numpy(mask.copy()).float(),
            'args_height': (height_mean, height_std),
            'dom_path': self.pred_list[idx],
            'ndsm_path': self.gt_list[idx],
            'reldepth_path': self.depth_list[idx],
        }
