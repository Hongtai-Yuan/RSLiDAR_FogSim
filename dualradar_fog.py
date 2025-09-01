#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import shutil
from pathlib import Path
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
from typing import Optional
from fog_simulation import ParameterSet, simulate_fog  # 确保可导入


# --------- 工具：解析“实际生效的 alpha / mor”并打印一次 ---------
def resolve_alpha(args):
    """
    返回 (alpha, source)：
      source in {"alpha","mor","visibility","default"}
    """
    if args.alpha is not None:
        return float(args.alpha), "alpha"
    if args.mor is not None:
        return float(np.log(20.0) / args.mor), "mor"
    if args.visibility is not None:
        return float(3.912 / args.visibility), "visibility"
    return 0.04, "default"


def print_effective_params(args):
    alpha, src = resolve_alpha(args)
    mor = float(np.log(20.0) / alpha) if alpha > 0 else np.inf
    beta0 = args.gamma / np.pi

    print("==== Effective Simulation Parameters (resolved at start) ====")
    print(f"alpha: {alpha:.6g}   (source={src})")
    print(f"mor  : {mor:.6g} m   (ln(20)/alpha)")
    print(f"beta0: {beta0:.6g}   (= gamma/pi)")
    print(f"gamma: {args.gamma}")
    print(f"stepsize: {args.stepsize} m, integral: [{args.integral_min}, {args.integral_max}] m, tau_h(ns): {args.tau_h}")
    print(f"noise: {args.noise}, gain: {args.gain}, variant: {args.noise_variant}")
    print(f"intensity_mult: {args.intensity_mult}, round_intensity: {args.round_intensity}")
    print(f"min_dist: {args.min_dist} m, keep_near: {args.keep_near}")
    print(f"apply_range: {args.apply_range}, range: {tuple(args.range)}")
    print("============================================================\n")


# --------- 工具：判断是否需要“原样透传”以及是否为“全 0 输入” ---------
def passthrough_reason(pc: np.ndarray, raw_size: int):
    """
    返回 (need_passthrough: bool, reason_code: str|None)
      reason_code 可能是:
        - "SIZE"            : 元素数不是6的倍数
        - "ZERO_INPUT"      : xyz 全 0 或 r≈0
        - "NAN_INF"         : 含 NaN/Inf
        - "COORD_OUTLIER"   : 坐标异常大（>1e6）
      其中只有 ZERO_INPUT 会在主进程打印提示；其余静默。
    """
    if raw_size % 6 != 0:
        return True, "SIZE"
    if pc.size == 0:
        return True, "ZERO_INPUT"
    if not np.isfinite(pc).all():
        return True, "NAN_INF"

    xyz = pc[:, :3]
    if np.allclose(xyz, 0.0, atol=1e-12):
        return True, "ZERO_INPUT"

    r = np.linalg.norm(xyz, axis=1)
    if np.all(r < 1e-12):
        return True, "ZERO_INPUT"

    if np.nanmax(np.abs(xyz)) > 1e6:
        return True, "COORD_OUTLIER"

    return False, None


# --------- 核心处理函数：仅返回需要打印的“少量提示” ---------
def process_one_file(in_path: Path, out_root: Path, in_root: Path, args) -> Optional[str]:
    try:
        rel = in_path.relative_to(in_root)
        out_path = out_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        raw = np.fromfile(str(in_path), dtype=np.float32)

        # 透传判断
        need_pass, reason = False, None
        if raw.size % 6 == 0:
            pc = raw.reshape((-1, 6))
            need_pass, reason = passthrough_reason(pc, raw.size)
        else:
            pc = None
            need_pass, reason = True, "SIZE"

        if need_pass:
            # 原样复制，不刷屏，只有 ZERO_INPUT 才返回提示
            shutil.copyfile(in_path, out_path)
            if reason == "ZERO_INPUT":
                return f"[ZERO_INPUT] {rel}"
            return None

        # 正常雾仿真
        xyz = pc[:, :3]
        intensity = pc[:, 3:4]
        extras = pc[:, 4:6]  # ring, timestamp

        # 强度预处理
        if args.intensity_mult not in (None, 1.0):
            intensity = intensity * float(args.intensity_mult)
            if args.round_intensity:
                intensity = np.round(intensity)

        # 近距过滤
        if args.min_dist > 0:
            dist = np.linalg.norm(xyz, axis=1)
            far_mask = dist > float(args.min_dist)
        else:
            far_mask = np.ones(len(pc), dtype=bool)

        pc_sim = np.concatenate([xyz[far_mask], intensity[far_mask], extras[far_mask]], axis=1)

        # 参数集
        alpha_resolved, _ = resolve_alpha(args)
        p = ParameterSet(
            gamma=args.gamma,
            gamma_min=1e-7, gamma_max=1e-5, gamma_scale=10_000_000,
            stepsize=args.stepsize
        )
        # 可选字段
        try:
            p.integral_min = float(args.integral_min)
            p.integral_max = float(args.integral_max)
        except Exception:
            pass
        try:
            p.tau_h = float(args.tau_h)
            p.tau_h_ns = float(args.tau_h)
        except Exception:
            pass

        p.alpha = alpha_resolved
        if args.beta is not None:
            p.beta = float(args.beta)
        if p.alpha > 0:
            p.mor = np.log(20.0) / p.alpha
            p.beta_scale = 1000.0 * p.mor
        p.beta_0 = p.gamma / np.pi

        # r_range
        if len(pc_sim):
            p.r_range = float(np.max(np.linalg.norm(pc_sim[:, :3], axis=1)))
        else:
            p.r_range = 0.0

        # 固定随机性：每文件一致
        if args.seed is not None:
            seed = (hash(str(rel)) ^ args.seed) & 0xFFFFFFFF
            np.random.seed(seed)

        # 仿真
        fogged_pc, _, _ = simulate_fog(p, pc_sim, args.noise, args.gain, args.noise_variant)

        # 拼回 6 列
        if fogged_pc.shape[1] < 4:
            # 异常：退回透传（静默）
            shutil.copyfile(in_path, out_path)
            return None

        fog_xyzi = fogged_pc[:, :4].astype(np.float32, copy=False)
        fog_extras = extras[far_mask].astype(np.float32, copy=False)
        out_pc = np.concatenate([fog_xyzi, fog_extras], axis=1)

        # 回插近距点（未加雾）
        if args.keep_near and args.min_dist > 0:
            near_pc = pc[~far_mask].astype(np.float32, copy=False)
            if near_pc.size:
                out_pc = np.vstack([out_pc, near_pc])

        # 范围裁剪
        if args.apply_range and out_pc.size:
            min_x, min_y, min_z, max_x, max_y, max_z = args.range
            x, y, z = out_pc[:, 0], out_pc[:, 1], out_pc[:, 2]
            m = (
                (x >= min_x) & (x <= max_x) &
                (y >= min_y) & (y <= max_y) &
                (z >= min_z) & (z <= max_z)
            )
            out_pc = out_pc[m]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_pc.astype(np.float32, copy=False).tofile(str(out_path))

        # 仅当“强度全 0”才提示一次
        if out_pc.size and np.all(out_pc[:, 3] == 0):
            return f"[ZERO_OUTPUT] {rel}"
        return None

    except Exception as e:
        # 异常：打印错误信息（尽量不影响整体）
        return f"[ERR] {in_path}: {e}"


def main():
    ap = argparse.ArgumentParser("Apply 'our fog simulation' to ALL .bin in a folder (recursive)")
    ap.add_argument("--input-dir", type=str, required=True, help="输入文件夹（递归处理 .bin）")
    ap.add_argument("--output-dir", type=str, required=True, help="输出文件夹（镜像结构，文件名不变）")
    ap.add_argument("--workers", type=int, default=max(cpu_count() - 1, 1), help="并行进程数")

    # 参数（与单文件一致）
    ap.add_argument("--intensity-mult", type=float, default=255.0)
    ap.add_argument("--round-intensity", action="store_true", default=True)
    ap.add_argument("--no-round-intensity", dest="round_intensity", action="store_false")
    ap.add_argument("--min-dist", type=float, default=1.75)
    ap.add_argument("--keep-near", action="store_true")

    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--mor", type=float, default=None)
    ap.add_argument("--visibility", type=float, default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=1e-6)
    ap.add_argument("--stepsize", type=float, default=0.1)
    ap.add_argument("--integral-min", type=float, default=0.0)
    ap.add_argument("--integral-max", type=float, default=200.0)
    ap.add_argument("--tau-h", type=float, default=20.0)

    ap.add_argument("--noise", type=float, default=10.0)
    ap.add_argument("--gain", action="store_true", default=True)
    ap.add_argument("--no-gain", dest="gain", action="store_false")
    ap.add_argument("--noise-variant", type=str, default="v4")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--apply-range", action="store_true")
    ap.add_argument("--range", type=float, nargs=6, default=[-40, -10, -3, 40, 60.4, 1])

    args = ap.parse_args()

    # ★ 在这里打印“实际生效”的参数（解析后的 alpha 等）
    print_effective_params(args)

    in_root = Path(args.input_dir).resolve()
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    files = sorted(in_root.rglob("*.bin"))
    print(f"Found {len(files)} .bin files under {in_root}\n")

    worker = partial(process_one_file, out_root=out_root, in_root=in_root, args=args)

    # 只显示进度条；只有 ZERO_* 或 ERR 才输出一行提示
    if args.workers <= 1:
        for msg in tqdm(map(worker, files), total=len(files), desc="Processing", unit="file"):
            if msg and (msg.startswith("[ZERO") or msg.startswith("[ERR]")):
                print(msg)
    else:
        with Pool(processes=args.workers) as pool:
            for msg in tqdm(pool.imap_unordered(worker, files), total=len(files), desc="Processing", unit="file"):
                if msg and (msg.startswith("[ZERO") or msg.startswith("[ERR]")):
                    print(msg)


if __name__ == "__main__":
    main()
