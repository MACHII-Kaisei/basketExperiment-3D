import numpy as np
from pathlib import Path

def _resolve_camera_name(camera_name):
    return f'camera{camera_name}' if isinstance(camera_name, int) else str(camera_name)

def _read_numeric_lines(path):
    """ファイルから数値データを読み取るヘルパー関数"""
    numeric_lines = []
    if not Path(path).exists():
        raise FileNotFoundError(f"Parameter file not found: {path}")
        
    with open(path, 'r') as inf:
        for line in inf:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                floats = [float(token) for token in stripped.split()]
            except ValueError:
                continue
            if floats:
                numeric_lines.append(floats)
    if not numeric_lines:
        raise ValueError(f'No numeric data found in {path}')
    return numeric_lines

def read_intrinsics(camera_name, folder='camera_parameters'):
    """
    内部パラメータ(intrinsics)を読み込む
    GUIのArUcoプレビュー表示(drawFrameAxes)で使用
    """
    cam_name = _resolve_camera_name(camera_name)
    path = Path(folder) / f'{cam_name}_intrinsics.dat'
    numeric_lines = _read_numeric_lines(path)
    flat_values = [value for line in numeric_lines for value in line]

    if len(flat_values) < 9:
        raise ValueError(f'Intrinsic file {path} must contain at least 9 numeric values.')

    cmtx = np.array(flat_values[:9], dtype=np.float64).reshape(3, 3)
    
    dist_values = flat_values[9:]
    if not dist_values:
        # 歪み係数がない場合は0で埋める
        dist = np.zeros((1, 5), dtype=np.float64)
    else:
        dist = np.array(dist_values, dtype=np.float64).reshape(1, -1)
        
    return cmtx, dist

def read_extrinsics(camera_name, folder='camera_parameters'):
    """外部パラメータ(Rotation, Translation)を読み込む"""
    cam_name = _resolve_camera_name(camera_name)
    path = Path(folder) / f'{cam_name}_rot_trans.dat'
    numeric_lines = _read_numeric_lines(path)
    flat_values = [value for line in numeric_lines for value in line]

    if len(flat_values) < 12:
        raise ValueError(f'Extrinsic file {path} must contain at least 12 values.')

    R = np.array(flat_values[:9], dtype=np.float64).reshape(3, 3)
    T = np.array(flat_values[9:12], dtype=np.float64).reshape(3, 1)
    return R, T

def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4,4))
    P[:3,:3] = R
    P[:3, 3] = t.reshape(3)
    P[3,3] = 1
    return P

def get_projection_matrix(camera_id):
    """
    投影行列Pを取得する
    GUIコードがimportしているため維持（実際にはMATLAB側で計算するが、エラー回避のため残す）
    """
    cmtx, _ = read_intrinsics(camera_id)
    R, T = read_extrinsics(camera_id)
    P = cmtx @ _make_homogeneous_rep_matrix(R, T)[:3,:]
    return P