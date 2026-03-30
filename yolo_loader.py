import sys
import os
from ultralytics import YOLO
import numpy as np

def get_model(model_path):
    """モデルをロードして返す"""
    if not os.path.isabs(model_path):
        model_path = os.path.abspath(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    print(f"[Python Wrapper] Loading YOLO model from: {model_path}")
    model = YOLO(model_path)
    return model

def detect_safe(model, image_path, conf=0.5, device='cpu'):
    """
    推論を実行し、結果をMATLABが理解できる「Pythonのリスト」として返します。
    PyTorchのTensorをMATLABに直接渡さないことでエラーを防ぎます。
    
    Returns:
        list: [[x, y, w, h], [x, y, w, h], ...] (なければ空リスト)
    """
    # 推論実行
    results = model(image_path, verbose=False, conf=conf, device=device)
    result = results[0]
    
    # 検出がない場合
    if result.boxes is None or len(result.boxes) == 0:
        return []
    
    # Tensor -> Numpy -> Python List に変換
    # xywh: 中心x, 中心y, 幅, 高さ
    boxes_np = result.boxes.xywh.cpu().numpy()
    
    return boxes_np.tolist()