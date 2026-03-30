"""
Reprocess Data Script (Auto ArUco Detection & YOLO Tracking)
Fix: Force MATLAB to use the current virtual environment (.venv).
Includes diagnostics for 'ultralytics' import errors.
Supports both ArUco PnP and Backboard 4-point calibration methods.
"""

import matlab.engine
import os
from pathlib import Path
import sys
import cv2
import importlib

# === 設定エリア =========================================
# dataフォルダ内にある、解析したい被験者のフォルダ名
# 例: data/cy22056 なら "cy22056" と入力
TARGET_STUDENT_ID = "cy22219"

# 解析する試行番号の範囲 (例: 1〜5)
TRIAL_START = 1
TRIAL_END = 1

# キャリブレーション方式の選択
# "backboard" : バックボード4点法（推奨、高精度）
# "aruco"     : ArUco PnP法（従来方式）
# "existing"  : 既存の marker_pose.mat を使用（キャリブレーション済みの場合）
CALIBRATION_METHOD = "existing"

# バックボード4点法の場合の設定（画像座標）
# 各試行の最初のフレームでバックボードの4角を指定
# 順序: 左上, 右上, 右下, 左下
# None の場合は対話的に指定
BACKBOARD_POINTS_LEFT = None   # 例: [(100, 200), (500, 200), (500, 400), (100, 400)]
BACKBOARD_POINTS_RIGHT = None  # 例: [(120, 210), (520, 210), (520, 410), (120, 410)]
# ========================================================


def get_backboard_points_interactive(frame, camera_name):
    """対話的にバックボード4点を指定"""
    points = []
    
    # 表示用にリサイズ
    display_width = 960
    display_height = 540
    orig_h, orig_w = frame.shape[:2]
    scale = min(display_width / orig_w, display_height / orig_h)
    
    # 最初にリサイズした基準フレームを作成
    base_display = cv2.resize(frame.copy(), None, fx=scale, fy=scale)
    
    window_name = f"Backboard Points - {camera_name}"
    
    def local_mouse_callback(event, x, y, flags, param):
        """ローカルマウスコールバック（クロージャで points と scale を参照）"""
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                # 表示座標を元画像座標に変換
                orig_x = int(x / scale)
                orig_y = int(y / scale)
                points.append((orig_x, orig_y))
                print(f"  点 {len(points)}: ({orig_x}, {orig_y})")
    
    point_labels = ['TL', 'TR', 'BR', 'BL']  # Top-Left, Top-Right, Bottom-Right, Bottom-Left
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (128, 0, 128)]
    
    print(f"\n  {camera_name}: Click 4 corners of backboard")
    print("  Order: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left")
    print("  [R]: Reset, [Enter]: Confirm, [ESC]: Cancel")
    
    # ウィンドウを作成してコールバックを設定
    cv2.destroyAllWindows()  # 既存のウィンドウをすべて破棄
    cv2.waitKey(1)  # ウィンドウ破棄を確実に処理
    
    # ウィンドウを明示的に作成し、最初のフレームを表示
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(window_name, base_display)
    cv2.waitKey(1)  # ウィンドウ作成を確実に処理
    cv2.setMouseCallback(window_name, local_mouse_callback)
    
    while True:
        # 表示フレームを作成（base_displayをコピー）
        display_frame = base_display.copy()
        
        # 指定済みの点を描画（表示座標系で）
        for i, (px, py) in enumerate(points):
            disp_x = int(px * scale)
            disp_y = int(py * scale)
            cv2.circle(display_frame, (disp_x, disp_y), 8, colors[i], -1)
            cv2.circle(display_frame, (disp_x, disp_y), 10, (255, 255, 255), 2)
            cv2.putText(display_frame, point_labels[i], (disp_x + 12, disp_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 線で結ぶ（表示座標系で）
        if len(points) >= 2:
            for i in range(len(points) - 1):
                p1 = (int(points[i][0] * scale), int(points[i][1] * scale))
                p2 = (int(points[i+1][0] * scale), int(points[i+1][1] * scale))
                cv2.line(display_frame, p1, p2, (0, 255, 255), 2)
            if len(points) == 4:
                p1 = (int(points[3][0] * scale), int(points[3][1] * scale))
                p2 = (int(points[0][0] * scale), int(points[0][1] * scale))
                cv2.line(display_frame, p1, p2, (0, 255, 255), 2)
        
        # ステータス表示
        if len(points) < 4:
            status = f"Next: {point_labels[len(points)]} ({len(points)}/4)"
        else:
            status = "4 points done - Press [Enter] to confirm"
        cv2.putText(display_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(30) & 0xFF
        
        if key == 27:  # ESC
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            return None
        elif key == ord('r') or key == ord('R'):  # リセット
            points = []
            print("  リセットしました")
        elif key == 13 and len(points) == 4:  # Enter
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            return points
    
    return None

def calibrate_with_backboard(eng, frame_left, frame_right, param_file, pose_file):
    """バックボード4点法でキャリブレーション"""
    global BACKBOARD_POINTS_LEFT, BACKBOARD_POINTS_RIGHT
    
    # 点の取得（設定済みか対話的に指定）
    if BACKBOARD_POINTS_LEFT is not None and BACKBOARD_POINTS_RIGHT is not None:
        points_left = list(BACKBOARD_POINTS_LEFT)
        points_right = list(BACKBOARD_POINTS_RIGHT)
        print("  > 設定済みの座標を使用")
    else:
        print("  > 対話的にバックボード4点を指定します")
        points_left = get_backboard_points_interactive(frame_left, "leftCamera (Camera 0)")
        if points_left is None:
            return False
        
        points_right = get_backboard_points_interactive(frame_right, "rightCamera (Camera 1)")
        if points_right is None:
            return False
        
        # 指定した座標を表示（次回用にコピペ可能）
        print(f"\n  指定した座標（次回用）:")
        print(f"  BACKBOARD_POINTS_LEFT = {points_left}")
        print(f"  BACKBOARD_POINTS_RIGHT = {points_right}")
    
    # MATLABに渡す
    import matlab
    points_left_matlab = matlab.double(points_left)
    points_right_matlab = matlab.double(points_right)
    
    try:
        success, calib_info = eng.calibrate_from_backboard(
            points_left_matlab, points_right_matlab, param_file, pose_file, nargout=2)
        
        if success:
            try:
                width_err = float(calib_info['width_error'])
                height_err = float(calib_info['height_error'])
                measured_w = float(calib_info['measured_width'])
                measured_h = float(calib_info['measured_height'])
                print(f"  > キャリブレーション成功")
                print(f"    計測サイズ: {measured_w:.0f} × {measured_h:.0f} mm")
                print(f"    誤差: 幅 {width_err:.0f}mm, 高さ {height_err:.0f}mm")
            except:
                print(f"  > キャリブレーション成功")
            return True
        else:
            print("  > キャリブレーション失敗")
            return False
    except Exception as e:
        print(f"  > MATLABエラー: {e}")
        return False

def calibrate_with_aruco(eng, frame, param_file, pose_file):
    """ArUco PnP法でキャリブレーション"""
    temp_img_path = "temp_start_frame.jpg"
    cv2.imwrite(temp_img_path, frame)
    
    try:
        success = eng.detect_aruco_pose(temp_img_path, param_file, pose_file, nargout=1)
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
        return success
    except Exception as e:
        print(f"  > MATLABエラー: {e}")
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
        return False

def main():
    # パスの設定
    current_dir = Path(os.getcwd())
    data_dir = current_dir / "data" / TARGET_STUDENT_ID
    param_file = str(current_dir / "params" / "stereoParams.mat")
    yolo_model_path = current_dir / "models" / "best-yolo11n.pt"
    pose_file = str(current_dir / "params" / "marker_pose.mat")

    print(f"作業ディレクトリ: {current_dir}")
    print(f"Python実行パス: {sys.executable}")
    print(f"キャリブレーション方式: {CALIBRATION_METHOD}")

    # --- 1. Python環境の診断 ---
    print("\n--- ライブラリ診断 ---")
    try:
        import ultralytics
        print(f"✔ ultralytics found: {ultralytics.__file__}")
    except ImportError:
        print("❌ ultralytics が見つかりません！")
        print("以下のコマンドを実行してインストールしてください:")
        print(f"& {sys.executable} -m pip install ultralytics")
        return

    # --- 2. ファイル存在確認 ---
    if not os.path.exists(param_file):
        print("\n【エラー】stereoParams.mat が見つかりません。")
        return
    if not yolo_model_path.exists():
        print("\n【エラー】best.pt が見つかりません。")
        return
    if not data_dir.exists():
        print(f"\n【エラー】データフォルダが見つかりません: {data_dir}")
        return

    # --- 3. MATLAB起動 ---
    print("\n=== MATLAB Engineを起動中... ===")
    try:
        eng = matlab.engine.start_matlab()
        
        # Python環境の固定
        eng.pyenv('Version', sys.executable, nargout=0)
        eng.addpath(str(current_dir), nargout=0)
        eng.addpath(str(current_dir / "matlab"), nargout=0)
        print("MATLAB Engine 起動完了\n")
        
        # GPUチェック
        try:
            is_gpu = eng.eval("py.torch.cuda.is_available()", nargout=1)
            if is_gpu:
                gpu_name = eng.eval("string(py.torch.cuda.get_device_name(0))", nargout=1)
                print(f"GPU状態: 有効 ({gpu_name})")
            else:
                print("GPU状態: 無効 (CPUで実行します)")
        except Exception as e:
            print(f"GPU状態: 確認不可 ({e})")

        # 【重要】MATLAB側で ultralytics を事前にインポートしてエラー詳細を吐かせる
        print("MATLAB側での ultralytics インポートテスト...")
        try:
            eng.eval("py.importlib.import_module('ultralytics')", nargout=0)
            print("✔ MATLABでのインポート成功")
        except Exception as e:
            print("❌ MATLABでのインポートに失敗しました。詳細:")
            print(e)
            print("--------------------------------------------------")
            print("可能性が高い原因: DLLの競合やパスの問題です。")
            print("--------------------------------------------------")
            eng.quit()
            return

    except Exception as e:
        print(f"MATLAB起動エラー: {e}")
        return

    # --- 4. キャリブレーション（backboard/aruco方式の場合、最初の動画で実行）---
    calibration_done = False
    
    if CALIBRATION_METHOD == "existing":
        if os.path.exists(pose_file):
            print(f"\n既存の marker_pose.mat を使用します")
            calibration_done = True
        else:
            print(f"\n【エラー】marker_pose.mat が見つかりません: {pose_file}")
            eng.quit()
            return
    
    # --- 5. 解析ループ ---
    processed_count = 0
    for i in range(TRIAL_START, TRIAL_END + 1):
        print(f"\n--- 試行 {i} を解析中 ---")
        
        video0 = str(data_dir / f"trial_{i:02d}_cam0.mp4")
        video1 = str(data_dir / f"trial_{i:02d}_cam1.mp4")
        output_csv = str(data_dir / f"trial_{i:02d}_reprocessed.csv")
        
        if not os.path.exists(video0) or not os.path.exists(video1):
            print(f"  > スキップ: 動画ファイルが見つかりません")
            continue

        # 最初の有効な動画でキャリブレーション実行
        if not calibration_done:
            print(f"  > キャリブレーション実行中...")
            
            # フレーム取得
            cap0 = cv2.VideoCapture(video0)
            cap1 = cv2.VideoCapture(video1)
            
            if not cap0.isOpened() or not cap1.isOpened():
                print("  > エラー: 動画を開けません")
                cap0.release()
                cap1.release()
                continue
            
            ret0, frame0 = cap0.read()
            ret1, frame1 = cap1.read()
            cap0.release()
            cap1.release()
            
            if not ret0 or not ret1:
                print("  > エラー: フレーム読み込み失敗")
                continue
            
            if CALIBRATION_METHOD == "backboard":
                success = calibrate_with_backboard(eng, frame0, frame1, param_file, pose_file)
            elif CALIBRATION_METHOD == "aruco":
                success = calibrate_with_aruco(eng, frame0, param_file, pose_file)
            else:
                print(f"  > 不明なキャリブレーション方式: {CALIBRATION_METHOD}")
                continue
            
            if not success:
                print("  > キャリブレーション失敗、次の試行で再試行します")
                continue
            
            calibration_done = True

        # YOLOトラッキング
        try:
            print(f"  > YOLOトラッキング開始...")
            eng.run_tracking_func(video0, video1, output_csv, param_file, pose_file, nargout=0)
            print(f"  > 完了: {output_csv}")
            processed_count += 1
        except Exception as e:
            print(f"  > エラー発生: {e}")

    # 終了処理
    eng.quit()
    print(f"\n=== 全処理完了 ({processed_count} 件成功) ===")

if __name__ == "__main__":
    main()