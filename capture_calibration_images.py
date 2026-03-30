import cv2
import yaml
import os
import time
from pathlib import Path

def load_config(config_path="calibration_settings.yaml"):
    """設定ファイルを読み込む"""
    if not os.path.exists(config_path):
        print(f"エラー: {config_path} が見つかりません。")
        return None
    
    with open(config_path, "r", encoding="utf-8") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"YAML読み込みエラー: {e}")
            return None

def draw_text_with_bg(img, text, pos, font_scale=1.0, font_thickness=2, text_color=(255, 255, 255), bg_color=(0, 0, 0), padding=10):
    """
    背景色付きでテキストを描画するヘルパー関数
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = pos
    
    # テキストサイズを取得
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    
    # 背景の矩形座標を計算
    cv2.rectangle(img, 
                  (x - padding, y - text_h - padding), 
                  (x + text_w + padding, y + baseline + padding), 
                  bg_color, 
                  -1) # 塗りつぶし
    
    # テキストを描画
    cv2.putText(img, text, (x, y), font, font_scale, text_color, font_thickness)

def main():
    # 1. 設定読み込み
    config = load_config()
    if config is None:
        return

    camera0_id = config.get("camera0", 0)
    camera1_id = config.get("camera1", 1)
    width = config.get("frame_width", 1920)
    height = config.get("frame_height", 1080)
    
    # 撮影間隔（フレーム数）の設定読み込み（デフォルト100）
    capture_interval_frames = config.get("calibration_capture_interval", 100)

    print("="*60)
    print(" ステレオキャリブレーション画像撮影ツール (UI改善版)")
    print("="*60)
    print(f" Camera Left : ID {camera0_id}")
    print(f" Camera Right: ID {camera1_id}")
    print(f" Resolution  : {width}x{height}")
    print(f" Interval    : {capture_interval_frames} frames")
    print("-" * 60)
    print(" [S]キー : 自動撮影開始/停止")
    print(" [Q]キー : 終了")
    print("="*60)

    # 2. 保存先ディレクトリ作成
    base_dir = Path("calibration_data")
    left_dir = base_dir / "left"
    right_dir = base_dir / "right"

    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    # 3. カメラ初期化
    cap0 = cv2.VideoCapture(camera0_id)
    cap1 = cv2.VideoCapture(camera1_id)

    for cap in [cap0, cap1]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) # オートフォーカスOFF推奨

    if not cap0.isOpened() or not cap1.isOpened():
        print("エラー: カメラを開けませんでした。IDを確認してください。")
        return

    # 4. 撮影ループ
    capturing = False
    frames_since_capture = 0 # 経過フレームカウンタ
    image_count = 0
    
    # 既存ファイル数カウント
    existing_files = list(left_dir.glob("image_*.jpg"))
    if existing_files:
        image_count = len(existing_files)
        print(f"既存の画像 {image_count} 枚を検出しました。続きから番号を振ります。")

    try:
        while True:
            ret0, frame0 = cap0.read()
            ret1, frame1 = cap1.read()

            if not ret0 or not ret1:
                print("フレーム取得エラー")
                break

            display_frame0 = frame0.copy()
            
            # --- 自動撮影ロジック (フレームベース) ---
            status_bg_color = (0, 100, 0) # 深緑 (待機中)
            status_text_header = "STANDBY"
            
            if capturing:
                frames_since_capture += 1
                status_bg_color = (0, 0, 150) # 深紅 (撮影中)
                status_text_header = "REC ON"
                
                # 指定フレーム数経過したら保存
                if frames_since_capture >= capture_interval_frames:
                    image_count += 1
                    filename = f"image_{image_count:02d}.jpg"
                    
                    p0 = left_dir / filename
                    p1 = right_dir / filename
                    
                    cv2.imwrite(str(p0), frame0)
                    cv2.imwrite(str(p1), frame1)
                    
                    print(f"Saved: {filename}")
                    frames_since_capture = 0 # カウンタリセット
                    
                    # フラッシュエフェクト（画面全体を一瞬白く）
                    cv2.rectangle(display_frame0, (0, 0), (width, height), (255, 255, 255), 40)

            # --- 画面表示 (巨大化・減算ゲージ版) ---
            
            # 1. 状態表示 (左上) - さらに大きく
            draw_text_with_bg(display_frame0, 
                            f"{status_text_header} (Press 's')", 
                            (30, 80), 
                            font_scale=2.0, 
                            font_thickness=4, 
                            bg_color=status_bg_color)

            # 2. カウント表示 (左下)
            draw_text_with_bg(display_frame0, 
                            f"Saved: {image_count}", 
                            (30, height - 50), 
                            font_scale=1.5, 
                            font_thickness=3, 
                            bg_color=(50, 50, 50)) # グレー背景

            # 3. 次の撮影までのカウントダウンバー (減っていくゲージ)
            if capturing:
                bar_width = 600   # バーの最大幅を拡大
                bar_height = 40   # バーの太さを拡大
                bar_x = 30
                bar_y = 130
                
                # 残りフレーム数
                frames_left = max(0, capture_interval_frames - frames_since_capture)
                
                # 割合 (1.0 -> 0.0)
                progress = frames_left / capture_interval_frames
                
                # 色の変化 (緑 -> 黄 -> 赤)
                bar_color = (0, 255, 0) # Green
                if progress < 0.3:
                    bar_color = (0, 0, 255) # Red
                elif progress < 0.6:
                    bar_color = (0, 255, 255) # Yellow

                # 背景バー (最大幅の枠)
                cv2.rectangle(display_frame0, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
                
                # 進捗バー (減っていく)
                current_bar_width = int(bar_width * progress)
                if current_bar_width > 0:
                    cv2.rectangle(display_frame0, (bar_x, bar_y), (bar_x + current_bar_width, bar_y + bar_height), bar_color, -1)
                
                # テキスト (残りフレーム数)
                draw_text_with_bg(display_frame0, f"Next: {frames_left}", (bar_x + bar_width + 20, bar_y + 30), 
                                font_scale=1.2, font_thickness=2, bg_color=(0,0,0))

            # プレビュー作成（リサイズして結合）
            preview_h = 480
            scale = preview_h / height
            preview_w = int(width * scale)
            
            p0 = cv2.resize(display_frame0, (preview_w, preview_h))
            p1 = cv2.resize(frame1, (preview_w, preview_h))
            
            combined = cv2.hconcat([p0, p1])
            cv2.imshow("Stereo Calibration Capture", combined)

            # キー操作
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                capturing = not capturing
                frames_since_capture = 0 # 切り替え時にリセット
                if capturing:
                    print(f"\n--- 自動撮影開始 ({capture_interval_frames} frames interval) ---")
                else:
                    print("--- 待機中 ---")

    finally:
        cap0.release()
        cap1.release()
        cv2.destroyAllWindows()
        print("\n終了しました。")

if __name__ == "__main__":
    main()