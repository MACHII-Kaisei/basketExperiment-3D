"""
3D Trajectory Visualizer
生成されたCSVファイルを読み込み、
バスケットボールコートの要素（バックボード、リング、フリースローライン）と
ボール軌跡をインタラクティブな3Dグラフとして表示します。
"""

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from pathlib import Path

# === 設定 ===
CSV_FILE = "C:\\BasketExperiment-yolo\\data\\cy22219\\trial_01.csv"  # 確認したいCSVファイル名

# コート寸法 (mm) - ArUco座標系基準, FIBA国際規格
ARUCO_MARKER_SIZE = 550              # ArUcoマーカーのサイズ
BACKBOARD_WIDTH = 1800               # バックボード幅
BACKBOARD_HEIGHT = 1050              # バックボード高さ
RING_HEIGHT = 3050                   # リング高さ（床から）
RING_RADIUS = 225                    # リング内径/2
RING_OFFSET_FROM_BACKBOARD = 151     # バックボード面からリング中心までの距離
FREETHROW_DISTANCE = 4600            # フリースローラインまでの距離（バックボード面から、FIBA規格）
ARUCO_DISTANCE_FROM_BACKBOARD = 2600 # ArUcoからバックボード面までの距離 (4600 - 2000)

# 表示オプション
SHOW_BACKBOARD = True
SHOW_RING = True
SHOW_FREETHROW_LINE = True
SHOW_ARUCO_MARKER = True
SHOW_FLOOR_GRID = True
# ============


def draw_backboard(ax):
    """バックボードを描画"""
    # バックボード位置（ArUco座標系）
    # Y = ArUcoからバックボードまでの距離
    bb_y = ARUCO_DISTANCE_FROM_BACKBOARD
    
    # バックボードの4角（ArUco座標系）
    # バックボード中心の高さ = リング高さ + バックボード高さ/2 - 150mm
    # (リングはバックボード下端から約150mm上にある)
    bb_center_z = RING_HEIGHT + BACKBOARD_HEIGHT/2 - 150
    
    w2 = BACKBOARD_WIDTH / 2
    h2 = BACKBOARD_HEIGHT / 2
    
    # バックボード面（XZ平面に平行、Y=bb_y）
    corners = [
        [-w2, bb_y, bb_center_z - h2],  # 左下
        [w2, bb_y, bb_center_z - h2],   # 右下
        [w2, bb_y, bb_center_z + h2],   # 右上
        [-w2, bb_y, bb_center_z + h2],  # 左上
    ]
    
    verts = [[corners[0], corners[1], corners[2], corners[3]]]
    ax.add_collection3d(Poly3DCollection(verts, facecolors='white', linewidths=2, 
                                         edgecolors='black', alpha=0.5))
    
    # バックボードの枠を強調
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i+1) % 4]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'k-', linewidth=2)
    
    # ラベル
    ax.text(0, bb_y + 100, bb_center_z + h2 + 200, 'Backboard', fontsize=10, ha='center')


def draw_ring(ax):
    """リングを描画"""
    # リング位置（ArUco座標系）
    ring_y = ARUCO_DISTANCE_FROM_BACKBOARD - RING_OFFSET_FROM_BACKBOARD
    ring_z = RING_HEIGHT
    
    # リングを円として描画
    theta = np.linspace(0, 2 * np.pi, 50)
    ring_x = RING_RADIUS * np.cos(theta)
    ring_y_arr = ring_y + RING_RADIUS * np.sin(theta)  # Y方向にも広がる
    ring_z_arr = np.full_like(theta, ring_z)
    
    # リングはXY平面に平行（水平）なので、Y方向ではなくX方向の円
    ring_x = RING_RADIUS * np.cos(theta)
    ring_y_circle = ring_y * np.ones_like(theta)  # 一定のY位置に
    # 実際のリングは水平なので、XZ平面で円を描く
    ring_x = RING_RADIUS * np.cos(theta)
    ring_y_circle = ring_y - RING_RADIUS * np.sin(theta)  # 手前に突き出る
    ring_z_arr = np.full_like(theta, ring_z)
    
    ax.plot(ring_x, ring_y_circle, ring_z_arr, 'orange', linewidth=3, label='Ring')
    
    # リング中心にマーカー
    ax.scatter([0], [ring_y], [ring_z], color='orange', s=50, marker='o')
    
    # ネット（簡易的に線で表現）
    net_depth = 400  # ネットの深さ
    for i in range(0, 360, 30):
        rad = np.radians(i)
        nx = RING_RADIUS * np.cos(rad)
        ny = ring_y - RING_RADIUS * np.sin(rad)
        ax.plot([nx, nx*0.3], [ny, ny], [ring_z, ring_z - net_depth], 
                color='white', linewidth=0.5, alpha=0.5)


def draw_freethrow_line(ax):
    """フリースローラインを描画"""
    # フリースローライン位置（ArUco座標系）
    # ArUcoはフリースローラインの2m前方なので
    ft_y = ARUCO_DISTANCE_FROM_BACKBOARD - FREETHROW_DISTANCE  # 負の値になる
    
    # フリースローラインの幅（コート幅の一部、約3.6m）
    ft_width = 3600
    
    # 床面上（Z=0）に線を描画
    ax.plot([-ft_width/2, ft_width/2], [ft_y, ft_y], [0, 0], 
            'b-', linewidth=3, label='Free Throw Line')
    
    # ラベル
    ax.text(0, ft_y - 300, 0, 'Free Throw Line', fontsize=9, ha='center', color='blue')


def draw_aruco_marker(ax):
    """ArUcoマーカーを描画"""
    ms2 = ARUCO_MARKER_SIZE / 2
    
    # マーカー板 (床面上、Z=0)
    xx = np.array([-ms2, ms2, ms2, -ms2])
    yy = np.array([-ms2, -ms2, ms2, ms2])
    zz = np.array([0, 0, 0, 0])
    
    verts = [list(zip(xx, yy, zz))]
    ax.add_collection3d(Poly3DCollection(verts, facecolors='gray', linewidths=1, 
                                         edgecolors='black', alpha=0.5))
    
    # 座標軸 (X:赤, Y:緑, Z:青)
    axis_len = ARUCO_MARKER_SIZE * 1.5
    ax.quiver(0, 0, 0, axis_len, 0, 0, color='r', arrow_length_ratio=0.1, linewidth=2)
    ax.quiver(0, 0, 0, 0, axis_len, 0, color='g', arrow_length_ratio=0.1, linewidth=2)
    ax.quiver(0, 0, 0, 0, 0, axis_len, color='b', arrow_length_ratio=0.1, linewidth=2)
    
    ax.text(axis_len + 100, 0, 0, "X", color='red', fontsize=12, fontweight='bold')
    ax.text(0, axis_len + 100, 0, "Y (→BB)", color='green', fontsize=12, fontweight='bold')
    ax.text(0, 0, axis_len + 100, "Z (↑)", color='blue', fontsize=12, fontweight='bold')
    
    # ArUcoラベル
    ax.text(0, -ms2 - 200, 0, 'ArUco Origin', fontsize=9, ha='center')


def draw_floor_grid(ax):
    """床面のグリッドを描画"""
    # グリッド範囲
    grid_range = 5000  # 5m
    grid_step = 1000   # 1m間隔
    
    for i in np.arange(-grid_range, grid_range + grid_step, grid_step):
        # X方向の線
        ax.plot([i, i], [-grid_range, ARUCO_DISTANCE_FROM_BACKBOARD + 500], [0, 0], 
                'gray', linewidth=0.3, alpha=0.3)
        # Y方向の線
        ax.plot([-grid_range, grid_range], [i, i], [0, 0], 
                'gray', linewidth=0.3, alpha=0.3)


def main():
    # ファイルパス確認
    csv_path = Path(CSV_FILE).resolve()
    if not csv_path.exists():
        print(f"エラー: ファイルが見つかりません: {csv_path}")
        print("先に reprocess_data_auto.py を実行してデータを作成してください。")
        return

    print(f"データを読み込んでいます: {csv_path.name}")
    
    try:
        # CSV読み込み
        df = pd.read_csv(csv_path)
        
        # データの単位確認 (mmかmか)
        if 'X_mm' in df.columns:
            x = df['X_mm'].values
            y = df['Y_mm'].values
            z = df['Z_mm'].values
            unit = "mm"
        elif 'X_m' in df.columns:
            x = df['X_m'].values * 1000  # mmに変換して表示
            y = df['Y_m'].values * 1000
            z = df['Z_m'].values * 1000
            unit = "mm (converted from m)"
        else:
            print("エラー: 座標データ(X_mm または X_m)が見つかりません")
            return

        # 3Dプロット作成
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # --- コート要素の描画 ---
        if SHOW_FLOOR_GRID:
            draw_floor_grid(ax)
        
        if SHOW_ARUCO_MARKER:
            draw_aruco_marker(ax)
        
        if SHOW_BACKBOARD:
            draw_backboard(ax)
        
        if SHOW_RING:
            draw_ring(ax)
        
        if SHOW_FREETHROW_LINE:
            draw_freethrow_line(ax)
        
        # --- ボール軌跡のプロット ---
        ax.plot(x, y, z, 'b-', linewidth=2.5, label='Ball Trajectory')
        
        # 軌跡上の点をグラデーションで表示（時間経過を可視化）
        n_points = len(x)
        colors = plt.cm.coolwarm(np.linspace(0, 1, n_points))
        ax.scatter(x, y, z, c=colors, s=20, alpha=0.6)
        
        # 開始・終了点
        ax.scatter(x[0], y[0], z[0], color='lime', s=150, marker='o', 
                  edgecolors='black', linewidths=2, label='Start', zorder=5)
        ax.scatter(x[-1], y[-1], z[-1], color='red', s=150, marker='X', 
                  edgecolors='black', linewidths=2, label='End', zorder=5)

        # --- 見た目の調整 ---
        ax.set_xlabel(f'X [{unit}]', fontsize=11)
        ax.set_ylabel(f'Y [{unit}] (→Backboard)', fontsize=11)
        ax.set_zlabel(f'Z [{unit}] (↑Height)', fontsize=11)
        ax.set_title(f'3D Basketball Shot Trajectory\nFile: {csv_path.name}', fontsize=14)
        
        # 軸範囲を設定（コート全体が見えるように）
        x_range = max(abs(x.max()), abs(x.min()), BACKBOARD_WIDTH/2) + 500
        y_min = min(y.min(), -FREETHROW_DISTANCE + ARUCO_DISTANCE_FROM_BACKBOARD) - 500
        y_max = ARUCO_DISTANCE_FROM_BACKBOARD + 500
        z_max = max(z.max(), RING_HEIGHT + BACKBOARD_HEIGHT/2) + 500
        
        ax.set_xlim(-x_range, x_range)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(0, z_max)
        
        # アスペクト比を調整（完全な等倍ではなく見やすさ重視）
        ax.set_box_aspect([2*x_range, y_max - y_min, z_max])
        
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 初期視点を設定（斜め上から）
        ax.view_init(elev=20, azim=-60)
        
        # 統計情報を表示
        print(f"\n=== 軌跡統計 ===")
        print(f"フレーム数: {n_points}")
        print(f"X範囲: {x.min():.0f} ~ {x.max():.0f} mm")
        print(f"Y範囲: {y.min():.0f} ~ {y.max():.0f} mm (バックボード方向)")
        print(f"Z範囲: {z.min():.0f} ~ {z.max():.0f} mm (高さ)")
        print(f"\nリング高さ: {RING_HEIGHT} mm")
        print(f"リングY位置: {ARUCO_DISTANCE_FROM_BACKBOARD - RING_OFFSET_FROM_BACKBOARD:.0f} mm")
        
        print("\nウィンドウを表示します。マウスで回転・拡大縮小が可能です。")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()