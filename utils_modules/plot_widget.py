"""
Plot Widget for Trajectory Visualization

3D軌跡プロット表示用ウィジェット
"""

import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
from pathlib import Path


class TrajectoryPlotWindow:
    """軌跡プロット表示ウィンドウ"""
    
    def __init__(self, parent, student_id: str, trial_number: int = None):
        """
        軌跡プロットウィンドウを初期化
        
        Args:
            parent: 親ウィンドウ
            student_id: 学籍番号
            trial_number: 試行番号（Noneの場合は全試行）
        """
        self.window = tk.Toplevel(parent)
        self.window.title(f"軌跡プロット - {student_id}")
        self.window.geometry("900x700")
        
        self.student_id = student_id
        self.trial_number = trial_number
        
        # メインフレーム
        main_frame = tk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # タイトル
        title_text = f"被験者: {student_id}"
        if trial_number:
            title_text += f" - 試行{trial_number}"
        else:
            title_text += " - 全試行"
        
        title = tk.Label(main_frame, text=title_text, font=("Arial", 16, "bold"))
        title.pack(pady=10)
        
        # プロット領域
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # ツールバー
        toolbar = NavigationToolbar2Tk(self.canvas, main_frame)
        toolbar.update()
        
        # 統計情報表示
        stats_frame = tk.LabelFrame(main_frame, text="統計情報", font=("Arial", 12))
        stats_frame.pack(fill=tk.X, pady=10)
        
        self.stats_label = tk.Label(stats_frame, text="", font=("Arial", 10), justify=tk.LEFT)
        self.stats_label.pack(padx=10, pady=5)
        
        # データ読み込みとプロット
        self.load_and_plot()
    
    def load_and_plot(self):
        """データを読み込んでプロット"""
        data_dir = Path("data") / self.student_id
        
        if not data_dir.exists():
            self.stats_label.config(text="エラー: データフォルダが見つかりません")
            return
        
        # データ読み込み
        if self.trial_number:
            # 単一試行
            csv_path = data_dir / f"trial_{self.trial_number:02d}.csv"
            if not csv_path.exists():
                self.stats_label.config(text=f"エラー: {csv_path.name} が見つかりません")
                return
            
            df = pd.read_csv(csv_path)
            self.plot_single_trial(df, self.trial_number)
        else:
            # 全試行
            all_trials = []
            for i in range(1, 11):
                csv_path = data_dir / f"trial_{i:02d}.csv"
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    df['trial'] = i
                    all_trials.append(df)
            
            if all_trials:
                combined_df = pd.concat(all_trials, ignore_index=True)
                self.plot_all_trials(combined_df)
            else:
                self.stats_label.config(text="エラー: 試行データが見つかりません")
    
    def plot_single_trial(self, df: pd.DataFrame, trial_num: int):
        """単一試行のプロット"""
        # 3D軌跡プロット
        self.ax.plot(df['X_m'], df['Y_m'], df['Z_m'], 'b-', linewidth=2, label=f'試行{trial_num}')
        
        # 始点と終点
        self.ax.scatter(df['X_m'].iloc[0], df['Y_m'].iloc[0], df['Z_m'].iloc[0], 
                       c='green', s=100, marker='o', label='開始')
        self.ax.scatter(df['X_m'].iloc[-1], df['Y_m'].iloc[-1], df['Z_m'].iloc[-1], 
                       c='red', s=100, marker='X', label='終了')
        
        # ArUcoマーカー表示（原点）
        self.ax.scatter([0], [0], [0], c='black', s=200, marker='^', label='ArUco原点')
        
        # 軸ラベル
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        self.ax.set_zlabel('Z [m]')
        self.ax.set_title(f'3D軌跡 - 試行{trial_num}')
        self.ax.legend()
        self.ax.grid(True)
        
        # 統計情報
        stats_text = f"データポイント数: {len(df)}\n"
        stats_text += f"最大高度: {df['Z_m'].max():.3f} m\n"
        stats_text += f"最小高度: {df['Z_m'].min():.3f} m\n"
        
        if 'VX_ms' in df.columns:
            speed = np.sqrt(df['VX_ms']**2 + df['VY_ms']**2 + df['VZ_ms']**2)
            stats_text += f"最大速度: {speed.max():.3f} m/s\n"
            stats_text += f"平均速度: {speed.mean():.3f} m/s"
        
        self.stats_label.config(text=stats_text)
    
    def plot_all_trials(self, df: pd.DataFrame):
        """全試行のプロット"""
        # 色マップ
        colors = plt.cm.rainbow(np.linspace(0, 1, df['trial'].nunique()))
        
        # 各試行をプロット
        for i, trial_num in enumerate(sorted(df['trial'].unique())):
            trial_df = df[df['trial'] == trial_num]
            self.ax.plot(trial_df['X_m'], trial_df['Y_m'], trial_df['Z_m'], 
                        color=colors[i], linewidth=1.5, alpha=0.7, label=f'試行{trial_num}')
        
        # ArUcoマーカー表示（原点）
        self.ax.scatter([0], [0], [0], c='black', s=200, marker='^', label='ArUco原点')
        
        # 軸ラベル
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        self.ax.set_zlabel('Z [m]')
        self.ax.set_title('3D軌跡 - 全試行')
        self.ax.legend(loc='upper right', fontsize=8)
        self.ax.grid(True)
        
        # 統計情報
        stats_text = f"試行数: {df['trial'].nunique()}\n"
        stats_text += f"総データポイント数: {len(df)}\n"
        stats_text += f"最大高度: {df['Z_m'].max():.3f} m\n"
        stats_text += f"平均高度: {df['Z_m'].mean():.3f} m"
        
        self.stats_label.config(text=stats_text)


# matplotlibのpltをインポート（色マップ用）
import matplotlib.pyplot as plt
