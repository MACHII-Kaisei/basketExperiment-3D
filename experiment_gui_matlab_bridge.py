"""
Experiment GUI - MATLAB Engine Version (YOLO対応版)
(同期ズレ対策: grab/retrieve方式 + MATLAB自動同期対応版)
Fix 1: 事前テスト(10本)終了時に一時停止メッセージを表示
Fix 2: 画面遷移時に試行番号をリセットせず、メタデータから続きの番号を読み込む
Fix 3: バックボードキャリブレーション時に拡大ウィンドウを表示して精度を向上
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
from datetime import datetime
from pathlib import Path
import yaml
import threading
import socket
import json
import csv
import time
import os
import sys

# MATLAB Engineのインポート
try:
    import matlab.engine
    MATLAB_AVAILABLE = True
except ImportError:
    MATLAB_AVAILABLE = False
    print("WARNING: 'matlab.engine' not found. Please install MATLAB Engine API for Python.")

# フォルダ構成に対応したインポート
# 実行環境にこれらのモジュールが存在することを前提としています
try:
    from core.experiment_config import DEFAULT_TRIAL_COUNT, get_subject_dir
    from core.data_manager import DataManager
    from utils_modules.utils import read_intrinsics, get_projection_matrix
except ImportError:
    # ローカルで動作確認するためのダミー実装（実際の環境では無視されます）
    DEFAULT_TRIAL_COUNT = 20
    def get_subject_dir(sid): return Path(f"data/{sid}")
    class DataManager:
        def create_subject_folder(self, sid): pass
        def load_metadata(self, sid): return {}
        def add_trial_to_metadata(self, sid, info): pass
    def read_intrinsics(path): return {}
    def get_projection_matrix(intrinsics): return None

class ExperimentRecordProcessGUI:
    def __init__(self, root: tk.Tk, test_mode: bool = False):
        self.root = root
        self.test_mode = test_mode
        self.root.title("バスケットボールトラッキング実験管理 (YOLO & GPU Mode)")
        self.root.geometry("1024x768")
        
        # データ管理
        self.dm = DataManager()
        self.subject_info = {}
        self.trials = []
        self.current_trial = 0
        
        # カメラ関連
        self.cap0 = None
        self.cap1 = None
        self.recording = False
        self.video_writer0 = None
        self.video_writer1 = None
        self.current_video0_path = None
        self.current_video1_path = None
        self.recording_thread = None
        
        # 設定読み込み
        self.config = {}
        self.load_config()
        
        # MATLABエンジン初期化 (非同期)
        self.eng = None
        self.matlab_status = "Not Started"
        if MATLAB_AVAILABLE:
            self.matlab_status = "Initializing..."
            threading.Thread(target=self.init_matlab_engine, daemon=True).start()
        else:
            self.matlab_status = "Not Available"
        
        # GUI状態
        self.preview_running = False
        self.latest_preview_frame = None
        
        # 画面作成
        self.frames = {}
        self.create_frames()
        
        # ステータスバー
        self.status_bar_var = tk.StringVar(value=f"準備中... (MATLAB: {self.matlab_status})")
        self.status_bar = tk.Label(self.root, textvariable=self.status_bar_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.show_frame("SubjectInputScreen")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.check_matlab_status()

    def load_config(self):
        try:
            with open("calibration_settings.yaml", "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            self.config = {"camera0": 0, "camera1": 1, "fps": 30.0}

    def init_matlab_engine(self):
        """MATLABエンジンを起動し、Python環境を同期させる"""
        try:
            print("Starting MATLAB Engine...")
            self.eng = matlab.engine.start_matlab()
            
            # 【重要】現在のPython環境(.venv)をMATLABに使用させる
            print(f"Syncing Python Environment: {sys.executable}")
            self.eng.pyenv('Version', sys.executable, nargout=0)
            
            # カレントディレクトリをパスに追加
            self.eng.addpath(os.getcwd(), nargout=0)
            self.eng.addpath(os.path.join(os.getcwd(), 'matlab'), nargout=0)
            
            # GPUチェック（ログ出力のみ）
            try:
                is_gpu = self.eng.eval("py.torch.cuda.is_available()", nargout=1)
                if is_gpu:
                    gpu_name = self.eng.eval("string(py.torch.cuda.get_device_name(0))", nargout=1)
                    print(f"GPU Detected: {gpu_name}")
                else:
                    print("GPU Not Detected (CPU Mode)")
            except Exception as e:
                print(f"GPU Check Failed: {e}")

            self.matlab_status = "Ready"
            print("MATLAB Engine Ready.")
            
        except Exception as e:
            self.matlab_status = f"Error: {str(e)}"
            print(f"Failed to start MATLAB: {e}")

    def check_matlab_status(self):
        if self.matlab_status == "Ready":
            self.status_bar_var.set("準備完了 (MATLAB Engine Ready & Python Synced)")
        elif "Error" in self.matlab_status:
            self.status_bar_var.set(f"警告: MATLAB起動失敗 - {self.matlab_status}")
        else:
            self.status_bar_var.set(f"準備中... (MATLAB: {self.matlab_status})")
            self.root.after(1000, self.check_matlab_status)

    def create_frames(self):
        container = tk.Frame(self.root)
        container.pack(fill="both", expand=True)
        
        for F, name in [(SubjectInputScreen, "SubjectInputScreen"),
                        (CalibrationMethodScreen, "CalibrationMethodScreen"),
                        (ArucoSetupScreen, "ArucoSetupScreen"),
                        (BackboardCalibrationScreen, "BackboardCalibrationScreen"),
                        (TrialManagementScreen, "TrialManagementScreen"),
                        (CompletionScreen, "CompletionScreen")]:
            frame = F(container, self)
            self.frames[name] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
    
    def show_frame(self, frame_name: str):
        frame = self.frames[frame_name]
        frame.tkraise()
        if hasattr(frame, 'on_show'):
            frame.on_show()
    
    def show_error(self, title: str, message: str):
        messagebox.showerror(title, message)
    
    def show_info(self, title: str, message: str):
        messagebox.showinfo(title, message)
    
    def on_closing(self):
        if messagebox.askokcancel("終了", "プログラムを終了しますか？"):
            self.recording = False
            if self.cap0 is not None: self.cap0.release()
            if self.cap1 is not None: self.cap1.release()
            if self.video_writer0 is not None: self.video_writer0.release()
            if self.video_writer1 is not None: self.video_writer1.release()
            cv2.destroyAllWindows()
            
            if self.eng is not None:
                print("Stopping MATLAB Engine...")
                self.eng.quit()
            
            self.root.destroy()


class SubjectInputScreen(tk.Frame):
    """画面1: 被験者情報入力 + UDP送信"""
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        title = tk.Label(self, text="被験者情報入力", font=("Arial", 24, "bold"))
        title.pack(pady=30)
        
        form_frame = tk.Frame(self)
        form_frame.pack(pady=20)
        
        tk.Label(form_frame, text="学籍番号 (半角英数):", font=("Arial", 14)).grid(row=0, column=0, pady=10, sticky="e")
        self.student_id_entry = tk.Entry(form_frame, font=("Arial", 14), width=30)
        self.student_id_entry.grid(row=0, column=1, pady=10, padx=10)
        
        tk.Label(form_frame, text="氏名:", font=("Arial", 14)).grid(row=1, column=0, pady=10, sticky="e")
        self.name_entry = tk.Entry(form_frame, font=("Arial", 14), width=30)
        self.name_entry.grid(row=1, column=1, pady=10, padx=10)
        
        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=30)
        
        next_btn = tk.Button(btn_frame, text="次へ (ArUco設定)", font=("Arial", 14), 
                            command=self.on_next, bg="#4CAF50", fg="white", padx=20, pady=10)
        next_btn.pack()
        
        # UDP送信設定
        udp_frame = tk.LabelFrame(self, text="UDP送信設定", font=("Arial", 14, "bold"))
        udp_frame.pack(pady=20, padx=40, fill="x")
        
        tk.Label(udp_frame, text="完了した試行データをUDP経由で送信できます", font=("Arial", 10), fg="gray").pack(pady=5)
        
        select_frame = tk.Frame(udp_frame)
        select_frame.pack(pady=10)
        tk.Label(select_frame, text="被験者:", font=("Arial", 12)).pack(side="left", padx=5)
        self.subject_combo = ttk.Combobox(select_frame, font=("Arial", 12), width=25, state="readonly")
        self.subject_combo.pack(side="left", padx=5)
        refresh_btn = tk.Button(select_frame, text="更新", font=("Arial", 10), command=self.refresh_subjects, padx=10)
        refresh_btn.pack(side="left", padx=5)
        
        trial_frame = tk.Frame(udp_frame)
        trial_frame.pack(pady=10)
        tk.Label(trial_frame, text="試行範囲:", font=("Arial", 12)).pack(side="left", padx=5)
        self.trial_start_var = tk.IntVar(value=1)
        self.trial_end_var = tk.IntVar(value=10)
        tk.Spinbox(trial_frame, from_=1, to=20, textvariable=self.trial_start_var, width=5, font=("Arial", 12)).pack(side="left", padx=2)
        tk.Label(trial_frame, text="〜", font=("Arial", 12)).pack(side="left", padx=2)
        tk.Spinbox(trial_frame, from_=1, to=20, textvariable=self.trial_end_var, width=5, font=("Arial", 12)).pack(side="left", padx=2)
        
        ip_frame = tk.Frame(udp_frame)
        ip_frame.pack(pady=10)
        tk.Label(ip_frame, text="Meta Quest 3 IP:", font=("Arial", 12)).pack(side="left", padx=5)
        self.ip_entry = tk.Entry(ip_frame, font=("Arial", 12), width=20)
        self.ip_entry.insert(0, "172.31.12.115")
        self.ip_entry.pack(side="left", padx=5)
        tk.Label(ip_frame, text="Port:", font=("Arial", 12)).pack(side="left", padx=5)
        self.port_entry = tk.Entry(ip_frame, font=("Arial", 12), width=8)
        self.port_entry.insert(0, "5005")
        self.port_entry.pack(side="left", padx=5)
        
        btn_frame_udp = tk.Frame(udp_frame)
        btn_frame_udp.pack(pady=15)
        self.send_btn = tk.Button(btn_frame_udp, text="UDP送信開始", font=("Arial", 14, "bold"), 
                                  command=self.on_send_udp, bg="#2196F3", fg="white", padx=30, pady=10)
        self.send_btn.pack()
        self.udp_status_label = tk.Label(udp_frame, text="", font=("Arial", 10), fg="blue")
        self.udp_status_label.pack(pady=5)
    
    def on_show(self):
        self.refresh_subjects()
    
    def refresh_subjects(self):
        data_dir = Path("data")
        if not data_dir.exists():
            self.subject_combo['values'] = []
            return
        subjects = []
        for subject_dir in data_dir.iterdir():
            if subject_dir.is_dir():
                metadata_path = subject_dir / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                            name = metadata.get('name', '')
                            subjects.append(f"{subject_dir.name} ({name})")
                    except:
                        subjects.append(subject_dir.name)
                else:
                    subjects.append(subject_dir.name)
        self.subject_combo['values'] = sorted(subjects)
        if subjects: self.subject_combo.current(0)
    
    def on_next(self):
        student_id = self.student_id_entry.get().strip()
        name = self.name_entry.get().strip()
        
        if not student_id or not name:
            self.controller.show_error("入力エラー", "学籍番号と氏名を入力してください")
            return
        
        if not student_id.isascii():
            self.controller.show_error("入力エラー", "学籍番号は半角英数字のみ使用してください。")
            return
        
        self.controller.subject_info = {
            "student_id": student_id,
            "name": name,
            "date": datetime.now().strftime("%Y-%m-%d")
        }
        
        self.controller.dm.create_subject_folder(student_id)
        self.controller.show_frame("CalibrationMethodScreen")

    def on_send_udp(self):
        if not self.subject_combo.get():
            messagebox.showerror("エラー", "被験者を選択してください")
            return
        selected = self.subject_combo.get()
        student_id = selected.split()[0]
        trial_start = self.trial_start_var.get()
        trial_end = self.trial_end_var.get()
        target_ip = self.ip_entry.get().strip()
        try:
            port = int(self.port_entry.get().strip())
        except ValueError:
            messagebox.showerror("エラー", "ポート番号が不正です")
            return
        
        if not messagebox.askyesno("確認", "UDP送信を開始しますか？"):
            return
        
        self.send_btn.config(state="disabled", text="送信中...")
        self.udp_status_label.config(text="送信中...", fg="orange")
        threading.Thread(target=self._send_udp_thread, args=(student_id, trial_start, trial_end, target_ip, port), daemon=True).start()

    def _send_udp_thread(self, student_id, start, end, ip, port):
        try:
            trial_numbers = list(range(start, end + 1))
            self._send_udp_data(student_id, trial_numbers, ip, port)
            self.controller.root.after(0, lambda: self.send_btn.config(state="normal", text="UDP送信開始"))
            self.controller.root.after(0, lambda: self.udp_status_label.config(text=f"✓ 送信完了", fg="green"))
            self.controller.root.after(0, lambda: messagebox.showinfo("完了", "データ送信が完了しました"))
        except Exception as e:
            self.controller.root.after(0, lambda: self.send_btn.config(state="normal", text="UDP送信開始"))
            self.controller.root.after(0, lambda: self.udp_status_label.config(text=f"✗ エラー", fg="red"))
            self.controller.root.after(0, lambda: messagebox.showerror("エラー", str(e)))

    def _send_udp_data(self, student_id, trial_numbers, target_ip, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        addr = (target_ip, port)
        def send_json(payload):
            sock.sendto(json.dumps(payload).encode('utf-8'), addr)
        
        subject_dir = get_subject_dir(student_id)
        send_json({"type": "session_start", "student_id": student_id, "total_trials": len(trial_numbers), "timestamp": time.time()})
        time.sleep(0.1)
        
        successful_trials = 0
        for trial_num in trial_numbers:
            csv_path = subject_dir / f"trial_{trial_num:02d}.csv"
            if not csv_path.exists(): continue
            send_json({"type": "trial_start", "student_id": student_id, "trial_number": trial_num})
            time.sleep(0.05)
            
            frame_count = 0
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        if "X_mm" in row:
                            x, y, z = float(row["X_mm"])/1000, float(row["Y_mm"])/1000, float(row["Z_mm"])/1000
                            vx, vy, vz, ax, ay, az = 0,0,0,0,0,0
                        else:
                            x, y, z = float(row.get("X_m", 0)), float(row.get("Y_m", 0)), float(row.get("Z_m", 0))
                            vx, vy, vz = float(row.get("VX_ms", 0)), float(row.get("VY_ms", 0)), float(row.get("VZ_ms", 0))
                            ax, ay, az = float(row.get("AX_ms2", 0)), float(row.get("AY_ms2", 0)), float(row.get("AZ_ms2", 0))
                        
                        frame_idx = int(row.get("frame_index", frame_count))
                        payload = {
                            "type": "trajectory_point", "student_id": student_id, "trial_number": trial_num,
                            "frame": frame_idx, "x": x, "y": y, "z": z, "vx": vx, "vy": vy, "vz": vz,
                            "ax": ax, "ay": ay, "az": az, "unit": "m", "frame_name": "aruco_world"
                        }
                        send_json(payload)
                        frame_count += 1
                        time.sleep(0.001)
                    except ValueError: continue
            
            send_json({"type": "trial_end", "student_id": student_id, "trial_number": trial_num, "frame_count": frame_count})
            time.sleep(0.1)
            successful_trials += 1
            self.controller.root.after(0, lambda i=successful_trials, n=len(trial_numbers): self.udp_status_label.config(text=f"送信中... ({i}/{n} 試行完了)"))
        
        send_json({"type": "session_end", "student_id": student_id, "trials_sent": successful_trials})
        sock.close()


class CalibrationMethodScreen(tk.Frame):
    """画面1.5: キャリブレーション方式選択"""
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        title = tk.Label(self, text="座標系キャリブレーション方式", font=("Arial", 24, "bold"))
        title.pack(pady=30)
        
        description = tk.Label(self, text="キャリブレーション方式を選択してください", font=("Arial", 12))
        description.pack(pady=10)
        
        # 選択フレーム
        select_frame = tk.Frame(self)
        select_frame.pack(pady=30, fill="x", padx=50)
        
        # バックボード方式（推奨）
        bb_frame = tk.LabelFrame(select_frame, text="バックボード4点法（推奨）", font=("Arial", 14, "bold"), padx=20, pady=20)
        bb_frame.pack(fill="x", pady=10)
        tk.Label(bb_frame, text="バックボードの角4点を両カメラで指定し、\n高精度な座標系を構築します。", 
                font=("Arial", 11), justify="left").pack(anchor="w")
        tk.Label(bb_frame, text="✓ 奥行き精度が高い\n✓ PnP誤差の影響を受けない", 
                font=("Arial", 10), fg="green", justify="left").pack(anchor="w", pady=5)
        self.bb_btn = tk.Button(bb_frame, text="バックボード4点法を使用", font=("Arial", 12),
                               command=self.on_select_backboard, bg="#4CAF50", fg="white", padx=20, pady=10)
        self.bb_btn.pack(pady=10)
        
        # ArUco方式（従来）
        aruco_frame = tk.LabelFrame(select_frame, text="ArUco PnP法（従来方式）", font=("Arial", 14), padx=20, pady=20)
        aruco_frame.pack(fill="x", pady=10)
        tk.Label(aruco_frame, text="ArUcoマーカーをカメラで検出し、\nPnPで座標系を推定します。", 
                font=("Arial", 11), justify="left").pack(anchor="w")
        tk.Label(aruco_frame, text="⚠ マーカー角度により奥行き精度が低下する場合あり", 
                font=("Arial", 10), fg="orange", justify="left").pack(anchor="w", pady=5)
        self.aruco_btn = tk.Button(aruco_frame, text="ArUco PnP法を使用", font=("Arial", 12),
                                  command=self.on_select_aruco, bg="#2196F3", fg="white", padx=20, pady=10)
        self.aruco_btn.pack(pady=10)
        
        # 戻るボタン
        back_btn = tk.Button(self, text="戻る", font=("Arial", 12), command=self.on_back, padx=15, pady=8)
        back_btn.pack(pady=20)
    
    def on_select_backboard(self):
        self.controller.show_frame("BackboardCalibrationScreen")
    
    def on_select_aruco(self):
        self.controller.show_frame("ArucoSetupScreen")
    
    def on_back(self):
        self.controller.show_frame("SubjectInputScreen")


class BackboardCalibrationScreen(tk.Frame):
    """バックボード4点指定によるキャリブレーション画面（ズーム機能付き修正版）"""
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        # 状態管理
        self.points_left = []   # [(x, y), ...] 最大4点 (画像座標)
        self.points_right = []  # [(x, y), ...] 最大4点 (画像座標)
        self.current_camera = 'left'
        self.calibration_done = False
        self.latest_frame_left = None
        self.latest_frame_right = None
        
        # 画像表示サイズ (GUI上のサイズ)
        self.display_width = 480
        self.display_height = 270
        
        # レイアウト
        title = tk.Label(self, text="バックボード4点キャリブレーション", font=("Arial", 20, "bold"))
        title.pack(pady=10)
        
        # 指示テキスト
        self.instruction_label = tk.Label(self, text="左カメラ画像でバックボードの4角を「大まかに」クリックしてください\n(クリック後に拡大画面が表示されます)", 
                                         font=("Arial", 12), fg="blue")
        self.instruction_label.pack(pady=5)
        
        # カメラビューフレーム
        camera_frame = tk.Frame(self)
        camera_frame.pack(pady=10)
        
        # 左カメラ
        left_frame = tk.LabelFrame(camera_frame, text="左カメラ (Camera 0)", font=("Arial", 11))
        left_frame.pack(side="left", padx=10)
        self.canvas_left = tk.Canvas(left_frame, width=self.display_width, height=self.display_height, bg="black")
        self.canvas_left.pack()
        self.canvas_left.bind("<Button-1>", self.on_canvas_left_click)
        self.points_left_label = tk.Label(left_frame, text="指定点: 0/4", font=("Arial", 10))
        self.points_left_label.pack()
        
        # 右カメラ
        right_frame = tk.LabelFrame(camera_frame, text="右カメラ (Camera 1)", font=("Arial", 11))
        right_frame.pack(side="left", padx=10)
        self.canvas_right = tk.Canvas(right_frame, width=self.display_width, height=self.display_height, bg="black")
        self.canvas_right.pack()
        self.canvas_right.bind("<Button-1>", self.on_canvas_right_click)
        self.points_right_label = tk.Label(right_frame, text="指定点: 0/4", font=("Arial", 10))
        self.points_right_label.pack()
        
        # ステータス
        self.status_label = tk.Label(self, text="状態: 点を指定してください", font=("Arial", 12), fg="orange")
        self.status_label.pack(pady=10)
        
        # 精度情報
        self.accuracy_label = tk.Label(self, text="", font=("Arial", 10), fg="gray")
        self.accuracy_label.pack()
        
        # ボタンフレーム
        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=15)
        
        self.reset_btn = tk.Button(btn_frame, text="リセット", font=("Arial", 12),
                                  command=self.on_reset, padx=15, pady=8)
        self.reset_btn.pack(side="left", padx=10)
        
        self.calibrate_btn = tk.Button(btn_frame, text="キャリブレーション実行", font=("Arial", 12),
                                      command=self.on_calibrate, bg="#FF9800", fg="white", padx=15, pady=8, state="disabled")
        self.calibrate_btn.pack(side="left", padx=10)
        
        self.next_btn = tk.Button(btn_frame, text="次へ (試行管理)", font=("Arial", 12),
                                 command=self.on_next, bg="#4CAF50", fg="white", padx=15, pady=8, state="disabled")
        self.next_btn.pack(side="left", padx=10)
        
        back_btn = tk.Button(btn_frame, text="戻る", font=("Arial", 12), command=self.on_back, padx=15, pady=8)
        back_btn.pack(side="left", padx=10)
        
        # 点の色
        self.point_colors = ['red', 'blue', 'green', 'purple']
        self.point_labels = ['左上', '右上', '右下', '左下']
    
    def on_show(self):
        """画面表示時の初期化"""
        # カメラ初期化
        if self.controller.cap0 is None:
            c0 = self.controller.config.get("camera0", 0)
            c1 = self.controller.config.get("camera1", 1)
            self.controller.cap0 = cv2.VideoCapture(c0)
            self.controller.cap1 = cv2.VideoCapture(c1)
            self.controller.cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.controller.cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.controller.cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.controller.cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.controller.cap0.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.controller.cap1.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # 状態リセット
        self.on_reset()
        
        # プレビュー開始
        self.controller.preview_running = True
        self.update_preview()
    
    def update_preview(self):
        """両カメラのプレビュー更新"""
        if not self.controller.preview_running:
            return
        
        # 左カメラ
        ret0, frame0 = self.controller.cap0.read()
        if ret0:
            self.latest_frame_left = frame0.copy()
            self.draw_frame_with_points(self.canvas_left, frame0, self.points_left)
        
        # 右カメラ
        ret1, frame1 = self.controller.cap1.read()
        if ret1:
            self.latest_frame_right = frame1.copy()
            self.draw_frame_with_points(self.canvas_right, frame1, self.points_right)
        
        self.after(50, self.update_preview)
    
    def draw_frame_with_points(self, canvas, frame, points):
        """フレームと指定点を描画"""
        frame_resized = cv2.resize(frame, (self.display_width, self.display_height))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # 点を描画
        for i, (x, y) in enumerate(points):
            # 画像座標を表示座標に変換
            display_x = int(x * self.display_width / 1920)
            display_y = int(y * self.display_height / 1080)
            
            # OpenCVで円を描画
            color_bgr = {'red': (0, 0, 255), 'blue': (255, 0, 0), 'green': (0, 255, 0), 'purple': (128, 0, 128)}
            cv2.circle(frame_rgb, (display_x, display_y), 8, color_bgr[self.point_colors[i]], -1)
            cv2.circle(frame_rgb, (display_x, display_y), 10, (255, 255, 255), 2)
            
            # ラベル
            cv2.putText(frame_rgb, self.point_labels[i], (display_x + 12, display_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 点を線で結ぶ
        if len(points) >= 2:
            for i in range(len(points)):
                if i < len(points) - 1:
                    p1 = (int(points[i][0] * self.display_width / 1920), 
                          int(points[i][1] * self.display_height / 1080))
                    p2 = (int(points[i+1][0] * self.display_width / 1920), 
                          int(points[i+1][1] * self.display_height / 1080))
                    cv2.line(frame_rgb, p1, p2, (255, 255, 0), 2)
            # 4点目まで指定されたら閉じる
            if len(points) == 4:
                p1 = (int(points[3][0] * self.display_width / 1920), 
                      int(points[3][1] * self.display_height / 1080))
                p2 = (int(points[0][0] * self.display_width / 1920), 
                      int(points[0][1] * self.display_height / 1080))
                cv2.line(frame_rgb, p1, p2, (255, 255, 0), 2)
        
        # Tkinter画像に変換
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        canvas.imgtk = imgtk
        canvas.create_image(0, 0, anchor="nw", image=imgtk)
    
    def on_canvas_left_click(self, event):
        """左カメラキャンバスクリック（ズームウィンドウ呼び出し）"""
        if len(self.points_left) >= 4 or self.latest_frame_left is None:
            return
        
        # 表示座標を画像座標(1920x1080)に変換（大まかな位置）
        rough_x = int(event.x * 1920 / self.display_width)
        rough_y = int(event.y * 1080 / self.display_height)
        
        self.open_zoom_window(self.latest_frame_left.copy(), rough_x, rough_y, 'left')
    
    def on_canvas_right_click(self, event):
        """右カメラキャンバスクリック（ズームウィンドウ呼び出し）"""
        if len(self.points_right) >= 4 or self.latest_frame_right is None:
            return
        
        # 表示座標を画像座標(1920x1080)に変換（大まかな位置）
        rough_x = int(event.x * 1920 / self.display_width)
        rough_y = int(event.y * 1080 / self.display_height)
        
        self.open_zoom_window(self.latest_frame_right.copy(), rough_x, rough_y, 'right')

    def open_zoom_window(self, frame, center_x, center_y, side):
        """拡大ウィンドウを表示して正確な点を取得する"""
        zoom_win = tk.Toplevel(self)
        zoom_win.title("位置の微調整 (クリックして確定)")
        zoom_win.grab_set() # モーダルにする
        
        # 切り出しサイズ (元画像上でのピクセル数)
        crop_size = 400
        half_crop = crop_size // 2
        
        # 切り出し範囲の計算 (画像の端にはみ出さないように)
        img_h, img_w = frame.shape[:2]
        x1 = max(0, min(center_x - half_crop, img_w - crop_size))
        y1 = max(0, min(center_y - half_crop, img_h - crop_size))
        x2 = x1 + crop_size
        y2 = y1 + crop_size
        
        # 切り出し
        crop_img = frame[y1:y2, x1:x2]
        
        # 表示倍率 (1.5倍に拡大して見やすくする)
        scale = 1.5 
        display_size = int(crop_size * scale)
        # 最近傍補間でリサイズしてピクセルが見えやすくする
        crop_resized = cv2.resize(crop_img, (display_size, display_size), interpolation=cv2.INTER_NEAREST)
        
        # 表示
        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(crop_rgb))
        
        canvas = tk.Canvas(zoom_win, width=display_size, height=display_size, cursor="cross")
        canvas.pack()
        canvas.create_image(0, 0, anchor="nw", image=img_tk)
        
        # ガイド線（中心十字）- 中心をクリックしやすくするガイド
        center = display_size // 2
        canvas.create_line(center, 0, center, display_size, fill="cyan", dash=(4, 4))
        canvas.create_line(0, center, display_size, center, fill="cyan", dash=(4, 4))
        
        # 使い方ラベル
        tk.Label(zoom_win, text="正確な位置をクリックしてください", font=("Arial", 10), bg="yellow").pack(fill="x")

        def on_zoom_click(event):
            # 拡大画像上のクリック位置
            click_x = event.x
            click_y = event.y
            
            # 元画像(1920x1080)上の座標に戻す
            # 1. 倍率を戻す
            orig_crop_x = click_x / scale
            orig_crop_y = click_y / scale
            # 2. 切り出しオフセットを足す
            final_x = int(x1 + orig_crop_x)
            final_y = int(y1 + orig_crop_y)
            
            # 範囲外チェック
            final_x = max(0, min(final_x, img_w - 1))
            final_y = max(0, min(final_y, img_h - 1))
            
            # 登録処理
            if side == 'left':
                self.points_left.append((final_x, final_y))
                self.points_left_label.config(text=f"指定点: {len(self.points_left)}/4")
            else:
                self.points_right.append((final_x, final_y))
                self.points_right_label.config(text=f"指定点: {len(self.points_right)}/4")
            
            self.update_instruction()
            self.check_ready_for_calibration()
            zoom_win.destroy()

        canvas.bind("<Button-1>", on_zoom_click)
        
        # ウィンドウが閉じられるまで待機しない（mainloop内なのでcallbackで処理）
        # 画像参照を保持しておく必要がある
        canvas.image = img_tk
    
    def update_instruction(self):
        """指示テキストを更新"""
        left_count = len(self.points_left)
        right_count = len(self.points_right)
        
        if left_count < 4:
            next_point = self.point_labels[left_count]
            self.instruction_label.config(
                text=f"左カメラ: バックボードの「{next_point}」をクリックしてください\n(指定済み: {left_count}/4)",
                fg="blue")
        elif right_count < 4:
            next_point = self.point_labels[right_count]
            self.instruction_label.config(
                text=f"右カメラ: バックボードの「{next_point}」をクリックしてください\n(指定済み: {right_count}/4)",
                fg="green")
        else:
            self.instruction_label.config(
                text="すべての点が指定されました。「キャリブレーション実行」を押してください。",
                fg="purple")
    
    def check_ready_for_calibration(self):
        """キャリブレーション実行可能かチェック"""
        if len(self.points_left) == 4 and len(self.points_right) == 4:
            self.calibrate_btn.config(state="normal")
            self.status_label.config(text="状態: キャリブレーション実行可能", fg="green")
        else:
            self.calibrate_btn.config(state="disabled")
    
    def on_reset(self):
        """点のリセット"""
        self.points_left = []
        self.points_right = []
        self.calibration_done = False
        
        self.points_left_label.config(text="指定点: 0/4")
        self.points_right_label.config(text="指定点: 0/4")
        self.status_label.config(text="状態: 点を指定してください", fg="orange")
        self.accuracy_label.config(text="")
        self.instruction_label.config(
            text="左カメラ画像でバックボードの4角を「大まかに」クリックしてください\n(クリック後に拡大画面が表示されます)",
            fg="blue")
        
        self.calibrate_btn.config(state="disabled")
        self.next_btn.config(state="disabled")
    
    def on_calibrate(self):
        """キャリブレーション実行"""
        if self.controller.eng is None:
            messagebox.showerror("エラー", "MATLABエンジンが起動していません")
            return
        
        if len(self.points_left) != 4 or len(self.points_right) != 4:
            messagebox.showerror("エラー", "4点すべてを指定してください")
            return
        
        self.status_label.config(text="状態: キャリブレーション実行中...", fg="orange")
        self.calibrate_btn.config(state="disabled")
        self.controller.root.update()
        
        try:
            # 点データをMATLAB形式に変換
            import matlab
            points_left_matlab = matlab.double(self.points_left)
            points_right_matlab = matlab.double(self.points_right)
            
            param_file = str(Path("params/stereoParams.mat").resolve())
            pose_file = str(Path("params/marker_pose.mat").resolve())
            
            # MATLAB関数呼び出し
            success, calib_info = self.controller.eng.calibrate_from_backboard(
                points_left_matlab, points_right_matlab, param_file, pose_file, nargout=2)
            
            if success:
                # 精度情報を表示
                try:
                    width_err = float(calib_info['width_error'])
                    height_err = float(calib_info['height_error'])
                    measured_w = float(calib_info['measured_width'])
                    measured_h = float(calib_info['measured_height'])
                    
                    self.accuracy_label.config(
                        text=f"計測サイズ: {measured_w:.0f} × {measured_h:.0f} mm (誤差: 幅{width_err:.0f}mm, 高さ{height_err:.0f}mm)",
                        fg="green" if width_err < 50 and height_err < 50 else "orange")
                except:
                    pass
                
                self.status_label.config(text="状態: キャリブレーション完了", fg="green")
                self.calibration_done = True
                self.next_btn.config(state="normal")
                messagebox.showinfo("成功", "キャリブレーションが完了しました。\nmarker_pose.mat を保存しました。")
            else:
                self.status_label.config(text="状態: キャリブレーション失敗", fg="red")
                self.calibrate_btn.config(state="normal")
                messagebox.showerror("失敗", "キャリブレーションに失敗しました。\n点の指定を確認してください。")
                
        except Exception as e:
            self.status_label.config(text="状態: エラー", fg="red")
            self.calibrate_btn.config(state="normal")
            messagebox.showerror("MATLABエラー", str(e))
    
    def on_next(self):
        """次の画面へ"""
        self.controller.preview_running = False
        self.controller.show_frame("TrialManagementScreen")
    
    def on_back(self):
        """前の画面へ"""
        self.controller.preview_running = False
        self.controller.show_frame("CalibrationMethodScreen")


class ArucoSetupScreen(tk.Frame):
    """画面2: ArUco設定 (MATLAB呼び出し)"""
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        title = tk.Label(self, text="ArUco原点設定", font=("Arial", 24, "bold"))
        title.pack(pady=20)
        instructions = tk.Label(self, text="ArUcoマーカー（ID=0）をカメラ0に映し、\n'ArUco検出実行' ボタンを押してください。", font=("Arial", 12))
        instructions.pack(pady=10)
        self.preview_label = tk.Label(self, bg="black")
        self.preview_label.pack(pady=20)
        self.status_label = tk.Label(self, text="状態: 未設定", font=("Arial", 14), fg="red")
        self.status_label.pack(pady=10)
        
        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=20)
        self.setup_btn = tk.Button(btn_frame, text="ArUco検出実行", font=("Arial", 14), 
                                   command=self.on_setup_aruco, bg="#2196F3", fg="white", padx=20, pady=10)
        self.setup_btn.pack(side="left", padx=10)
        self.next_btn = tk.Button(btn_frame, text="次へ (試行管理)", font=("Arial", 14), 
                                 command=self.on_next, bg="#4CAF50", fg="white", padx=20, pady=10, state="disabled")
        self.next_btn.pack(side="left", padx=10)
        back_btn = tk.Button(btn_frame, text="戻る", font=("Arial", 12), command=self.on_back, padx=15, pady=8)
        back_btn.pack(side="left", padx=10)
        
    def on_show(self):
        if self.controller.cap0 is None:
            c0 = self.controller.config.get("camera0", 0)
            c1 = self.controller.config.get("camera1", 1)
            self.controller.cap0 = cv2.VideoCapture(c0)
            self.controller.cap1 = cv2.VideoCapture(c1)
            self.controller.cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.controller.cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.controller.cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.controller.cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.controller.cap0.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.controller.cap1.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.controller.preview_running = True
        self.update_preview()
    
    def update_preview(self):
        if not self.controller.preview_running: return
        ret, frame = self.controller.cap0.read()
        if ret:
            self.latest_frame = frame
            frame_resized = cv2.resize(frame, (640, 360))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.preview_label.imgtk = imgtk
            self.preview_label.configure(image=imgtk)
        self.after(30, self.update_preview)
    
    def on_setup_aruco(self):
        if self.controller.eng is None:
            messagebox.showerror("エラー", "MATLABエンジンが起動していません")
            return
        if not hasattr(self, 'latest_frame') or self.latest_frame is None:
            return

        self.controller.preview_running = False
        time.sleep(0.2) 

        temp_img_path = str(Path("temp_aruco_setup.jpg").resolve())
        try:
            is_success, buffer = cv2.imencode(".jpg", self.latest_frame)
            if is_success:
                with open(temp_img_path, "wb") as f: buffer.tofile(f)
            else:
                raise Exception("Encode failed")
        except Exception as e:
             self.controller.preview_running = True
             self.update_preview()
             messagebox.showerror("エラー", f"画像の保存に失敗: {e}")
             return
        
        param_file = str(Path("params/stereoParams.mat").resolve())
        pose_file = str(Path("params/marker_pose.mat").resolve())
        self.status_label.config(text="状態: MATLABで計算中...", fg="orange")
        self.controller.root.update()
        
        try:
            success = self.controller.eng.detect_aruco_pose(temp_img_path, param_file, pose_file, nargout=1)
            
            if success:
                self.status_label.config(text="状態: 設定完了", fg="green")
                self.setup_btn.config(state="disabled")
                self.next_btn.config(state="normal")
                messagebox.showinfo("成功", "ArUco座標を保存しました")
            else:
                self.status_label.config(text="状態: 検出失敗", fg="red")
                messagebox.showerror("失敗", "ArUcoマーカーが見つかりませんでした")
        except Exception as e:
            self.status_label.config(text="状態: エラー", fg="red")
            messagebox.showerror("MATLABエラー", str(e))
        finally:
            self.controller.preview_running = True
            self.update_preview()
    
    def on_next(self):
        self.controller.preview_running = False
        self.controller.show_frame("TrialManagementScreen")
    
    def on_back(self):
        self.controller.preview_running = False
        self.controller.show_frame("CalibrationMethodScreen")


class TrialManagementScreen(tk.Frame):
    """画面3: 試行管理 (同期撮影修正 + 事前テスト終了通知 + 続きから再開)"""
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        title = tk.Label(self, text="試行管理", font=("Arial", 20, "bold"))
        title.pack(pady=10)
        
        main_container = tk.Frame(self)
        main_container.pack(fill="both", expand=True, padx=20, pady=10)
        left_frame = tk.Frame(main_container)
        left_frame.pack(side="left", fill="both", expand=True)
        self.preview_label = tk.Label(left_frame, bg="black")
        self.preview_label.pack(pady=10)
        right_frame = tk.Frame(main_container, width=300)
        right_frame.pack(side="right", fill="y", padx=10)
        
        info_frame = tk.LabelFrame(right_frame, text="試行情報", font=("Arial", 12))
        info_frame.pack(fill="x", pady=10)
        self.trial_label = tk.Label(info_frame, text=f"試行: 1/{DEFAULT_TRIAL_COUNT}", font=("Arial", 14))
        self.trial_label.pack(pady=5)
        self.status_label = tk.Label(info_frame, text="状態: 待機中", font=("Arial", 12), fg="orange")
        self.status_label.pack(pady=5)
        self.progress = ttk.Progressbar(info_frame, length=250, mode="determinate")
        self.progress.pack(pady=5)
        
        control_frame = tk.LabelFrame(right_frame, text="トラッキング制御", font=("Arial", 12))
        control_frame.pack(fill="x", pady=10)
        self.start_btn = tk.Button(control_frame, text="開始 (S)", font=("Arial", 12), 
                                   command=self.on_start_recording, bg="#4CAF50", fg="white", padx=15, pady=8)
        self.start_btn.pack(pady=5)
        self.end_btn = tk.Button(control_frame, text="終了 (E)", font=("Arial", 12), 
                                command=self.on_end_recording, bg="#f44336", fg="white", padx=15, pady=8, state="disabled")
        self.end_btn.pack(pady=5)
        
        history_frame = tk.LabelFrame(right_frame, text="試行履歴", font=("Arial", 12))
        history_frame.pack(fill="both", expand=True, pady=10)
        self.history_listbox = tk.Listbox(history_frame, font=("Arial", 10))
        self.history_listbox.pack(fill="both", expand=True)
        
        self.processing = False
    
    def on_show(self):
        # --- 修正: 既存のデータを確認して続きから開始するロジック ---
        student_id = self.controller.subject_info.get("student_id")
        start_trial = 1
        
        if student_id:
            try:
                # メタデータを読み込んで完了済みの試行数を取得
                metadata = self.controller.dm.load_metadata(student_id)
                completed = metadata.get("trials_completed", 0)
                
                # 次の試行番号 = 完了数 + 1
                start_trial = completed + 1
                
                # もし完了済み試行が履歴リストになければ追加（画面復帰時の表示用）
                if self.history_listbox.size() == 0:
                    for t in metadata.get("trials", []):
                        num = t.get("trial_number")
                        res = t.get("result", "")
                        mark = "✓" if res == "success" else "✗"
                        self.history_listbox.insert(tk.END, f"試行{num:02d}: {mark} {res}")

            except FileNotFoundError:
                # まだデータがない場合は1からスタート
                start_trial = 1
        
        self.controller.current_trial = start_trial
        # -------------------------------------------------------

        self.update_trial_info()
        self.controller.preview_running = True
        self.update_preview()
        if self.controller.eng is None and MATLAB_AVAILABLE:
            messagebox.showwarning("警告", "MATLABエンジンが起動していません")
    
    def update_trial_info(self):
        self.trial_label.config(text=f"試行: {self.controller.current_trial}/{DEFAULT_TRIAL_COUNT}")
        self.progress["value"] = (self.controller.current_trial - 1) / DEFAULT_TRIAL_COUNT * 100
    
    def update_preview(self):
        if not self.controller.preview_running: return
        frame = None
        if self.controller.recording and self.controller.latest_preview_frame is not None:
            frame = self.controller.latest_preview_frame
        else:
            self.controller.cap0.grab()
            ret, frame = self.controller.cap0.retrieve()
            if not ret: frame = None
        
        if frame is not None:
            frame_resized = cv2.resize(frame, (640, 360))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.preview_label.imgtk = imgtk
            self.preview_label.configure(image=imgtk)
        self.after(30, self.update_preview)
    
    def on_start_recording(self):
        student_id = self.controller.subject_info.get("student_id", "UNKNOWN")
        subject_dir = get_subject_dir(student_id)
        trial_num = self.controller.current_trial
        
        self.controller.current_video0_path = str((subject_dir / f"trial_{trial_num:02d}_cam0.mp4").resolve())
        self.controller.current_video1_path = str((subject_dir / f"trial_{trial_num:02d}_cam1.mp4").resolve())
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = self.controller.config.get("fps", 30.0)
        w = int(self.controller.cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.controller.cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.controller.video_writer0 = cv2.VideoWriter(self.controller.current_video0_path, fourcc, fps, (w, h))
        self.controller.video_writer1 = cv2.VideoWriter(self.controller.current_video1_path, fourcc, fps, (w, h))
        
        self.controller.recording = True
        self.status_label.config(text="状態: 録画中...", fg="green")
        self.start_btn.config(state="disabled")
        self.end_btn.config(state="normal")
        
        self.controller.recording_thread = threading.Thread(target=self.recording_loop, daemon=True)
        self.controller.recording_thread.start()
    
    def recording_loop(self):
        """同期ずれを最小化する録画ループ"""
        while self.controller.recording:
            self.controller.cap0.grab()
            self.controller.cap1.grab()
            
            ret0, frame0 = self.controller.cap0.retrieve()
            ret1, frame1 = self.controller.cap1.retrieve()
            
            if not ret0 or not ret1: continue
            
            self.controller.latest_preview_frame = frame0.copy()
            self.controller.video_writer0.write(frame0)
            self.controller.video_writer1.write(frame1)
            time.sleep(0.001)
    
    def on_end_recording(self):
        self.controller.recording = False
        if self.controller.recording_thread:
            self.controller.recording_thread.join(timeout=2.0)
        if self.controller.video_writer0: self.controller.video_writer0.release()
        if self.controller.video_writer1: self.controller.video_writer1.release()
        
        self.controller.preview_running = False
        time.sleep(0.5)
        
        self.status_label.config(text="状態: MATLAB処理中(YOLO)...", fg="blue")
        self.processing = True
        threading.Thread(target=self.run_matlab_processing, daemon=True).start()
    
    def run_matlab_processing(self):
        student_id = self.controller.subject_info.get("student_id", "UNKNOWN")
        subject_dir = get_subject_dir(student_id)
        trial_num = self.controller.current_trial
        
        video0 = self.controller.current_video0_path
        video1 = self.controller.current_video1_path
        output_csv = str((subject_dir / f"trial_{trial_num:02d}.csv").resolve())
        param_file = str(Path("params/stereoParams.mat").resolve())
        pose_file = str(Path("params/marker_pose.mat").resolve()) 
        
        success = False
        error_msg = ""
        
        if not Path(video0).exists() or Path(video0).stat().st_size == 0:
            self.controller.root.after(0, lambda: self.finish_processing(False, "", "", "", "録画失敗"))
            return

        if self.controller.eng is not None:
            try:
                # YOLOトラッキング関数を実行
                self.controller.eng.run_tracking_func(video0, video1, output_csv, param_file, pose_file, nargout=0)
                success = True
            except Exception as e:
                error_msg = str(e)
        else:
            error_msg = "Engine not started"
        
        self.controller.root.after(0, lambda: self.finish_processing(success, output_csv, video0, video1, error_msg))

    def finish_processing(self, success, output_csv, video0, video1, error_msg):
        self.processing = False
        
        self.controller.preview_running = True
        self.update_preview()

        if not success:
            messagebox.showerror("MATLABエラー", error_msg)
            self.status_label.config(text="状態: エラー", fg="red")
            self.start_btn.config(state="normal")
            self.end_btn.config(state="disabled")
            return
            
        result = self.show_result_dialog()
        total_frames = 0
        if os.path.exists(output_csv):
             with open(output_csv, 'r') as f: total_frames = sum(1 for line in f) - 1
        
        trial_info = {
            "trial_number": self.controller.current_trial, "result": result,
            "timestamp": datetime.now().isoformat(), "csv_file": os.path.basename(output_csv),
            "video0_file": video0, "video1_file": video1, "total_frames": total_frames
        }
        self.controller.trials.append(trial_info)
        self.controller.dm.add_trial_to_metadata(self.controller.subject_info["student_id"], trial_info)
        
        result_mark = "✓" if result == "success" else "✗"
        self.history_listbox.insert(tk.END, f"試行{self.controller.current_trial:02d}: {result_mark} {result}")
        
        if self.controller.current_trial < DEFAULT_TRIAL_COUNT:
            self.controller.current_trial += 1
            self.update_trial_info()
            self.status_label.config(text="状態: 待機中", fg="orange")
            self.start_btn.config(state="normal")
            self.end_btn.config(state="disabled")

            # --- 追加: 事前テスト(10本)終了時の通知 ---
            if self.controller.current_trial == 11:
                messagebox.showinfo("事前テスト終了", 
                    "事前テスト（10本）が終了しました。\n"
                    "被験者にフィードバック（MR/PC）を行ってください。\n\n"
                    "※準備ができたら、そのまま事後テスト（試行11〜）を開始してください。")
            # ---------------------------------------

        else:
            self.controller.show_frame("CompletionScreen")

    def show_result_dialog(self):
        dialog = tk.Toplevel(self)
        dialog.title("試行結果")
        dialog.geometry("300x200")
        dialog.transient(self)
        dialog.grab_set()
        result_var = tk.StringVar(value="success")
        tk.Label(dialog, text="試行結果を選択してください", font=("Arial", 12)).pack(pady=20)
        tk.Radiobutton(dialog, text="成功", variable=result_var, value="success", font=("Arial", 11)).pack()
        tk.Radiobutton(dialog, text="失敗", variable=result_var, value="failure", font=("Arial", 11)).pack()
        tk.Radiobutton(dialog, text="スキップ", variable=result_var, value="skip", font=("Arial", 11)).pack()
        def on_ok(): dialog.destroy()
        tk.Button(dialog, text="OK", command=on_ok, font=("Arial", 12), padx=20, pady=5).pack(pady=20)
        dialog.wait_window()
        return result_var.get()


class CompletionScreen(tk.Frame):
    """画面4: 完了画面"""
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        title = tk.Label(self, text="実験完了", font=("Arial", 28, "bold"), fg="green")
        title.pack(pady=50)
        self.message_label = tk.Label(self, text="", font=("Arial", 16))
        self.message_label.pack(pady=20)
        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=30)
        finish_btn = tk.Button(btn_frame, text="終了", font=("Arial", 14), 
                              command=self.controller.on_closing, bg="#f44336", fg="white", padx=30, pady=10)
        finish_btn.pack(side="left", padx=10)

    def on_show(self):
        name = self.controller.subject_info.get("name", "不明")
        self.message_label.config(text=f"{name} さん\nお疲れ様でした！")

if __name__ == "__main__":
    root = tk.Tk()
    app = ExperimentRecordProcessGUI(root)
    root.mainloop()