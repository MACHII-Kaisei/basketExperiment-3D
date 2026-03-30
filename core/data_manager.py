"""
Data Manager Module

データ保存、読み込み、フォルダ管理を行うモジュール
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd

from core.experiment_config import (
    get_subject_dir,
    get_trial_csv_path,
    get_metadata_path,
    CSV_COLUMNS
)


class DataManager:
    """実験データの保存・読み込みを管理するクラス"""
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        DataManagerインスタンスを初期化
        
        Args:
            base_dir: ベースディレクトリ（テスト用、Noneの場合はデフォルトの"data"）
        """
        self.base_dir = Path(base_dir) if base_dir else None
    
    def _get_subject_dir(self, student_id: str) -> Path:
        """被験者ディレクトリパスを取得（base_dir考慮）"""
        if self.base_dir:
            return self.base_dir / student_id
        else:
            return get_subject_dir(student_id)
    
    def _get_trial_csv_path(self, student_id: str, trial_number: int) -> Path:
        """試行CSVパスを取得（base_dir考慮）"""
        if self.base_dir:
            filename = f"trial_{trial_number:02d}.csv"
            return self.base_dir / student_id / filename
        else:
            return get_trial_csv_path(student_id, trial_number)
    
    def _get_metadata_path(self, student_id: str) -> Path:
        """メタデータパスを取得（base_dir考慮）"""
        if self.base_dir:
            return self.base_dir / student_id / "metadata.json"
        else:
            return get_metadata_path(student_id)
    
    def create_subject_folder(self, student_id: str) -> Path:
        """
        被験者用のフォルダを作成
        
        Args:
            student_id: 学籍番号
            
        Returns:
            Path: 作成されたフォルダパス
            
        Raises:
            OSError: フォルダ作成失敗
            
        Example:
            >>> dm = DataManager()
            >>> path = dm.create_subject_folder("B2230123")
            >>> # -> data/B2230123/ が作成される
        """
        subject_dir = self._get_subject_dir(student_id)
        
        try:
            subject_dir.mkdir(parents=True, exist_ok=True)
            return subject_dir
        except OSError as e:
            raise OSError(f"Failed to create folder {subject_dir}: {e}") from e
    
    def save_trial_csv(
        self,
        student_id: str,
        trial_number: int,
        data: pd.DataFrame
    ) -> Path:
        """
        試行データをCSVファイルに保存
        
        Args:
            student_id: 学籍番号
            trial_number: 試行番号（1〜10）
            data: トラッキングデータ（DataFrame形式）
            
        Returns:
            Path: 保存されたCSVファイルパス
            
        Raises:
            ValueError: trial_numberが範囲外、またはデータフォーマットが不正
            IOError: ファイル保存失敗
            
        Example:
            >>> dm = DataManager()
            >>> df = pd.DataFrame({...})
            >>> path = dm.save_trial_csv("B2230123", 1, df)
        """
        # データフォーマット検証
        if data.empty:
            raise ValueError("Data is empty")
        
        missing_columns = set(CSV_COLUMNS) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns}")
        
        # パス取得
        csv_path = self._get_trial_csv_path(student_id, trial_number)
        
        # フォルダ作成（存在しない場合）
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # CSV保存（列の順序を保証）
            data[CSV_COLUMNS].to_csv(csv_path, index=False, encoding='utf-8')
            return csv_path
        except IOError as e:
            raise IOError(f"Failed to save CSV to {csv_path}: {e}") from e
    
    def save_metadata(
        self,
        student_id: str,
        name: str,
        date: str,
        trials: List[Dict],
        statistics: Optional[Dict] = None
    ) -> Path:
        """
        被験者のメタデータをJSONファイルに保存
        
        Args:
            student_id: 学籍番号
            name: 氏名
            date: 実験日（YYYY-MM-DD形式）
            trials: 試行情報のリスト
            statistics: 統計情報（オプション）
            
        Returns:
            Path: 保存されたJSONファイルパス
            
        Example:
            >>> trials = [
            ...     {
            ...         "trial_number": 1,
            ...         "result": "success",
            ...         "timestamp": "2025-11-18T14:30:15",
            ...         "csv_file": "trial_01.csv"
            ...     }
            ... ]
            >>> statistics = {
            ...     "success_count": 7,
            ...     "failure_count": 3,
            ...     "success_rate": 0.70
            ... }
            >>> dm.save_metadata("B2230123", "山田太郎", "2025-11-18", trials, statistics)
        """
        metadata = {
            "student_id": student_id,
            "name": name,
            "date": date,
            "trials_completed": len(trials),
            "start_time": trials[0]["timestamp"] if trials else None,
            "end_time": trials[-1]["timestamp"] if trials else None,
            "trials": trials
        }
        
        if statistics:
            metadata["statistics"] = statistics
        
        metadata_path = self._get_metadata_path(student_id)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            return metadata_path
        except IOError as e:
            raise IOError(f"Failed to save metadata to {metadata_path}: {e}") from e
    
    def load_metadata(self, student_id: str) -> Dict:
        """
        メタデータを読み込み
        
        Args:
            student_id: 学籍番号
            
        Returns:
            dict: メタデータ
            
        Raises:
            FileNotFoundError: metadata.jsonが存在しない
            json.JSONDecodeError: JSON形式が不正
            
        Example:
            >>> dm = DataManager()
            >>> metadata = dm.load_metadata("B2230123")
        """
        metadata_path = self._get_metadata_path(student_id)
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON format in {metadata_path}",
                e.doc, e.pos
            ) from e
    
    def add_trial_to_metadata(
        self,
        student_id: str,
        trial_info: Dict
    ) -> None:
        """
        既存のメタデータに試行情報を追加
        
        Args:
            student_id: 学籍番号
            trial_info: 追加する試行情報
            
        Example:
            >>> trial_info = {
            ...     "trial_number": 2,
            ...     "result": "failure",
            ...     "timestamp": "2025-11-18T14:32:00",
            ...     "csv_file": "trial_02.csv"
            ... }
            >>> dm.add_trial_to_metadata("B2230123", trial_info)
        """
        try:
            metadata = self.load_metadata(student_id)
        except FileNotFoundError:
            # メタデータが存在しない場合は新規作成
            metadata = {
                "student_id": student_id,
                "name": "",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "trials": []
            }
        
        # 試行を追加
        metadata["trials"].append(trial_info)
        metadata["trials_completed"] = len(metadata["trials"])
        
        # start_time/end_timeを更新
        if metadata["trials"]:
            metadata["start_time"] = metadata["trials"][0]["timestamp"]
            metadata["end_time"] = metadata["trials"][-1]["timestamp"]
        
        # 保存
        metadata_path = self._get_metadata_path(student_id)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def update_statistics(self, student_id: str) -> None:
        """
        統計情報を再計算してメタデータに保存
        
        Args:
            student_id: 学籍番号
            
        処理内容:
            - trials配列から成功/失敗をカウント
            - 成功率を計算
            - metadata.jsonを更新
            
        Example:
            >>> dm = DataManager()
            >>> dm.update_statistics("B2230123")
        """
        metadata = self.load_metadata(student_id)
        
        # 統計計算
        success_count = 0
        failure_count = 0
        
        for trial in metadata.get("trials", []):
            result = trial.get("result", "skip")
            if result == "success":
                success_count += 1
            elif result == "failure":
                failure_count += 1
        
        total_count = success_count + failure_count
        success_rate = success_count / total_count if total_count > 0 else 0.0
        
        metadata["statistics"] = {
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate": round(success_rate, 2)
        }
        
        # 保存
        metadata_path = self._get_metadata_path(student_id)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # テスト実行
    print("=== data_manager.py テスト ===\n")
    
    dm = DataManager()
    test_id = "TEST001"
    
    # 1. フォルダ作成テスト
    print("1. create_subject_folder テスト")
    folder = dm.create_subject_folder(test_id)
    print(f"   作成: {folder}")
    assert folder.exists(), "フォルダが作成されていません"
    print("   ✓ フォルダ作成成功\n")
    
    # 2. CSV保存テスト
    print("2. save_trial_csv テスト")
    test_data = pd.DataFrame({
        "frame_index": [0, 1, 2],
        "timestamp_sec": [0.0, 0.033, 0.066],
        "X_m": [0.1, 0.2, 0.3],
        "Y_m": [0.5, 0.6, 0.7],
        "Z_m": [1.0, 1.1, 1.2],
        "VX_ms": [0.01, 0.02, 0.03],
        "VY_ms": [0.04, 0.05, 0.06],
        "VZ_ms": [0.07, 0.08, 0.09],
        "AX_ms2": [0.001, 0.002, 0.003],
        "AY_ms2": [0.004, 0.005, 0.006],
        "AZ_ms2": [0.007, 0.008, 0.009],
        "is_predicted": [0, 0, 1]
    })
    
    csv_path = dm.save_trial_csv(test_id, 1, test_data)
    print(f"   保存: {csv_path}")
    assert csv_path.exists(), "CSVファイルが作成されていません"
    
    # 読み込んで確認
    loaded_data = pd.read_csv(csv_path)
    assert len(loaded_data) == 3, "行数が一致しません"
    print("   ✓ CSV保存成功\n")
    
    # 3. メタデータ保存テスト
    print("3. save_metadata テスト")
    trials = [
        {
            "trial_number": 1,
            "result": "success",
            "timestamp": "2025-11-18T14:30:15",
            "csv_file": "trial_01.csv"
        }
    ]
    statistics = {
        "success_count": 1,
        "failure_count": 0,
        "success_rate": 1.0
    }
    
    metadata_path = dm.save_metadata(test_id, "テスト太郎", "2025-11-18", trials, statistics)
    print(f"   保存: {metadata_path}")
    assert metadata_path.exists(), "メタデータファイルが作成されていません"
    print("   ✓ メタデータ保存成功\n")
    
    # 4. メタデータ読み込みテスト
    print("4. load_metadata テスト")
    loaded_metadata = dm.load_metadata(test_id)
    print(f"   学籍番号: {loaded_metadata['student_id']}")
    print(f"   氏名: {loaded_metadata['name']}")
    print(f"   試行数: {loaded_metadata['trials_completed']}")
    assert loaded_metadata["student_id"] == test_id
    assert loaded_metadata["name"] == "テスト太郎"
    print("   ✓ メタデータ読み込み成功\n")
    
    # 5. 試行追加テスト
    print("5. add_trial_to_metadata テスト")
    trial_info = {
        "trial_number": 2,
        "result": "failure",
        "timestamp": "2025-11-18T14:32:00",
        "csv_file": "trial_02.csv"
    }
    dm.add_trial_to_metadata(test_id, trial_info)
    
    updated_metadata = dm.load_metadata(test_id)
    assert updated_metadata["trials_completed"] == 2
    print(f"   試行数: {updated_metadata['trials_completed']}")
    print("   ✓ 試行追加成功\n")
    
    # 6. 統計更新テスト
    print("6. update_statistics テスト")
    dm.update_statistics(test_id)
    
    final_metadata = dm.load_metadata(test_id)
    stats = final_metadata["statistics"]
    print(f"   成功: {stats['success_count']}")
    print(f"   失敗: {stats['failure_count']}")
    print(f"   成功率: {stats['success_rate']}")
    assert stats["success_count"] == 1
    assert stats["failure_count"] == 1
    assert stats["success_rate"] == 0.5
    print("   ✓ 統計更新成功\n")
    
    print("✅ All tests passed!")
    print(f"\nテストデータは {folder} に保存されています")
