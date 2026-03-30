"""
Experiment Configuration Module

実験管理システムの設定値、定数、パス管理を行うモジュール
"""

from pathlib import Path

# 試行設定
DEFAULT_TRIAL_COUNT = 20
DATA_ROOT_DIR = "data"

# CSV列名
CSV_COLUMNS = [
    "frame_index",
    "timestamp_sec",
    "X_m",
    "Y_m",
    "Z_m",
    "VX_ms",
    "VY_ms",
    "VZ_ms",
    "AX_ms2",
    "AY_ms2",
    "AZ_ms2",
    "is_predicted"
]

# メタデータファイル名
METADATA_FILENAME = "metadata.json"
TRIAL_FILENAME_TEMPLATE = "trial_{:02d}.csv"


def get_subject_dir(student_id: str) -> Path:
    """
    被験者のデータディレクトリパスを取得
    
    Args:
        student_id: 学籍番号
        
    Returns:
        Path: data/{student_id}/
        
    Example:
        >>> get_subject_dir("B2230123")
        Path("data/B2230123")
    """
    return Path(DATA_ROOT_DIR) / student_id


def get_trial_csv_path(student_id: str, trial_number: int) -> Path:
    """
    試行のCSVファイルパスを取得
    
    Args:
        student_id: 学籍番号
        trial_number: 試行番号（1〜10）
        
    Returns:
        Path: CSVファイルパス
        
    Raises:
        ValueError: trial_numberが範囲外の場合
        
    Example:
        >>> get_trial_csv_path("B2230123", 1)
        Path("data/B2230123/trial_01.csv")
    """
    if not 1 <= trial_number <= DEFAULT_TRIAL_COUNT:
        raise ValueError(
            f"trial_number must be between 1 and {DEFAULT_TRIAL_COUNT}, "
            f"got {trial_number}"
        )
    
    filename = TRIAL_FILENAME_TEMPLATE.format(trial_number)
    return get_subject_dir(student_id) / filename


def get_metadata_path(student_id: str) -> Path:
    """
    メタデータファイルパスを取得
    
    Args:
        student_id: 学籍番号
        
    Returns:
        Path: metadata.jsonのパス
        
    Example:
        >>> get_metadata_path("B2230123")
        Path("data/B2230123/metadata.json")
    """
    return get_subject_dir(student_id) / METADATA_FILENAME


if __name__ == "__main__":
    # テスト実行
    print("=== experiment_config.py テスト ===")
    
    test_id = "B2230123"
    
    # get_subject_dir テスト
    subject_dir = get_subject_dir(test_id)
    print(f"Subject dir: {subject_dir}")
    assert subject_dir == Path("data/B2230123")
    
    # get_trial_csv_path テスト
    trial_path = get_trial_csv_path(test_id, 1)
    print(f"Trial 1 CSV: {trial_path}")
    assert trial_path == Path("data/B2230123/trial_01.csv")
    
    trial_path_10 = get_trial_csv_path(test_id, 10)
    print(f"Trial 10 CSV: {trial_path_10}")
    assert trial_path_10 == Path("data/B2230123/trial_10.csv")
    
    # get_metadata_path テスト
    metadata_path = get_metadata_path(test_id)
    print(f"Metadata: {metadata_path}")
    assert metadata_path == Path("data/B2230123/metadata.json")
    
    # エラーケーステスト
    try:
        get_trial_csv_path(test_id, 0)
        assert False, "Should raise ValueError"
    except ValueError as e:
        print(f"✓ ValueError for trial_number=0: {e}")
    
    try:
        get_trial_csv_path(test_id, 11)
        assert False, "Should raise ValueError"
    except ValueError as e:
        print(f"✓ ValueError for trial_number=11: {e}")
    
    print("\n✅ All tests passed!")
