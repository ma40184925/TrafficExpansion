from pathlib import Path
import sys
import os


class PathManager:
    def __init__(self):
        # 获取当前文件 (path_manager.py) 的绝对路径
        current_file = Path(__file__).resolve()

        # 根据目录结构回溯到项目根目录
        # current: src/utils/path_manager.py
        # parent 1: src/utils
        # parent 2: src
        # parent 3: 项目根目录
        self.project_root = current_file.parent.parent.parent

        # 定义数据目录
        self.data_root = self.project_root / "data"
        self.data_raw = self.data_root / "raw"
        self.data_processed = self.data_root / "processed"

        # 自动创建目录（如果不存在）
        self._ensure_dirs()

    def _ensure_dirs(self):
        """确保关键目录存在"""
        for p in [self.data_root, self.data_raw, self.data_processed]:
            p.mkdir(parents=True, exist_ok=True)

    def get_raw_path(self, filename: str) -> Path:
        """获取原始数据文件的完整路径"""
        return self.data_raw / filename

    def get_processed_path(self, filename: str) -> Path:
        """获取处理后数据文件的完整路径"""
        return self.data_processed / filename


# 实例化一个单例对象，方便外部直接导入使用
pm = PathManager()

# --- 调试代码 (直接运行此脚本时会打印路径) ---
if __name__ == "__main__":
    print(f"项目根目录: {pm.project_root}")
    print(f"数据目录: {pm.data_root}")
    print(f"测试路径: {pm.get_raw_path('test.xlsx')}")