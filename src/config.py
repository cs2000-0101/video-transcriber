"""
配置加载与验证模块。

从 config.yaml 加载配置，验证必需字段，对缺失字段回退到默认值，
对无效值抛出明确的错误信息。
"""

from dataclasses import dataclass
import os
from typing import Any, Dict, Optional

import yaml


# ============================================================
# 默认值 —— 所有可选字段的回退值
# ============================================================
DEFAULTS: Dict[str, Any] = {
    "model.size": "large-v3",
    "model.compute_type": "float16",
    "model.language": "zh",
    "model.download_root": "./models",
    "audio_splitting.max_segment_length": 600,
    "audio_splitting.silence_threshold": -40,
    "audio_splitting.min_silence_duration": 500,
    "audio_splitting.overlap_duration": 200,
    "audio_extraction.sample_rate": 16000,
    "output.format": "both",
    "output.with_timestamps": True,
    "output.output_dir": "./transcripts",
    "temp_dir": "./temp",
}

# ============================================================
# 有效值集合 —— 用于验证枚举类字段
# ============================================================
VALID_MODEL_SIZES = {"tiny", "base", "small", "medium", "large-v2", "large-v3"}
VALID_COMPUTE_TYPES = {"float16", "int8_float16", "int8"}
VALID_OUTPUT_FORMATS = {"md", "docx", "both"}


# ============================================================
# 配置数据类
# ============================================================
@dataclass
class AppConfig:
    """视频转录工具的运行时配置。
    
    所有字段均由 config.yaml 加载（缺失时使用默认值），
    经过验证的类型和有效值检查。
    """
    model_size: str
    model_compute_type: str
    model_language: str
    model_download_root: str
    audio_splitting_max_segment_length: int
    audio_splitting_silence_threshold: int
    audio_splitting_min_silence_duration: int
    audio_splitting_overlap_duration: int
    audio_extraction_sample_rate: int
    output_format: str
    output_with_timestamps: bool
    output_output_dir: str
    temp_dir: str


# ============================================================
# 辅助函数
# ============================================================
def _get_nested(data: dict, path: str, default: Any = None) -> Any:
    """从嵌套字典中按点号分隔路径取值。

    例如 _get_nested(d, "model.size") 等价于 d["model"]["size"]，
    但任一级别缺失时返回 default 而不是抛出 KeyError。

    Args:
        data: 嵌套字典
        path: 点号分隔的键路径，如 "model.size"
        default: 路径不存在时的默认值

    Returns:
        路径对应的值，或 default
    """
    keys = path.split(".")
    current: Any = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


# ============================================================
# 配置加载与验证
# ============================================================
def load_config(config_path: str) -> AppConfig:
    """加载并验证 YAML 配置文件。

    加载流程：
    1. 检查文件是否存在
    2. 解析 YAML
    3. 对每个字段取值（缺失时回退到 DEFAULTS）
    4. 对枚举字段验证有效值
    5. 对数值字段验证范围
    6. 返回不可变的 AppConfig 实例

    Args:
        config_path: config.yaml 文件路径

    Returns:
        AppConfig: 经过验证的配置对象

    Raises:
        FileNotFoundError: 配置文件不存在
        ValueError: 配置内容为空、格式错误或字段值无效
    """
    # --- 1. 文件存在性检查 ---
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"配置文件不存在: {config_path}\n"
            f"请确保 config.yaml 文件存在于项目根目录。"
        )

    # --- 2. YAML 解析 ---
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(
            f"配置文件 YAML 格式错误: {config_path}\n"
            f"YAML 解析器报错: {e}"
        ) from e

    if data is None:
        raise ValueError(
            f"配置文件为空或仅包含注释: {config_path}\n"
            f"请按照模板填写所有必需字段。"
        )

    if not isinstance(data, dict):
        raise ValueError(
            f"配置文件顶层结构必须是字典（键值对），"
            f"当前类型为: {type(data).__name__}"
        )

    # --- 3. 逐字段取值（缺失回退默认值） ---
    # 使用 DEFAULTS 作为回退，支持部分配置覆盖
    raw: Dict[str, Any] = {}
    missing_fields: list[str] = []
    for key_path, default_val in DEFAULTS.items():
        val = _get_nested(data, key_path)
        if val is None:
            raw[key_path] = default_val
            missing_fields.append(key_path)
        else:
            raw[key_path] = val

    # --- 4. 字段验证 ---

    # model.size
    if raw["model.size"] not in VALID_MODEL_SIZES:
        raise ValueError(
            f"无效的 model.size: '{raw['model.size']}'\n"
            f"可选值: {sorted(VALID_MODEL_SIZES)}\n"
            f"参考 config.yaml 中 model.size 的注释了解各选项含义。"
        )

    # model.compute_type
    if raw["model.compute_type"] not in VALID_COMPUTE_TYPES:
        raise ValueError(
            f"无效的 model.compute_type: '{raw['model.compute_type']}'\n"
            f"可选值: {sorted(VALID_COMPUTE_TYPES)}\n"
            f"参考 config.yaml 中 model.compute_type 的注释了解各选项含义。"
        )

    # audio_splitting.max_segment_length（必须为正数）
    max_seg = raw["audio_splitting.max_segment_length"]
    if not isinstance(max_seg, (int, float)) or max_seg <= 0:
        raise ValueError(
            f"audio_splitting.max_segment_length 必须为正数，"
            f"当前值: {max_seg}（类型: {type(max_seg).__name__}）"
        )

    # audio_splitting.silence_threshold（必须为数值）
    sil_thresh = raw["audio_splitting.silence_threshold"]
    if not isinstance(sil_thresh, (int, float)):
        raise ValueError(
            f"audio_splitting.silence_threshold 必须为数值（dBFS），"
            f"当前值: {sil_thresh}（类型: {type(sil_thresh).__name__}）"
        )

    # audio_splitting.min_silence_duration（必须为非负整数）
    min_sil = raw["audio_splitting.min_silence_duration"]
    if not isinstance(min_sil, (int, float)) or min_sil < 0:
        raise ValueError(
            f"audio_splitting.min_silence_duration 必须为非负数（毫秒），"
            f"当前值: {min_sil}（类型: {type(min_sil).__name__}）"
        )

    # audio_splitting.overlap_duration（必须为非负整数）
    overlap = raw["audio_splitting.overlap_duration"]
    if not isinstance(overlap, (int, float)) or overlap < 0:
        raise ValueError(
            f"audio_splitting.overlap_duration 必须为非负数（毫秒），"
            f"当前值: {overlap}（类型: {type(overlap).__name__}）"
        )

    # audio_extraction.sample_rate（必须为正整数）
    sr = raw["audio_extraction.sample_rate"]
    if not isinstance(sr, int) or sr <= 0:
        raise ValueError(
            f"audio_extraction.sample_rate 必须为正整数（Hz），"
            f"当前值: {sr}（类型: {type(sr).__name__}）"
        )

    # output.format
    if raw["output.format"] not in VALID_OUTPUT_FORMATS:
        raise ValueError(
            f"无效的 output.format: '{raw['output.format']}'\n"
            f"可选值: {sorted(VALID_OUTPUT_FORMATS)}\n"
            f"参考 config.yaml 中 output.format 的注释了解各选项含义。"
        )

    # --- 5. 构造并返回 AppConfig ---
    return AppConfig(
        model_size=raw["model.size"],
        model_compute_type=raw["model.compute_type"],
        model_language=raw["model.language"],
        model_download_root=raw["model.download_root"],
        audio_splitting_max_segment_length=int(raw["audio_splitting.max_segment_length"]),
        audio_splitting_silence_threshold=int(raw["audio_splitting.silence_threshold"]),
        audio_splitting_min_silence_duration=int(raw["audio_splitting.min_silence_duration"]),
        audio_splitting_overlap_duration=int(raw["audio_splitting.overlap_duration"]),
        audio_extraction_sample_rate=int(raw["audio_extraction.sample_rate"]),
        output_format=raw["output.format"],
        output_with_timestamps=bool(raw["output.with_timestamps"]),
        output_output_dir=raw["output.output_dir"],
        temp_dir=raw["temp_dir"],
    )
