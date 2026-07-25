"""
音频提取与转换模块。

从视频文件中提取音频（ffmpeg），将音频文件转换为 16kHz 单声道 WAV，
输出 whisper 兼容的音频格式。
"""

import os
import shutil
import subprocess
import wave
from pathlib import Path


# ============================================================
# 支持的格式常量
# ============================================================

SUPPORTED_VIDEO_FORMATS = frozenset({
    "mp4", "mkv", "avi", "mov", "webm", "flv", "wmv", "m4v", "ts", "mts",
})

SUPPORTED_AUDIO_FORMATS = frozenset({
    "mp3", "wav", "flac", "aac", "ogg", "m4a", "wma", "opus",
})

# 合并所有支持的格式
_ALL_SUPPORTED_FORMATS = SUPPORTED_VIDEO_FORMATS | SUPPORTED_AUDIO_FORMATS


# ============================================================
# 内部辅助函数
# ============================================================

def _get_ext(file_path: str) -> str:
    """获取文件扩展名（小写，不含点号）。

    Args:
        file_path: 文件路径

    Returns:
        小写的扩展名，如 "mp4"；无扩展名时返回空字符串
    """
    return Path(file_path).suffix.lstrip(".").lower()


def _is_video(ext: str) -> bool:
    """判断扩展名是否属于视频格式。"""
    return ext in SUPPORTED_VIDEO_FORMATS


def _is_audio(ext: str) -> bool:
    """判断扩展名是否属于音频格式。"""
    return ext in SUPPORTED_AUDIO_FORMATS


def _is_already_target_wav(file_path: str, target_sample_rate: int = 16000) -> bool:
    """检查 WAV 文件是否已经是目标格式（指定采样率、单声道、PCM 16-bit）。

    Args:
        file_path: WAV 文件路径
        target_sample_rate: 目标采样率，默认 16000

    Returns:
        如果文件已是目标格式则返回 True，否则返回 False
    """
    try:
        with wave.open(file_path, "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
        return (channels == 1 and sample_rate == target_sample_rate
                and sample_width == 2)  # 16-bit = 2 bytes
    except (wave.Error, FileNotFoundError, EOFError):
        return False


def _run_ffmpeg(input_path: str, output_path: str) -> None:
    """调用 ffmpeg 将输入文件转换为 16kHz 单声道 WAV。

    对视频文件使用 `-vn` 跳过视频流，对纯音频文件同样适用。

    Args:
        input_path: 输入文件路径
        output_path: 输出 WAV 文件路径

    Raises:
        FileNotFoundError: ffmpeg 未安装或不在 PATH 中
        RuntimeError: ffmpeg 执行失败
    """
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-vn",                      # 忽略视频流（对音频文件无害）
        "-acodec", "pcm_s16le",     # 输出格式：PCM 16-bit little-endian
        "-ar", "16000",             # 采样率：16000 Hz
        "-ac", "1",                 # 声道数：单声道
        "-y",                       # 覆盖已存在的输出文件
        output_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            "未找到 ffmpeg 命令。\n"
            "请安装 ffmpeg 并将其添加到系统 PATH 环境变量中。\n"
            "下载地址: https://ffmpeg.org/download.html"
        )

    if result.returncode != 0:
        # 分析 stderr 以提供更可操作的提示
        stderr_text = result.stderr.strip()
        hints: list[str] = []

        if "No such file or directory" in stderr_text:
            hints.append("→ 输入文件可能已损坏或路径包含特殊字符")
        if "Invalid data found" in stderr_text or "invalid" in stderr_text.lower():
            hints.append("→ 文件格式可能不受支持或文件已损坏")
        if "Permission denied" in stderr_text:
            hints.append("→ 无权限访问该文件，请检查文件权限")
        if "protocol not found" in stderr_text.lower():
            hints.append("→ ffmpeg 缺少必要的协议支持，请检查 ffmpeg 安装")

        hint_text = "\n".join(hints) if hints else ""
        separator = "\n\n提示:\n" if hints else ""

        raise RuntimeError(
            f"ffmpeg 转换失败（返回码 {result.returncode}）\n"
            f"输入文件: {input_path}\n"
            f"输出文件: {output_path}\n"
            f"ffmpeg 错误信息:\n{stderr_text}"
            f"{separator}{hint_text}\n\n"
            f"常见问题检查:\n"
            f"  1. 确认 ffmpeg 已安装并在 PATH 中: 运行 'ffmpeg -version' 验证\n"
            f"  2. 确认输入文件未损坏: 尝试用媒体播放器打开\n"
            f"  3. 确认输入文件不是 DRM（数字版权保护）加密的内容"
        )


# ============================================================
# 公共 API：音频时长
# ============================================================

def get_wav_duration(wav_path: str) -> float:
    """读取 WAV 文件的时长（秒）。

    Args:
        wav_path: WAV 文件路径

    Returns:
        音频时长（秒），浮点数

    Raises:
        FileNotFoundError: WAV 文件不存在
        wave.Error: 文件不是有效的 WAV 格式
    """
    with wave.open(wav_path, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        if rate == 0:
            return 0.0
        return frames / float(rate)


# ============================================================
# 公共 API：音频提取
# ============================================================

def extract_audio(file_path: str, temp_dir: str = "./temp") -> str:
    """从视频或音频文件中提取/转换为 16kHz 单声道 WAV 音频。

    处理逻辑：
    1. 检查格式是否支持（不支持则抛出 ValueError）
    2. 检查输入文件是否存在
    3. 确保 temp_dir 目录存在
    4. 对于已是 16kHz 单声道 PCM WAV 的音频文件，直接复制到 temp_dir
    5. 其余情况调用 ffmpeg 进行提取/转换，输出统一为 16kHz 单声道 WAV

    Args:
        file_path: 输入文件路径（视频或音频）
        temp_dir: 临时文件输出目录，默认为 "./temp"

    Returns:
        转换后的 16kHz 单声道 WAV 文件路径

    Raises:
        FileNotFoundError: 输入文件不存在或 ffmpeg 未安装
        ValueError: 文件格式不支持
        RuntimeError: ffmpeg 执行失败
    """
    # --- 1. 检查格式是否支持 ---
    ext = _get_ext(file_path)
    if ext not in _ALL_SUPPORTED_FORMATS:
        raise ValueError(
            f"不支持的文件格式: .{ext}（文件: {file_path}）\n"
            f"支持的视频格式: {', '.join(sorted(SUPPORTED_VIDEO_FORMATS))}\n"
            f"支持的音频格式: {', '.join(sorted(SUPPORTED_AUDIO_FORMATS))}"
        )

    # --- 2. 检查输入文件是否存在 ---
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"输入文件不存在: {file_path}")

    # --- 3. 确保临时目录存在 ---
    os.makedirs(temp_dir, exist_ok=True)

    # --- 4. 确定输出路径 ---
    stem = Path(file_path).stem
    output_path = os.path.join(temp_dir, f"{stem}.wav")

    # --- 5. 处理音频文件 ---
    if _is_audio(ext):
        # 对 WAV 文件检查是否已经是目标格式
        if ext == "wav" and _is_already_target_wav(file_path):
            # 已经是 16kHz 单声道 PCM WAV → 直接复制
            shutil.copy2(file_path, output_path)
            return output_path
        else:
            # 需要转换（mp3, flac, aac, ogg, m4a, wma, opus 或非标准 WAV）
            _run_ffmpeg(file_path, output_path)
            return output_path

    # --- 6. 处理视频文件 ---
    if _is_video(ext):
        _run_ffmpeg(file_path, output_path)
        return output_path

    # 理论上不会到达这里（已在第 2 步过滤不支持格式）
    raise RuntimeError(f"意外的格式分类错误: .{ext}")
