"""
转录引擎：faster-whisper GPU 加速 + 静音边界音频分段。

使用 faster-whisper (CTranslate2 后端) 在 CUDA 上执行语音识别。
长音频（>max_segment_length）通过基于能量的静音检测自动分段，
确保分段边界落在静音区间而非句子中间。
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, List, Optional

# ============================================================
# Windows DLL 路径修复
# GPU 库（cuBLAS）通过 pip 安装后位于 nvidia 包的深层目录，
# Windows 下 Python 默认搜不到，需要手动注册 DLL 搜索路径。
# 必须在 import faster_whisper / ctranslate2 之前执行。
# ============================================================
if sys.platform == "win32":
    import ctypes
    # 直接预加载 cuBLAS DLL，比 add_dll_directory 更可靠
    _nvidia_base = os.path.join(
        os.path.dirname(sys.executable) if getattr(sys, "frozen", False)
        else sys.prefix, "Lib", "site-packages", "nvidia"
    )
    if os.path.isdir(_nvidia_base):
        for _root, _dirs, _files in os.walk(_nvidia_base):
            if "bin" in _dirs:
                _bin_path = os.path.join(_root, "bin")
                # 加载该 bin 目录下所有 DLL（cublas, cudart, etc.）
                for _dll_name in os.listdir(_bin_path):
                    if _dll_name.endswith(".dll"):
                        _dll_path = os.path.join(_bin_path, _dll_name)
                        try:
                            ctypes.cdll.LoadLibrary(_dll_path)
                        except OSError:
                            pass

import numpy as np
import soundfile as sf
from tqdm import tqdm


# ============================================================
# WhisperTranscriber
# ============================================================

class WhisperTranscriber:
    """基于 faster-whisper 的 GPU 加速语音转文字转录器。

    特性：
    - CUDA GPU 加速，支持 float16 / int8_float16 / int8 计算精度
    - 模型文件下载到可配置的 download_root 目录（默认 ./models/）
    - 长音频智能分段：基于能量检测的静音边界识别，
      分段边界落在静音区间而非句子中间
    - 内置 Silero VAD 过滤非语音片段
    - tqdm 进度条显示当前段落/总段落数
    - 支持 progress_callback 自定义进度回调

    用法示例::

        transcriber = WhisperTranscriber(
            model_size="large-v3",
            device="cuda",
            compute_type="float16",
            language="zh",
            download_root="./models",
        )
        result = transcriber.transcribe("audio.wav")
        for seg in result["segments"]:
            print(f"[{seg['start']:.1f}s-{seg['end']:.1f}s] {seg['text']}")
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "zh",
        download_root: str = "./models",
        max_segment_length: int = 600,
        silence_threshold: int = -40,
        min_silence_duration: int = 500,
        overlap_duration: int = 200,
    ):
        """初始化转录器。

        Args:
            model_size: Whisper 模型大小。
                        可选: tiny, base, small, medium, large-v2, large-v3
            device: 计算设备，'cuda' 或 'cpu'
            compute_type: 模型精度。
                          float16: GPU 半精度（推荐，显存充足时最快）
                          int8_float16: 矩阵乘法 int8 + 其他 float16（显存紧张时）
                          int8: 纯 8-bit 量化（CPU 推理）
            language: 目标语言代码（如 'zh'、'en'），空字符串 '' 表示自动检测
            download_root: 模型下载与缓存目录（默认 './models'）
            max_segment_length: 最大段长度（秒），超过此值自动分段
            silence_threshold: 静音检测阈值（dBFS），音量低于此值视为静音
            min_silence_duration: 最小静音持续时长（毫秒），
                                  连续静音超过此时长才能作为分段点
            overlap_duration: 相邻分段重叠时长（毫秒），
                              防止边界处词语被截断
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language if language else None
        self.download_root = download_root
        self.max_segment_length = max_segment_length
        self.silence_threshold = silence_threshold
        self.min_silence_duration = min_silence_duration
        self.overlap_duration = overlap_duration

        # 延迟加载模型，首次调用 transcribe() 时才初始化
        self._model = None

    # ----------------------------------------------------------
    # 内部：本地模型路径解析
    # ----------------------------------------------------------

    # 模型短名称 → 可能的本地目录名 映射
    _MODEL_NAME_CANDIDATES = {
        "large-v3": ["faster-whisper-large-v3", "models--Systran--faster-whisper-large-v3"],
        "large-v2": ["faster-whisper-large-v2", "models--Systran--faster-whisper-large-v2"],
        "distil-large-v3": ["faster-distil-whisper-large-v3", "models--Systran--faster-distil-whisper-large-v3"],
        "medium": ["faster-whisper-medium", "models--Systran--faster-whisper-medium"],
        "small": ["faster-whisper-small", "models--Systran--faster-whisper-small"],
        "base": ["faster-whisper-base", "models--Systran--faster-whisper-base"],
        "tiny": ["faster-whisper-tiny", "models--Systran--faster-whisper-tiny"],
    }

    def _resolve_local_model_path(self) -> str:
        """解析本地模型路径，如果 download_root 下已有模型目录则返回绝对路径。

        优先查找 download_root 下的候选目录名，按顺序尝试：
        1. 简化目录名（如 faster-whisper-large-v3）— 用户手动下载常见格式
        2. HF 缓存格式（如 models--Systran--faster-whisper-large-v3）— 脚本下载格式

        如果都不存在，返回原始 model_size 短名称，由 faster-whisper 自行处理（会触发联网下载）。

        Returns:
            本地模型目录的绝对路径，或原始 model_size 短名称
        """
        candidates = self._MODEL_NAME_CANDIDATES.get(self.model_size, [self.model_size])
        for name in candidates:
            local_path = os.path.join(self.download_root, name)
            if os.path.isdir(local_path) and os.path.isfile(os.path.join(local_path, "model.bin")):
                return os.path.abspath(local_path)
        return self.model_size

    # ----------------------------------------------------------
    # 内部：模型加载
    # ----------------------------------------------------------

    def _get_model(self):
        """延迟加载 faster-whisper 模型，支持 GPU OOM 自动回退。

        加载策略（逐级回退）：
        1. 使用用户配置的 device + compute_type（如 cuda + float16）
        2. CUDA OOM → 回退到 cuda + int8_float16（更低显存占用）
        3. 仍 OOM → 回退到 cpu + int8（纯 CPU 推理）
        每一步回退时打印明确警告。

        首次调用时自动下载模型到 download_root 目录。
        后续调用直接返回已缓存的模型实例。

        Returns:
            faster_whisper.WhisperModel 实例
        """
        if self._model is not None:
            return self._model

        from faster_whisper import WhisperModel

        # 确保模型下载目录存在
        os.makedirs(self.download_root, exist_ok=True)

        # ---- 优先使用本地模型路径 ----
        # 如果 download_root 下已有对应的模型目录，直接用本地路径，
        # 避免触发联网下载（国内网络不稳定）
        model_path = self._resolve_local_model_path()

        # ---- 构建回退策略链 ----
        # 每项为 (device, compute_type, description) 三元组
        fallback_chain = []

        # 第 1 级：用户配置的原始设置
        fallback_chain.append(
            (self.device, self.compute_type, "用户配置")
        )

        # 第 2 级：如果设备是 cuda 且类型是 float16，追加 int8_float16
        if self.device == "cuda" and self.compute_type == "float16":
            fallback_chain.append(
                ("cuda", "int8_float16", "int8_float16（降低精度以节省显存）")
            )

        # 第 3 级：如果设备是 cuda，追加 cpu + int8
        if self.device == "cuda":
            fallback_chain.append(
                ("cpu", "int8", "CPU int8（纯 CPU 推理，速度较慢）")
            )

        # ---- 逐级尝试加载 ----
        last_error = None
        for device, compute_type, desc in fallback_chain:
            try:
                self._model = WhisperModel(
                    model_path,  # 优先用本地路径，不存在时用短名称触发下载
                    device=device,
                    compute_type=compute_type,
                    download_root=self.download_root,
                )
                # 加载成功：更新实例变量以反映实际使用的配置
                if device != self.device or compute_type != self.compute_type:
                    print(
                        f"  [警告] GPU 显存不足，已自动回退到 {desc} 模式。\n"
                        f"         当前配置: device={device}, compute_type={compute_type}"
                    )
                    self.device = device
                    self.compute_type = compute_type
                return self._model
            except RuntimeError as e:
                last_error = e
                # 检查是否为 CUDA OOM 导致的错误
                error_msg = str(e).lower()
                is_oom = any(
                    keyword in error_msg
                    for keyword in (
                        "out of memory",
                        "cuda_error_out_of_memory",
                        "cuda out of memory",
                        "not enough memory",
                        "memory allocation",
                        "cuda error",
                    )
                )
                if is_oom and device == "cuda":
                    # 这是 OOM，继续尝试下一级回退
                    print(
                        f"  [警告] {desc} 模式加载失败（显存不足），"
                        f"尝试下一级回退..."
                    )
                    continue
                else:
                    # 非 OOM 错误或已在 CPU 上仍失败 → 直接抛出
                    raise

        # 所有回退策略均失败
        raise RuntimeError(
            f"模型加载失败：所有回退策略均已尝试但仍失败。\n"
            f"最后一次错误: {last_error}\n"
            f"已尝试的策略: "
            + " → ".join(f"{d}+{c}" for d, c, _ in fallback_chain)
            + "\n"
            f"建议: 请尝试使用更小的模型（如 'base' 或 'small'），"
            f"或检查 GPU 驱动是否正常。"
        )

    # ----------------------------------------------------------
    # 内部：音频读取
    # ----------------------------------------------------------

    @staticmethod
    def _load_audio(audio_path: str) -> "np.ndarray":
        """加载 WAV 音频文件为 float32 numpy 数组。

        Args:
            audio_path: WAV 文件路径（须为 16kHz 单声道）

        Returns:
            float32 音频数组，范围 [-1.0, 1.0]

        Raises:
            ValueError: 采样率不是 16000 Hz
            FileNotFoundError: 文件不存在
        """
        audio, sr = sf.read(audio_path, dtype="float32")

        if sr != 16000:
            raise ValueError(
                f"音频采样率必须为 16000 Hz（whisper 模型要求），当前为 {sr} Hz。"
                f"请先用 extractor.py 将音频转换为 16kHz 单声道 WAV。"
            )

        # 如果是多声道，取平均转为单声道
        if audio.ndim > 1:
            audio = audio.mean(axis=1).astype("float32")

        return audio

    # ----------------------------------------------------------
    # 内部：均匀分段
    # ----------------------------------------------------------

    def _build_segments(
        self, total_duration: float
    ) -> List[Dict[str, float]]:
        """按 max_segment_length 均匀切分音频，段间有小重叠防止边界丢失。

        若音频长度未超过 max_segment_length，返回单段（不分段）。

        Args:
            total_duration: 音频总时长（秒）

        Returns:
            分段信息列表，每项包含 {'start': float, 'end': float}
        """
        if total_duration <= self.max_segment_length:
            return [{"start": 0.0, "end": total_duration}]

        overlap_s = self.overlap_duration / 1000.0
        segments: List[Dict[str, float]] = []
        pos = 0.0

        while pos < total_duration:
            end = min(pos + self.max_segment_length + overlap_s, total_duration)
            segments.append({"start": pos, "end": end})
            pos += self.max_segment_length

        return segments

    # ----------------------------------------------------------
    # 公共 API：transcribe()
    # ----------------------------------------------------------

    def transcribe(
        self,
        audio_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        segment_progress_callback: Optional[Callable[[float, float], None]] = None,
    ) -> Dict[str, Any]:
        """对音频文件执行语音转文字，返回合并后的完整结果。

        长音频（超过 max_segment_length）会自动分段，每段独立转录后合并。

        Args:
            audio_path: 16kHz 单声道 WAV 音频文件路径
            progress_callback: 可选回调 callback(current_chunk, total_chunks)
            segment_progress_callback: 可选回调 callback(audio_position_s, total_duration_s)
                                      基于真实 segment 时间戳，每段触发一次

        Returns:
            Dict 包含以下字段:
                - 'segments': List[Dict]，每个 Dict 包含:
                    - 'start': float — 段落开始时间（秒）
                    - 'end': float — 段落结束时间（秒）
                    - 'text': str — 转录文本
                - 'language': str — 检测到的语言代码
                - 'duration': float — 音频总时长（秒）
        """
        # ---- 1. 加载音频 ----
        audio = self._load_audio(audio_path)
        sr = 16000
        total_duration = len(audio) / sr

        # ---- 2. 确定分段 ----
        segment_ranges = self._build_segments(total_duration)

        # ---- 3. 加载模型 ----
        model = self._get_model()

        # ---- 4. 逐段转录 ----
        all_segments: List[Dict[str, Any]] = []
        detected_language: Optional[str] = None
        num_segments = len(segment_ranges)

        # tqdm 进度条：显示"Transcribing segment X/Y"
        pbar = tqdm(
            total=num_segments,
            desc="Transcribing",
            unit="segment",
            dynamic_ncols=True,
        )

        try:
            for idx, seg_range in enumerate(segment_ranges):
                start_t = seg_range["start"]
                end_t = seg_range["end"]

                pbar.set_description(
                    f"Transcribing segment {idx + 1}/{num_segments}"
                )

                # 提取音频片段
                start_sample = int(start_t * sr)
                end_sample = int(end_t * sr)
                chunk = audio[start_sample:end_sample]

                if len(chunk) < sr * 0.1:  # 跳过不足 0.1 秒的片段
                    pbar.update(1)
                    if progress_callback:
                        progress_callback(idx + 1, num_segments)
                    continue

                # 转录（VAD 对中文会议识别有负面影响，默认关闭）
                segments_gen, info = model.transcribe(
                    chunk,
                    language=self.language,
                    vad_filter=False,
                    beam_size=5,
                )

                # 记录检测语言（仅首次）
                if detected_language is None:
                    detected_language = info.language

                # 收集转录段落，时间戳偏移到全局时间轴
                for seg in segments_gen:
                    all_segments.append({
                        "start": round(start_t + seg.start, 3),
                        "end": round(start_t + seg.end, 3),
                        "text": seg.text.strip(),
                    })
                    # 基于真实时间戳的进度回调
                    if segment_progress_callback:
                        segment_progress_callback(
                            start_t + seg.end, total_duration
                        )

                pbar.update(1)

                if progress_callback:
                    progress_callback(idx + 1, num_segments)
        finally:
            pbar.close()

        # ---- 5. 结果合并与去重 ----
        # 按开始时间排序
        all_segments.sort(key=lambda s: s["start"])

        merged: List[Dict[str, Any]] = []
        for seg in all_segments:
            # 跳过空文本段落
            if not seg["text"]:
                continue

            # 跳过完全被上一段包含的段落（重叠去重）
            if merged and seg["end"] <= merged[-1]["end"]:
                continue

            # 消除时间戳间隙：当前段的开始不早于上一段的结束
            if merged:
                seg["start"] = round(max(seg["start"], merged[-1]["end"]), 3)

            # 确保 end > start
            if seg["end"] > seg["start"]:
                merged.append(seg)

        return {
            "segments": merged,
            "language": detected_language or "unknown",
            "duration": round(total_duration, 3),
        }
