"""
转录引擎：faster-whisper GPU 加速 + 静音边界音频分段。

使用 faster-whisper (CTranslate2 后端) 在 CUDA 上执行语音识别。
长音频（>max_segment_length）通过基于能量的静音检测自动分段，
确保分段边界落在静音区间而非句子中间。
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

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
                    self.model_size,
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
    # 内部：静音边界检测（能量法）
    # ----------------------------------------------------------

    def _find_silence_boundaries(
        self, audio: "np.ndarray", sr: int
    ) -> List[float]:
        """基于短时能量检测静音边界，返回候选分段点列表。

        算法流程：
        1. 以 50ms 窗口计算 RMS 能量
        2. 转换为 dBFS 分贝值
        3. 标记低于 silence_threshold 的窗口为"静音"
        4. 合并连续的静音窗口为静音区间
        5. 筛选持续时长 >= min_silence_duration 的静音区间
        6. 取每个静音区间的中点作为候选分段点
        7. 在候选分段点中选择最接近理想分段位置的作为实际分段边界

        Args:
            audio: float32 音频数组
            sr: 采样率（16000 Hz）

        Returns:
            分段边界时间点列表（秒），升序排列。
            若音频长度未超过 max_segment_length，返回空列表。
        """
        total_duration = len(audio) / sr

        # 无需分段
        if total_duration <= self.max_segment_length:
            return []

        # ---- 第 1 步：短时能量计算 ----
        window_samples = int(sr * 0.05)  # 50ms 窗口
        hop_samples = window_samples     # 无重叠（省计算）

        num_windows = max(1, (len(audio) - window_samples) // hop_samples + 1)

        rms = np.zeros(num_windows, dtype=np.float64)
        for i in range(num_windows):
            start = i * hop_samples
            end = start + window_samples
            chunk = audio[start:end].astype(np.float64)
            rms[i] = np.sqrt(np.mean(chunk ** 2))

        # ---- 第 2 步：转换为 dBFS ----
        eps = 1e-10
        dbfs = 20.0 * np.log10(np.maximum(rms, eps))

        # ---- 第 3-4 步：标记静音窗口并合并 ----
        is_silent = dbfs < self.silence_threshold

        # 最小连续静音窗口数 = min_silence_duration_ms / 1000 * sr / hop_samples
        min_silent_windows = max(
            1, int(self.min_silence_duration / 1000.0 * sr / hop_samples)
        )

        # ---- 第 5 步：提取合格的静音区间中点 ----
        silence_midpoints: List[float] = []
        in_silence = False
        silence_start_idx = 0

        for i, silent in enumerate(is_silent):
            if silent and not in_silence:
                in_silence = True
                silence_start_idx = i
            elif not silent and in_silence:
                dur = i - silence_start_idx
                if dur >= min_silent_windows:
                    mid_sec = (silence_start_idx + i) / 2.0 * hop_samples / sr
                    silence_midpoints.append(mid_sec)
                in_silence = False

        # 处理末尾的静音段
        if in_silence:
            dur = len(is_silent) - silence_start_idx
            if dur >= min_silent_windows:
                mid_sec = (
                    (silence_start_idx + len(is_silent)) / 2.0 * hop_samples / sr
                )
                silence_midpoints.append(mid_sec)

        # ---- 第 6-7 步：选择最接近理想位置的分段点 ----
        if not silence_midpoints:
            # 回退方案：无可用静音区间 → 均匀切分
            boundaries: List[float] = []
            pos = float(self.max_segment_length)
            while pos < total_duration - 1.0:
                boundaries.append(pos)
                pos += self.max_segment_length
            return boundaries

        boundaries = []
        last_boundary = 0.0
        pos = float(self.max_segment_length)

        while pos < total_duration - 1.0:
            # 在所有候选静音中点中，找最接近理想位置 pos 的那个
            best = None
            best_dist = float("inf")
            for sp in silence_midpoints:
                # 至少离上一个边界 1 秒以上，且离音频结尾至少 1 秒
                if sp > last_boundary + 1.0 and sp < total_duration - 1.0:
                    dist = abs(sp - pos)
                    if dist < best_dist:
                        best_dist = dist
                        best = sp

            if best is not None and best > last_boundary:
                boundaries.append(best)
                last_boundary = best
                pos = best + self.max_segment_length
            else:
                # 找不到合适的静音点 → 硬切
                boundaries.append(pos)
                last_boundary = pos
                pos += self.max_segment_length

        return boundaries

    # ----------------------------------------------------------
    # 内部：构建分段区间
    # ----------------------------------------------------------

    def _build_segment_ranges(
        self, boundaries: List[float], total_duration: float
    ) -> List[Dict[str, float]]:
        """将分段边界转换为带重叠的分段起止时间区间。

        Args:
            boundaries: 分段边界点列表（秒）
            total_duration: 音频总时长（秒）

        Returns:
            分段信息列表，每项包含 {'start': float, 'end': float}
        """
        overlap_s = self.overlap_duration / 1000.0
        half_overlap = overlap_s / 2.0

        if not boundaries:
            return [{"start": 0.0, "end": total_duration}]

        segments: List[Dict[str, float]] = []

        # 第一段：0 → 第一个边界 + 半重叠
        segments.append({
            "start": 0.0,
            "end": min(boundaries[0] + half_overlap, total_duration),
        })

        # 中间段：前一边界 - 半重叠 → 当前边界 + 半重叠
        for i in range(1, len(boundaries)):
            start = max(0.0, boundaries[i - 1] - half_overlap)
            end = min(boundaries[i] + half_overlap, total_duration)
            segments.append({"start": start, "end": end})

        # 最后一段：最后一个边界 - 半重叠 → 音频结束
        segments.append({
            "start": max(0.0, boundaries[-1] - half_overlap),
            "end": total_duration,
        })

        return segments

    # ----------------------------------------------------------
    # 公共 API：transcribe()
    # ----------------------------------------------------------

    def transcribe(
        self,
        audio_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """对音频文件执行语音转文字，返回合并后的完整结果。

        长音频（超过 max_segment_length）会自动在静音边界处分段，
        每段独立转录后合并。所有 segment 时间戳连续无间隙。

        Args:
            audio_path: 16kHz 单声道 WAV 音频文件路径
            progress_callback: 可选进度回调 callback(current, total)，
                              current 为当前段索引（从 1 开始），
                              total 为总段数

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

        # ---- 2. 确定分段边界 ----
        boundaries = self._find_silence_boundaries(audio, sr)
        segment_ranges = self._build_segment_ranges(boundaries, total_duration)

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

                # 转录（启用 VAD 过滤非语音片段）
                segments_gen, info = model.transcribe(
                    chunk,
                    language=self.language,
                    vad_filter=True,
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
