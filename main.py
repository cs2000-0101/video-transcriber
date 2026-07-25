"""
CLI 入口：视频/音频批量转录工具命令行界面。

使用 argparse 解析命令行参数，串联 提取→转录→格式化 流水线，
支持单文件、多文件、文件夹批量处理，以及自定义配置文件路径。
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

# 在 Windows 上强制使用 UTF-8 编码输出，避免中文/特殊字符乱码
if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass  # 某些环境下 reconfigure 不可用，忽略

from src.config import load_config
from src.extractor import (
    extract_audio,
    get_wav_duration,
    SUPPORTED_VIDEO_FORMATS,
    SUPPORTED_AUDIO_FORMATS,
)
from src.transcriber import WhisperTranscriber
from src.formatter import save_transcript

# 最小有效音频长度（秒），短于此长度的音频将被跳过
MIN_AUDIO_DURATION = 0.5

# 合并所有支持的格式
_ALL_SUPPORTED_FORMATS = SUPPORTED_VIDEO_FORMATS | SUPPORTED_AUDIO_FORMATS


# ============================================================
# 辅助：文件收集
# ============================================================

def _collect_files(paths: List[str]) -> List[str]:
    """从路径列表中收集所有支持的媒体文件。

    对每个路径：
    - 如果是目录：递归遍历，收集所有支持格式的文件
    - 如果是文件：检查格式是否支持，支持则加入列表
    - 不存在或格式不支持：打印警告并跳过

    Args:
        paths: 文件或目录路径列表

    Returns:
        按文件名排序的绝对路径列表
    """
    files: List[str] = []

    for path_str in paths:
        path = Path(path_str)

        if path.is_dir():
            # 递归遍历目录
            for root, _dirs, filenames in os.walk(path_str):
                for fname in sorted(filenames):
                    ext = Path(fname).suffix.lstrip(".").lower()
                    if ext in _ALL_SUPPORTED_FORMATS:
                        files.append(os.path.join(root, fname))

        elif path.is_file():
            ext = path.suffix.lstrip(".").lower()
            if ext in _ALL_SUPPORTED_FORMATS:
                files.append(str(path.resolve()))
            else:
                print(f"  [警告] 跳过不支持的文件格式: {path_str}")

        else:
            print(f"  [警告] 路径不存在，已跳过: {path_str}")

    return files


# ============================================================
# 辅助：时间格式化
# ============================================================

def _format_elapsed(seconds: float) -> str:
    """将秒数格式化为人类可读的耗时字符串。

    Args:
        seconds: 经过的秒数（浮点）

    Returns:
        格式如 "1.5s" / "2m 30.3s" / "1h 5m 12.0s"
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


# ============================================================
# 主函数
# ============================================================

def main() -> None:
    """命令行入口主函数。

    流程：
    1. 解析命令行参数（paths + --config）
    2. 加载配置文件（验证 + 默认值回退）
    3. 从 paths 收集所有支持格式的文件
    4. 初始化 WhisperTranscriber（全局单例，模型仅加载一次）
    5. 逐文件执行：提取音频 → 转录 → 格式化输出
    6. 汇总统计
    """
    parser = argparse.ArgumentParser(
        description="Video-Transcriber -- 视频/音频批量转录工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py video.mp4                       # 单个视频文件
  python main.py file1.mp4 file2.mp3             # 多个文件批量处理
  python main.py ./videos/                       # 扫描文件夹
  python main.py video.mp4 --config custom.yaml  # 指定配置文件
        """,
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="输入文件或文件夹路径（支持视频和音频格式）",
    )
    parser.add_argument(
        "--config", "-c",
        default="./config.yaml",
        help="自定义配置文件路径（默认: ./config.yaml）",
    )

    args = parser.parse_args()

    # ---- 0. 打印启动横幅 ----
    print("=" * 60)
    print("Video-Transcriber -- 视频/音频转录工具")
    print("=" * 60)

    # ---- 1. 加载配置 ----
    config_path = args.config
    try:
        config = load_config(config_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"[错误] 配置加载失败: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[配置] 配置文件: {config_path}")
    print(f"[配置] 模型: {config.model_size} ({config.model_compute_type})")
    language_display = config.model_language if config.model_language else "自动检测"
    print(f"[配置] 语言: {language_display}")
    print(f"[配置] 输出格式: {config.output_format}")
    print(f"[配置] 输出目录: {config.output_output_dir}")
    if config.output_with_timestamps:
        print(f"[配置] 时间戳: 启用")
    else:
        print(f"[配置] 时间戳: 关闭")

    # ---- 2. 收集输入文件 ----
    print(f"\n[扫描] 正在扫描输入路径...")
    files = _collect_files(args.paths)

    if not files:
        print("\n[错误] 未找到任何支持的视频/音频文件。")
        print(f"   支持的视频格式: {', '.join(sorted(SUPPORTED_VIDEO_FORMATS))}")
        print(f"   支持的音频格式: {', '.join(sorted(SUPPORTED_AUDIO_FORMATS))}")
        sys.exit(1)

    print(f"\n[扫描] 共发现 {len(files)} 个文件待处理:")
    for f in files:
        print(f"   - {f}")

    # ---- 3. 初始化转录器（模型只加载一次，复用给所有文件） ----
    print(f"\n[初始化] 正在加载转录模型（首次运行需下载，请耐心等待）...")
    transcriber = WhisperTranscriber(
        model_size=config.model_size,
        device="cuda",
        compute_type=config.model_compute_type,
        language=config.model_language,
        download_root=config.model_download_root,
        max_segment_length=config.audio_splitting_max_segment_length,
        silence_threshold=config.audio_splitting_silence_threshold,
        min_silence_duration=config.audio_splitting_min_silence_duration,
        overlap_duration=config.audio_splitting_overlap_duration,
    )

    # ---- 4. 逐文件流水线处理 ----
    success_count = 0
    fail_count = 0
    total_start = time.time()

    for idx, file_path in enumerate(files, 1):
        file_name = os.path.basename(file_path)
        print(f"\n{'─' * 60}")
        print(f"[{idx}/{len(files)}] {file_name}")
        print(f"{'─' * 60}")

        file_start = time.time()

        try:
            # --- 步骤 1: 提取音频 ---
            t0 = time.time()
            print(f"  [提取] 正在提取/转换音频...")
            wav_path = extract_audio(file_path, temp_dir=config.temp_dir)
            t1 = time.time()
            print(f"  [提取] 完成 ({_format_elapsed(t1 - t0)})")
            print(f"         -> {wav_path}")

            # --- 步骤 1.5: 检查音频时长（c5: 跳过极短音频） ---
            try:
                wav_duration = get_wav_duration(wav_path)
            except Exception:
                wav_duration = None  # 无法读取时长时继续处理（非致命）

            if wav_duration is not None and wav_duration < MIN_AUDIO_DURATION:
                print(
                    f"  [跳过] 音频时长 {wav_duration:.2f}s 短于最小有效长度 "
                    f"{MIN_AUDIO_DURATION}s，已跳过此文件。"
                )
                # 清理临时文件
                try:
                    os.remove(wav_path)
                except OSError:
                    pass
                file_elapsed = time.time() - file_start
                print(f"  [耗时] {_format_elapsed(file_elapsed)}")
                continue  # 跳转到下一个文件，不计入成功或失败

            # --- 步骤 2: 转录 ---
            print(f"  [转录] 正在 GPU 转录（faster-whisper）...")
            t2 = time.time()
            result = transcriber.transcribe(wav_path)
            t3 = time.time()
            num_segs = len(result.get("segments", []))
            duration = result.get("duration", 0)
            detected_lang = result.get("language", "unknown")
            print(f"  [转录] 完成 ({_format_elapsed(t3 - t2)})")
            print(f"         -> {num_segs} 个段落, "
                  f"音频时长 {duration:.1f}s, "
                  f"语言: {detected_lang}")

            # --- 步骤 2.5: 检查转录结果是否为空（c4: 纯音乐/无声文件） ---
            if num_segs == 0:
                print(
                    f"  [警告] 转录结果为空！可能原因：\n"
                    f"         - 音频为纯音乐/纯器乐，不含人声\n"
                    f"         - 音频全程为静音\n"
                    f"         - 语言与模型不支持的语言不匹配\n"
                    f"         将生成空输出文件作为记录。"
                )

            # --- 步骤 3: 格式化输出 ---
            print(f"  [格式化] 正在生成输出文件...")
            t4 = time.time()
            output_files = save_transcript(result, file_path, config)
            t5 = time.time()
            print(f"  [格式化] 完成 ({_format_elapsed(t5 - t4)})")
            if output_files:
                for of in output_files:
                    print(f"         -> {of}")
            else:
                print(f"         -> （无输出文件生成）")

            # --- 清理临时 WAV 文件 ---
            try:
                os.remove(wav_path)
            except OSError:
                pass  # 清理失败不阻塞流水线

            file_elapsed = time.time() - file_start
            print(f"  [耗时] 文件总耗时: {_format_elapsed(file_elapsed)}")
            success_count += 1

        except FileNotFoundError as e:
            file_elapsed = time.time() - file_start
            print(
                f"  [失败] 文件不存在 ({_format_elapsed(file_elapsed)})\n"
                f"         原因: {e}\n"
                f"         操作: 已跳过此文件，继续处理下一个。"
            )
            fail_count += 1

        except ValueError as e:
            file_elapsed = time.time() - file_start
            print(
                f"  [失败] 参数/格式错误 ({_format_elapsed(file_elapsed)})\n"
                f"         原因: {e}\n"
                f"         操作: 已跳过此文件，继续处理下一个。"
            )
            fail_count += 1

        except RuntimeError as e:
            file_elapsed = time.time() - file_start
            print(
                f"  [失败] 运行时错误 ({_format_elapsed(file_elapsed)})\n"
                f"         原因: {e}\n"
                f"         操作: 已跳过此文件，继续处理下一个。"
            )
            fail_count += 1

        except Exception as e:
            file_elapsed = time.time() - file_start
            print(
                f"  [失败] 未知错误 ({_format_elapsed(file_elapsed)})\n"
                f"         类型: {type(e).__name__}\n"
                f"         原因: {e}\n"
                f"         操作: 已跳过此文件，继续处理下一个。"
            )
            fail_count += 1

    # ---- 5. 汇总统计 ----
    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"[完成] 处理完毕")
    print(f"   成功: {success_count}  |  失败: {fail_count}  |  总计: {len(files)}")
    print(f"   总耗时: {_format_elapsed(total_elapsed)}")
    print(f"   输出目录: {os.path.abspath(config.output_output_dir)}")
    print(f"{'=' * 60}")

    sys.exit(0 if fail_count == 0 else 1)


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    main()
