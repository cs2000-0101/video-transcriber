# Video-Transcriber 语音转文字工具

> 🎙️ 一键将视频/音频批量转录为 Markdown 和 Word 文档，全程本地 GPU 加速，数据不出本机。

---

## 📖 项目简介

**Video-Transcriber** 是一个 Windows 平台的视频/音频批量转录工具。它从视频中提取音频（基于 ffmpeg），利用 **faster-whisper**（CTranslate2 后端）在 NVIDIA GPU 上执行高速语音识别，最终输出格式化的 **Markdown**（`.md`）和/或 **Word**（`.docx`）文档。

### 核心特性

- 🚀 **GPU 加速**：基于 faster-whisper + CTranslate2，RTX 4070 Laptop (8GB VRAM) 实测安全运行
- 🧠 **智能分段**：基于能量检测的静音边界分段，确保不在句子中间截断
- 🎯 **高准确率**：默认 `large-v3` 模型，中文识别效果业界领先
- 📝 **双格式输出**：Markdown 纯文本 + Word 格式化文档，可选时间戳
- 📂 **批量处理**：支持单文件、多文件、文件夹扫描一键转录
- 🔒 **本地运行**：所有处理在本机完成，无数据上传风险
- ⚙️ **中文配置**：`config.yaml` 全中文注释，清晰易懂

### 支持格式

**视频格式（10 种）：** MP4, MKV, AVI, MOV, WebM, FLV, WMV, M4V, TS, MTS

**音频格式（8 种）：** MP3, WAV, FLAC, AAC, OGG, M4A, WMA, OPUS

---

## 🖥️ 环境要求

### 最低配置

| 组件 | 要求 |
|------|------|
| **操作系统** | Windows 10 / 11（64 位） |
| **GPU** | NVIDIA 显卡，支持 CUDA（建议 6GB+ 显存） |
| **内存** | 8GB+ RAM |
| **磁盘空间** | 5GB 可用空间（模型文件约 3GB） |
| **Python** | 3.10 或更高版本 |
| **ffmpeg** | 需安装并加入系统 PATH |

### 推荐配置

| 组件 | 推荐 |
|------|------|
| **GPU** | NVIDIA RTX 4070 Laptop 8GB 或更高 |
| **模型** | `large-v3` + `float16` 精度 |
| **CUDA** | CUDA 12.x（随 ctranslate2 自动安装） |

### 前置软件

**1. Python 3.10+**

从 [python.org](https://www.python.org/downloads/) 下载安装，安装时勾选「Add Python to PATH」。

**2. ffmpeg**

- 从 [ffmpeg.org](https://ffmpeg.org/download.html) 下载 Windows 版本
- 解压后将 `bin` 目录添加到系统 PATH 环境变量
- 验证安装：打开终端运行 `ffmpeg -version`

**3. NVIDIA GPU 驱动**

- 确保已安装最新的 NVIDIA 显卡驱动（支持 CUDA）
- 验证：运行 `nvidia-smi` 查看驱动版本和 CUDA 版本

---

## 📥 安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/cs2000-0101/video-transcriber.git
cd video-transcriber
```

### 2. 创建虚拟环境（推荐）

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
venv\Scripts\activate
```

### 3. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

> ⚠️ **注意：** `ctranslate2` 会自动检测 CUDA 版本并安装对应的 CUDA 支持包。如果遇到安装问题，请参考 [常见问题](#-常见问题) 章节。

### 4. 首次运行（自动下载模型）

首次运行时，faster-whisper 会自动从 Hugging Face 下载模型文件到 `./models/` 目录：

```bash
python main.py your_video.mp4
```

模型文件约 3GB（`large-v3`），下载时间取决于网络速度。后续运行无需重复下载。

> 💡 如果下载速度慢，可设置 Hugging Face 镜像：
> ```bash
> set HF_ENDPOINT=https://hf-mirror.com
> ```

---

## 🚀 使用方法

```bash
# 单个文件
python main.py video.mp4

# 多个文件批量
python main.py meeting1.mp4 lecture2.mp3

# 扫描整个文件夹
python main.py ./videos/

# 指定输出目录（默认 ./out）
python main.py video.mp4 -o ./我的转录结果

# 混合：文件夹 + 单个文件 + 指定输出
python main.py ./videos/ extra.mp3 -o ./output
```

### 输出结果

转录结果默认保存在 `./out/` 目录，文件名与源文件同名（扩展名变为 `.md` 或 `.docx`）。

输出目录通过 `-o` 参数或 `config.yaml` 中的 `output.output_dir` 配置。

### 运行示例

```
============================================================
Video-Transcriber -- 视频/音频转录工具
============================================================
[配置] 模型: large-v3 (float16)
[配置] 语言: zh
[配置] 输出格式: md
[配置] 输出目录: ./out

[扫描] 共发现 1 个文件待处理:
   - E:\videos\meeting.mp4

[初始化] 正在加载转录模型...
────────────────────────────────────────────────────────────
[1/1] meeting.mp4
  [提取] 完成 (1.2s)
  [音频] 时长 22.5 分钟 (1348s)
  [转录] |████████████████████| 100% | 已完成
  [转录] 完成 (31.2s) -> 1116 个段落, 语言: zh
  [格式化] 完成 -> ./out/meeting.md

============================================================
[完成] 处理完毕
   成功: 1  |  失败: 0  |  总计: 1
   总耗时: 35.0s
   输出目录: E:\项目\语音转文字\out
============================================================
```

---

## ⚙️ 配置说明

所有配置项在 `config.yaml` 中，文件包含详细的中文注释。以下是关键配置项的简要说明：

### 模型配置 (`model`)

| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| `size` | Whisper 模型大小 | `large-v3` | tiny, base, small, medium, large-v2, large-v3 |
| `compute_type` | 计算精度 | `float16` | float16, int8_float16, int8 |
| `language` | 目标语言（空=自动检测） | `zh` | zh, en, ja, ko, 或留空 |
| `download_root` | 模型下载目录 | `./models` | 任意路径 |

**模型选择建议：**
- `large-v3`：最高准确率，中文效果最好（推荐）
- `medium`：平衡速度与准确率
- `small`：速度快，适合低配 GPU
- `tiny` / `base`：极速但准确率较低

### 音频分段 (`audio_splitting`)

| 参数 | 说明 | 默认值 | 范围 |
|------|------|--------|------|
| `max_segment_length` | 最大段长（秒） | `600` | 60 - 1800 |
| `silence_threshold` | 静音阈值（dBFS） | `-40` | -60 到 -20 |
| `min_silence_duration` | 最小静音时长（ms） | `500` | 200 - 2000 |
| `overlap_duration` | 段间重叠（ms） | `200` | 0 - 1000 |

> 💡 调节 `silence_threshold`：更负的值（如 `-50`）更敏感，更容易检测到静音分段点。如果转录结果出现不合理的断句，可以适当调小（如 `-35`）。

### 输出配置 (`output`)

| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| `format` | 输出格式 | `both` | md, docx, both |
| `with_timestamps` | 是否包含时间戳 | `true` | true, false |
| `output_dir` | 输出目录 | `./out` | 任意路径 |

### 临时文件 (`temp_dir`)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `temp_dir` | 临时音频存放目录 | `./temp` |

临时 WAV 文件在转录完成后自动清理。

---

## ❓ 常见问题

### Q1: 提示 "未找到 ffmpeg 命令"

**A:** 需要安装 ffmpeg 并将其 `bin` 目录添加到系统 PATH：
1. 从 https://ffmpeg.org/download.html 下载 Windows 版本
2. 解压到 `C:\ffmpeg`
3. 将 `C:\ffmpeg\bin` 添加到系统环境变量 PATH
4. 重新打开终端，运行 `ffmpeg -version` 验证

### Q2: CUDA/GPU 不可用，如何用 CPU 转录？

**A:** 修改 `config.yaml` 中的 `compute_type` 为 `int8`，然后在代码中将 `device="cuda"` 改为 `device="cpu"`。CPU 转录速度约为 GPU 的 1/5 ~ 1/10。

### Q3: 首次运行下载模型很慢

**A:** 可设置 Hugging Face 镜像加速：
```bash
set HF_ENDPOINT=https://hf-mirror.com
python main.py video.mp4
```
或手动下载模型文件放到 `./models/` 目录中。

### Q4: 显存不足（CUDA out of memory）

**A:** 尝试以下方法：
1. 将 `model.compute_type` 改为 `int8_float16`（降低显存占用）
2. 将 `model.size` 改为 `medium` 或 `small`（更小的模型）
3. 减小 `audio_splitting.max_segment_length`（如改为 `300`）

### Q5: 转录结果为空

**A:** 可能原因：
- 音频为纯音乐/纯器乐，不含人声
- 设置了错误的语言（如中文音频却设置了 `language: en`）
- 环境噪音过大导致 VAD 过滤了所有片段

尝试将 `model.language` 设为空字符串 `""` 以启用自动语言检测。

### Q6: 转录中文准确率不高

**A:** 
1. 确保使用 `large-v3` 模型（`model.size: large-v3`）
2. 设置 `model.language: zh`
3. 确保音频质量清晰，减少背景噪音
4. 检查原始视频/音频的采样率是否足够（建议 44.1kHz 以上原始音源）

### Q7: 支持哪些语言？

**A:** faster-whisper 支持 99 种语言，包括中文、英语、日语、韩语、法语、德语、西班牙语等。设置 `model.language: ""` 可启用自动语言检测。

### Q8: 转录中途崩溃了怎么办？

**A:** 工具设计为逐文件处理，单个文件失败不会影响其他文件。程序会打印详细的错误信息，帮助定位问题。临时文件会在处理完成后自动清理。

### Q9: 如何贡献代码？

**A:** 欢迎提交 Issue 和 Pull Request！请先阅读 `config.yaml` 了解项目配置，代码结构请参考 `main.py` 和各 `src/` 模块。

### Q10: 可以商用吗？

**A:** 本项目使用 MIT 协议开源。但请注意 faster-whisper 模型本身有自己的使用条款，请参考 [OpenAI Whisper 模型许可](https://github.com/openai/whisper)。

---

## 📁 项目结构

```
video-transcriber/
├── main.py                # CLI 入口：参数解析 · 批量调度 · 流水线编排
├── config.yaml            # 配置文件（全中文注释）
├── requirements.txt       # Python 运行时依赖
├── README.md              # 项目说明（本文件）
├── .gitignore             # Git 忽略规则
└── src/
    ├── __init__.py        # 包标记
    ├── config.py          # 配置加载与验证模块
    ├── extractor.py       # 音频提取器（ffmpeg → 16kHz mono WAV）
    ├── transcriber.py     # 转录引擎（faster-whisper GPU + 静音分段）
    └── formatter.py       # 输出格式化器（.md + .docx）
```

---

## 📄 许可协议

本项目采用 [MIT License](https://opensource.org/licenses/MIT) 开源。

---

*Made with ❤️ for local-first transcription.*
