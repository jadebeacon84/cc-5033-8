# 水下纪录片时序降噪工具

运动补偿加权的时域融合，带运动失效回退、独立空间/时间控制，以及一个可在容器内启动的
中文校正界面。输出为真实可播放视频（时间戳取自源片 pts，不重排节奏、不填补缺帧）。

## 启动界面

```bash
# 项目根目录
./tools/run_ui.sh                # 打开 http://127.0.0.1:8765/
./tools/run_ui.sh 9000           # 指定端口
PYTHON=/usr/bin/python3 ./tools/run_ui.sh
```

脚本会先检查 numpy/opencv 与 ffmpeg，然后启动 `python3 -m denoise.server <port>`（仅用
标准库的 HTTP 服务，后台单工作线程逐条处理任务）。若解释器里没有 numpy/opencv，脚本会用
`tools/fetch_deps.py` 一次性把对应 manylinux wheel（含 imageio-ffmpeg 自带的 ffmpeg）解包到
`.tools/pylib`（约 265 MB，校验 PyPI 记录的 SHA-256，无需 pip），之后启动直接复用。
浏览器打开打印出的地址即可，左侧选片段、
调时窗与强度、点“开始降噪”。没有素材时执行：

```bash
python3 tools/make_clips.py
```

依赖见 `requirements.txt`；ffmpeg 也可由 `imageio-ffmpeg` 提供，或用 `FFMPEG_BIN` 指定。
ffprobe 用于读取真实 pts：优先 PATH，其次 ffmpeg 同目录，最后仓库内 `.tools/ffprobe`。

## 界面里能做什么

- **对比播放**：并排 / 拖动分屏 / 差异放大；流可切 原始输入、降噪输出、清晰真值；
  拖动按源时间轴逐帧，播放速率 0.5× / 1×（真实节奏）/ 2×。
- **运动 / 邻帧 / 残差**：运动场（固定尺度：满亮≈4 px/帧，暗=无效）、有效区域、采用权重、
  残差放大（色标按噪声 σ 固定，σ≈噪声、≥4σ≈真实错配），并列出每个邻帧的权重/覆盖/残差中位
  与采用还是丢弃及原因。
- **客观指标**：PSNR、SSIM、平坦区 σ、细线细节保留、时域闪烁，附 PSNR 曲线；只对带清晰真值的
  合成片段计算。
- **逐帧诊断**：回退原因、有效比例、时域证据、切镜/缺帧提示与中文警告。

时间轴下方色条标出切镜（黄）、缺帧槽位（红）、回退帧（褐）；缺帧槽位按源节奏留空，不插内容。

## 算法要点

1. **噪声强度来源**：合成片段由生成器注入逐通道 i.i.d. 高斯噪声，σ 写入 manifest，界面
   “噪声假设来源”显示 `origin=manifest` 与具体 σ；真实素材无 manifest 时用 8×8 平坦块 MAD 的
   低分位稳健估计，并标明 `origin=flat_patch_estimate`。两条路径都不假装知道真值。
2. **运动估计**：稠密 Farneback（必须以 0–255 uint8 输入，否则 OpenCV 会给出近似零场）→ 全局
   相位相关 → identity → 丢弃。有效 = 越界/残差/置信度共同判定，再中值清理。
3. **融合权重**：时间强度 × 距离衰减 × 有效区 × 残差软门（指数衰减）× 置信度，邻帧权重上限
   0.85，自身权重下限 1.0。**整候选丢弃规则**：若中位残差超过门限（例如切镜处稠密场编造出
   看似合理的 warp），整个邻帧不参与，绝不把错位帧平均成重影。
4. **空间/时间分离**：空间阶段独立强度（0 = 直接通过），按结构提示衰减，避免把整幅图涂平。
5. **时间轴**：解码真实 pts；间距 > 1.6 名义帧间隔处插入不可用槽位；输出保持源时间戳。
   CFR 走 rawvideo 管道，VFR 走 concat demuxer 且在每个 `file` 后写 `option framerate`。
   容器常把 pts 四舍五入到毫秒（24 fps 呈 42/41 ms 交替），因此“是否均匀”按与中位间隔的
   相对偏差（≤2%）判定，否则每条 CFR 片段都会走 VFR 并被重采样成 23.81 fps、多出一帧。

## 核验（客观、非缩略图）

`tests/` 与三份合成素材覆盖：噪声下降、细节保留、切镜不重影、缺帧/边界回退、时间戳保持、
低纹理场景。运行：

```bash
# 若系统解释器没有 pytest/依赖：python3 tools/fetch_deps.py --test   # 一次性
PYTHONPATH=.tools/pylib:. python3 -m pytest tests -q
```

界面“客观指标”对每条任务给出同口径数据（`outputs/jobs/<id>/metrics.json`）。例：

| 片段 | PSNR 输入→输出 | SSIM 输入→输出 | 平坦区 σ 输入→输出 | 细节保留 |
| --- | --- | --- | --- | --- |
| reef_sway_24fps | 27.99 → 38.05 dB | 0.603 → 0.943 | 0.0225 → 0.0063 | 0.755 |
| low_texture_silt | 24.64 → 35.22 dB | 0.415 → 0.895 | 0.0337 → 0.0085 | 0.714 |
| grading_timeline | 25.28 → 35.57 dB | 0.455 → 0.902 | 0.0310 → 0.0083 | 0.720 |

细节保留 = 输出/输入的细线对比度，接近 1 表示没有把纹理抹平；σ 下降与细节保留需同时成立才
算降噪成功。以上只说明噪声被压低且结构仍在，**不声称恢复源片本来没有拍到的细节**。

## 输出

每个任务目录 `outputs/jobs/<id>/`：

- `denoised.mp4` — 可播放交付（CRF 12，源时间戳）
- `denoised_lossless.mkv` — FFV1 无损，用于客观比较
- `input_proxy.mp4` / `truth_proxy.mp4` — 同时间轴的输入与真值对照
- `metrics.json` / `summary.json` — 逐帧指标、时间轴、回退与警告
- 界面底部给出下载链接，视频接口支持 HTTP Range（206），可拖动进度播放。

## 目录

```
denoise/   内核（motion / fusion / pipeline / media / metrics / server）
web/       界面（index.html, app.js, style.css, 内嵌中文字体子集）
media/clips/  合成素材 + manifest
tools/     run_ui.sh, fetch_deps.py, make_clips.py, rebuild_fonts.py
tests/     pytest 核验
```
