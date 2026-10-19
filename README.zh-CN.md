<div align="center">

<img src="https://raw.githubusercontent.com/kanaworksai-hue/sonicproof/main/docs/assets/banner-en.jpg" alt="Sonicproof：检查、挑选、质检 AI 生成的声音" width="820">

<p>
  <a href="https://github.com/kanaworksai-hue/sonicproof/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-6b4dff?style=flat-square"></a>
  <a href="https://pypi.org/project/sonicproof/"><img alt="PyPI" src="https://img.shields.io/pypi/v/sonicproof?style=flat-square&logo=pypi&logoColor=white&color=3775a9"></a>
  <img alt="Python 3.11 to 3.13" src="https://img.shields.io/badge/python-3.11–3.13-3776ab?style=flat-square&logo=python&logoColor=white">
  <img alt="macOS, Linux, Windows" src="https://img.shields.io/badge/platform-macOS_·_Linux_·_Windows-111?style=flat-square">
  <a href="#在-ai-agent-里使用"><img alt="MCP server" src="https://img.shields.io/badge/MCP-server-19b8d6?style=flat-square"></a>
  <a href="https://huggingface.co/google/embeddinggemma-2"><img alt="Model: EmbeddingGemma 2" src="https://img.shields.io/badge/model-EmbeddingGemma_2-ff9d00?style=flat-square&logo=huggingface&logoColor=white"></a>
  <a href="https://github.com/kanaworksai-hue/sonicproof/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/kanaworksai-hue/sonicproof/ci.yml?branch=main&style=flat-square&label=CI"></a>
  <a href="https://github.com/kanaworksai-hue/sonicproof/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/kanaworksai-hue/sonicproof?style=flat-square&logo=github&label=stars&color=f5c518"></a>
</p>

<p><a href="https://github.com/kanaworksai-hue"><img src="https://raw.githubusercontent.com/kanaworksai-hue/sonicproof/main/docs/brand/kanaworks-mark.png" height="22" alt="" align="absmiddle"></a>&nbsp;<a href="https://github.com/kanaworksai-hue"><b>A KanaWorks_AI project</b></a></p>

<p><a href="https://github.com/kanaworksai-hue/sonicproof/blob/main/README.md">English</a> · <b>简体中文</b> · <a href="https://github.com/kanaworksai-hue/sonicproof/blob/main/README.ja.md">日本語</a></p>

[安装](#安装) · [用法](#用法) · [工作台](#工作台) · [AI agent](#在-ai-agent-里使用) · [怎么看分数](#怎么看分数) · [⭐ Star](#喜欢就点个-star)

</div>

**AI 生成的声音，真的是你要的那个吗？**

Sonicproof 是一个在本机运行的生成音频质检工具，适用于 Stable Audio 等模型生成的音效、音乐和环境声。
它用一个多语言模型来“听”——Google 的 [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2)，
这个模型把文字、声音、图片和视频放进同一个向量空间：

- **check**——这段声音听起来像提示词吗？提示词可以用任何语言写。
- **rank**——同一个提示词生成了四个 seed，留哪一个？
- **qc**——技术上干净吗？响度、真峰值、削波、直流偏移、静音、尾音被截断、咔嗒声、相位、循环接缝——
  每个问题都附一句修复建议。
- **fit**——这个声音配这个画面吗？可以是一帧截图、一张概念图、一段视频，或者视频自带的音轨。
- **search**——在你自己的音效库里按文字（任何语言）、按示例声音或按图片找声音。
- **dedupe**——找出库里重新编码、改过音量、裁过头尾的重复副本，并告诉你该留哪一个。

全部在你的电脑上运行——不上传、不需要 API key。可以在浏览器里的本地[工作台](#工作台)用，
可以让[你的 AI agent](#在-ai-agent-里使用) 用（MCP server + agent skill），也可以在命令行和 [Python](#python) 里用。

![Sonicproof 工作台：生成的多个 take 按提示词分组，每组最好的一条标着“Use this one”](https://raw.githubusercontent.com/kanaworksai-hue/sonicproof/main/docs/images/workbench.jpg)

```text
$ sp check dog_bark.wav "狗在叫"
╭─ dog_bark.wav ───────────────────────────────────────────────────────╮
│                                                                      │
│   STRONG   +3.25σ  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━                      │
│  clearly sounds like the prompt                                      │
│                                                                      │
│    prompt  “狗在叫”                                                  │
│   ranking  #3 of 438 descriptions for this clip                      │
│  evidence  cos 0.742 · clip-side +3.60σ · prompt-side +2.91σ         │
│                                                                      │
│  Heard as                                                            │
│  +3.79σ  a dog barking  ✓  animals                                   │
│  +3.37σ  a small dog yapping  animals                                │
│  +3.10σ  a dog whining  ✓  animals                                   │
│  +3.03σ  a dog growling  ✓  animals                                  │
│  +2.26σ  a duck quacking  animals                                    │
│                                                                      │
╰──────────────────────────────────────────────────────────────────────╯
```

命令行的输出和声音描述词表是英文的；提示词用中文写没有问题。

## 安装

```bash
uv tool install "sonicproof[model]"        # 或者：pip install "sonicproof[model]"
sonicproof doctor --load                   # 下载一次模型，并检查环境
```

- **Python 3.11 及以上**（在 3.11–3.13 上测试过）。Apple Silicon（MPS）、CUDA 和 CPU 都能跑。`sp` 是 `sonicproof` 的简写。
- **模型**——[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2)，约 1.5 GB——第一次使用时从
  Hugging Face 下载到它的默认缓存目录。不需要 Hugging Face 账号或 token。
- **ffmpeg** 只在读取 M4A / AAC / MP4 等 libsndfile 不支持的格式，以及 `fit` / `search` 处理视频时需要。
  WAV、FLAC、OGG、MP3 不需要它。
- **Windows + NVIDIA 显卡：** PyPI 上的 Windows 版 PyTorch 只支持 CPU。请先按
  [pytorch.org](https://pytorch.org/get-started/locally/) 在一个 Python 环境里安装带 CUDA 的 PyTorch，再在同一个环境里
  `pip install "sonicproof[model]"`（`uv tool install` 和 `uvx` 会另建环境，装上的是 CPU 版）。
- **只做技术检查？** `qc` 不需要模型：`pip install sonicproof`（不带 `[model]`）就够了。
- **想用最新的 `main`：** `git clone https://github.com/kanaworksai-hue/sonicproof`，在里面运行
  `uv sync --extra model`，然后用 `uv run sp …` 执行命令（在其它目录下用 `uv run --project /path/to/sonicproof sp …`）。

## 用法

```bash
# 提示词遵从度。不写提示词时，从 Stable Audio 的 .json sidecar 文件里读。
sp check explosion.wav "巨大的爆炸，碎片四溅"
sp check thunder.wav

# 挑出最好的一条
sp rank "木门吱呀一声打开" door_v1.wav door_v2.wav door_v3.wav door_v4.wav

# 技术检查（不需要模型）。有任何一项不合格就以退出码 1 结束，方便写脚本。
sp qc *.wav
sp qc ambience_loop.wav --loop --target-lufs -16

# 所有结果合成一张评分卡：SHIP / REVIEW / REDO。加 --strict 时，不是 SHIP 就以退出码 1 结束。
sp report footsteps.wav "踩在碎石路上的脚步声"
sp report footsteps.wav "踩在碎石路上的脚步声" --strict

# 这个声音配这个画面吗？对场景的文字说明可加可不加。
sp fit sizzle.wav --image kitchen.png
sp fit rumble.wav --video storm.mp4 "暴风雨夜，雷声"
sp fit trailer.mp4                          # 用视频自带的音轨对它自己的画面
```

### 你的音效库

```bash
sp index ~/SFX                              # 建一次索引；以后只读新增或改过的文件
sp search "雨打在铁皮屋顶上" --in ~/SFX
sp search --like take_03.wav --in ~/SFX     # 和这条听起来像的
sp search --image frame.png --in ~/SFX      # 适合这张图的
sp dedupe ~/SFX                             # 完全重复和近似重复（不会删除任何文件）
```

音效库的文件夹必须明确指定：`index DIR`、`search … --in DIR`、`dedupe DIR`（家目录、磁盘根目录和系统目录会被拒绝——
请指向放音效的那个文件夹）。索引存在 `~/SFX/.sonicproof/`；加 `--in-cache` 则改存到 Sonicproof 的缓存里，
不往文件夹里写任何东西（只读文件夹会自动使用缓存）。`search` 和 `dedupe` 会自动更新索引，所以 `sp index` 可以不跑。

### 写脚本：`--json` 和退出码

分析类命令——`check`、`rank`、`qc`、`report`、`fit`、`index`、`search`、`dedupe`——都支持 `--json`
（`doctor`、`serve`、`mcp`、`skill` 不支持）。`qc --json` 永远输出数组，每个文件一条记录，只有一个文件也一样。
在 `qc` 和 `rank` 里，读不了的文件会标为 `"error"`，其它文件照常处理，此时退出码为 2。

| 退出码 | 含义 |
|---|---|
| `0` | 完成。`check` / `rank` / `fit` 不论判定结果都返回 0——请读判定结果。 |
| `1` | `qc` 有不合格项、`report --strict` 的结论不是 SHIP，或 `doctor` 有检查项失败。 |
| `2` | 输入有问题：文件不存在或读不了、音频为空、参数值不对。 |
| `3` | 环境有问题：没装模型 extra、模型没下载且处于离线、需要 ffmpeg、端口被占用。 |

加 `--json` 时，错误也是 JSON——在 stdout 输出 `{"error": {"code": "not_found", "message": "…", "file": "…"}}`
——脚本不用去解析 traceback。例外是命令行用法错误（缺参数、选项值超出范围）：仍然输出普通的用法提示，退出码为 2。
设置 `SONICPROOF_DEBUG=1` 可以看完整的 traceback。

## 工作台

```bash
sonicproof serve                            # 打开 http://localhost:7788
```

- **Inbox（收件箱）**——拖入音频文件或整个文件夹。提示词从 Stable Audio 的 `.json` sidecar 文件（与音频同名）里读（也可以自己输入）。
  同一提示词的多个 take 自动分组、排名，最好的一条标上 **Use this one**。如果一组里没有一条像样的，
  它会直接告诉你，并建议换一种写法。每张卡片显示叠在频谱图上的波形、一句话结论（◎ Very close · ○ Close ·
  △ Somewhat · ✕ Not a match）、0–100 的一致度条和每项技术检查的指示点；技术问题用通俗的话说明，并给出修复建议。
- **Library（音效库）**——选一个文件夹，建索引（有进度显示），然后按文字、拖入一个声音或按图片搜索。
  **Duplicates** 页列出重复副本和该留哪个（不会删除任何文件）。工作台的索引都放在缓存里，打开文件夹不会往里面写东西。
- **Picture（画面）**——拖入图片或视频：用声音描述词看看画面里有什么，用它给收件箱里的声音打分，在音效库里找配得上的声音，
  或者评判视频自带的音轨。

每个结论背后的原始数字点开 **Details** 就能看到，或者在 **Settings → Pro mode** 里全局显示。
设置里还能看缓存占用并清理；从收件箱删除文件，只会删除拖入时复制的副本（你的原文件不会被动到）。

界面支持英文和日文，默认跟随浏览器语言，也可以手动切换（暂无中文界面）。只有你按下播放时才会发出声音。
服务只监听 `localhost`，拒绝来自其它网站的请求。

![日文界面的工作台](https://raw.githubusercontent.com/kanaworksai-hue/sonicproof/main/docs/images/workbench-ja.jpg)

## 在 AI agent 里使用

最快的办法：把这个仓库的链接发给你的 agent，让它检查自己生成的声音。它需要的一切都在这一节里。

**1. 在终端里安装，并下载一次模型**（见[安装](#安装)）。MCP server 不会在工具调用里启动 1.5 GB 的下载——
那看起来就像卡死了——所以请确保 `sonicproof doctor --load` 已经跑过一次。也可以不做永久安装，直接：

```bash
uvx --from "sonicproof[model]" sonicproof doctor --load
```

**2. 添加 MCP server**——工具有 `check`、`rank`、`qc`、`report`、`fit`、`search`、`dedupe`、`index`，
每个工具的返回值和对应 CLI 命令加 `--json` 的输出相同。模型只加载一次，之后常驻。

```bash
# Claude Code
claude mcp add sonicproof -e HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" -- uvx --from "sonicproof[model]" sonicproof mcp
# Codex
codex mcp add sonicproof --env HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" -- uvx --from "sonicproof[model]" sonicproof mcp
# 任何客户端：streamable HTTP，地址 http://127.0.0.1:7789/mcp
sonicproof mcp --http --port 7789
```

其它客户端在 JSON 配置里写同样的命令（缓存目录请写成绝对路径）：

```json
{ "mcpServers": { "sonicproof": {
    "command": "uvx", "args": ["--from", "sonicproof[model]", "sonicproof", "mcp"],
    "env": { "HF_HOME": "/Users/you/.cache/huggingface" } } } }
```

- **为什么要传 `HF_HOME`：** MCP 客户端启动 server 时只给很少的环境变量。传了它，server 就和你的终端用同一个
  Hugging Face 缓存，不会再下载一遍模型。
- **工具调用里请用绝对路径。** 相对路径是相对 server 的工作目录解析的，通常不是你的目录。
- **HTTP 模式**只监听 `127.0.0.1`，并拒绝不是来自 localhost 的网页发来的请求，但没有认证：本机上的任何进程都能调用它。
  介意的话请用 stdio。
- **用 `main` 的克隆运行：** server 命令写成 `uv run --project /absolute/path/to/sonicproof --extra model sonicproof mcp`
  （`--project` 保留调用方的工作目录；`--directory` 会改掉它）。

**3. 添加 agent skill**——它教 agent 一套让生成音频靠得住的流程：*生成多个 take → 排名 → 检查最好的一条 →
都不像就改写提示词或换 seed → 交付选中的那条和它的评分卡*。它随包发布：

```bash
sp skill install              # → ~/.claude/skills/sonicproof/SKILL.md
sp skill install --project    # → ./.claude/skills/sonicproof/SKILL.md（只对当前项目）
sp skill show                 # 打印出来，给从别处读取 skill 的 agent
```

在克隆的仓库里它位于 [`src/sonicproof/skill/SKILL.md`](https://github.com/kanaworksai-hue/sonicproof/blob/main/src/sonicproof/skill/SKILL.md)。
[`examples/stable_audio_loop.py`](https://github.com/kanaworksai-hue/sonicproof/blob/main/examples/stable_audio_loop.py)
用脚本跑同样的流程：一个 MCP 客户端连 Stable Audio server，一个连 Sonicproof。

**如果你是正在读这个仓库的 agent：** 完成第 1–3 步，然后按 skill 做（`sp skill show` 可以打印出来）。
没有 MCP 时，所有分析命令都能在 shell 里加 `--json` 使用，例如 `sp rank "prompt" /abs/a.wav /abs/b.wav --json`。
请传本地文件的绝对路径；不要靠播放音频来“听”——Sonicproof 就是你的耳朵。

## 怎么看分数

每段声音都会得到四种判定之一。工作台用大白话和 0–100 的一致度条显示；CLI 和 `--json` 给出判定名和背后的原始分数。

| 工作台 | 判定 | 原始分数 | 含义 |
|---|---|---|---|
| ◎ Very close | **strong** | ≥ 2.0 | 明显就是提示词说的声音 |
| ○ Close | **plausible** | ≥ 1.1 | 听起来像提示词说的声音 |
| △ Somewhat | **weak** | ≥ 0.3 | 只是有点像 |
| ✕ Not a match | **miss** | < 0.3 | 听起来不像提示词说的声音 |

原始分数怎么来的：提示词和声音之间的直接相似度没什么信息量——几乎任何描述对任何声音的分数都差不多——
所以 Sonicproof 问两个相对的问题，再取平均：

1. **声音侧**——和约 440 条日常声音描述相比，*你的提示词*对*这段声音*的描述有多准？
2. **提示词侧**——和 110 条参考声音相比，*这段声音*离*你的提示词*近了多少？

两者都以“高出基准多少个标准差”来衡量，所以 CLI 显示的分数形如 `+3.79σ`。工作台的一致度条把 −1 … +3 映射到 0 … 100。

**Heard as（听起来像）** 列出这段声音最像什么；✓ 表示和你的提示词意思一致。判定结果出乎意料时，这个列表通常能解释原因。
静音的声音（没有任何部分高于 −60 dBFS）永远不会被判为一致：它们得到 `"verdict": "miss"` 和 `"note": "silent"`，
排在最后。UI 点击这类很短的单发音效照常判定。

Sonicproof 是第二意见，不是裁判：简短具体的提示词（“玻璃碎裂”）比塞满混音术语的长提示词效果好，
非常短促的瞬态声音最难区分。交付之前请一定亲耳听一遍。

### fit

`fit` 综合两方面的证据，校准方式相同，用同一套判定标准：

- **直接**——画面离*这段声音*，比离那些参考声音近了多少？
- **经由描述**——先把画面描述成文字（**Pictured as / 画面像**），再把这些描述当作提示词去检查声音。
  你自己写的场景说明也会计入。

**Pictured as** 里的 ✓ 表示声音也听起来像这一项；**Heard as** 里的 ✓ 表示画面里也有这一项。
判定为 weak 或 miss 时，Sonicproof 会根据画面建议一个提示词——重新生成时很有用。
视频按每秒一帧取样（最多 32 帧，较长的视频在整段上均匀取样）。

### dedupe

两段声音意思相近，不一定是同一个文件——比如同一提示词的四个 seed。所以 `dedupe` 只用嵌入找候选，
然后比较信号本身（时间对齐的频谱指纹）。这样，格式、编码、音量、采样率、声道数不同或裁过头尾的副本会被归为一组，
不同的 take 则不会。每组里它会选出建议保留的那份——无损优先于有损、没有削波、频谱更完整
（只有真的包含更多内容时，更高的采样率才算数，升采样的副本不算）、声道更多、更长、没被调小音量——并说明理由。

## Python

```python
from sonicproof.judge import Judge

judge = Judge(vision=True)    # 一个模型搞定所有功能；只用 Judge() 更轻量，但看不了图片
result = judge.check("door.wav", "吱呀作响的木门")
print(result.verdict, result.score, [a.text for a in result.heard_as])

fit = judge.fit("sizzle.wav", "kitchen.png", "image")
print(fit.verdict, [a.text for a in fit.pictured_as])

from sonicproof.library import Library, find_duplicates, search_text

lib = Library("~/SFX")
lib.update(lambda wave: judge.embedder.embed_audio([wave])[0])
for hit in search_text(lib, judge, "雨打在铁皮屋顶上"):
    print(hit.path, hit.score, hit.verdict)
for group in find_duplicates(lib):
    print(group.keep, group.reason)
```

## 开发

```bash
uv sync --extra model --group dev && uv run pytest
cd web && npm ci && npm run dev             # 带热重载的界面（:5173）；/api 代理到 :7788 上的 `sp serve`
npm run build                               #（在 web/ 里）把打包好的界面写进 src/sonicproof/web
```

界面文字在 `web/src/i18n/`（`en.ts`、`ja.ts`）；类型检查会保证两边的键一致。
测试、lint 和 pull request 请看 [CONTRIBUTING.md](https://github.com/kanaworksai-hue/sonicproof/blob/main/CONTRIBUTING.md)（英文）。

## 许可证

Sonicproof 使用 [MIT 许可证](https://github.com/kanaworksai-hue/sonicproof/blob/main/LICENSE)（另见 [NOTICE](https://github.com/kanaworksai-hue/sonicproof/blob/main/NOTICE)）。
它在第一次运行时从 Hugging Face 下载 EmbeddingGemma 2——模型不随包附带。该模型由 Google 以 Apache 2.0 发布，
其模型卡要求部署时遵守 [Gemma 禁止使用政策](https://ai.google.dev/gemma/prohibited_use_policy)。

KanaWorks_AI 的名称、标志和 Kana 角色版权归 KanaWorks_AI 所有，**不在** MIT 许可证范围内。
详见 [docs/brand](https://github.com/kanaworksai-hue/sonicproof/blob/main/docs/brand/README.md)。

## 喜欢就点个 Star

<div align="center">

<a href="https://x.com/KanaWorks_AI"><img src="https://raw.githubusercontent.com/kanaworksai-hue/sonicproof/main/docs/brand/kana-wave.png" width="150" alt="KanaWorks_AI 的吉祥物 Kana 在挥手"></a>

<p><b>Made with ♥ by <a href="https://x.com/KanaWorks_AI">KanaWorks_AI</a></b></p>

<p><sub>如果 Sonicproof 帮你省下了一遍试听，点个 ⭐ 能让更多人找到它。更多项目：<a href="https://github.com/kanaworksai-hue">github.com/kanaworksai-hue</a></sub></p>

</div>
