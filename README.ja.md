<div align="center">

<img src="https://raw.githubusercontent.com/kanaworksai-hue/sonicproof/main/docs/assets/banner-en.jpg" alt="Sonicproof：AI が生成した音を検査・選別・品質チェック" width="820">

<p>
  <a href="https://github.com/kanaworksai-hue/sonicproof/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-6b4dff?style=flat-square"></a>
  <a href="https://pypi.org/project/sonicproof/"><img alt="PyPI" src="https://img.shields.io/pypi/v/sonicproof?style=flat-square&logo=pypi&logoColor=white&color=3775a9"></a>
  <img alt="Python 3.11 to 3.13" src="https://img.shields.io/badge/python-3.11–3.13-3776ab?style=flat-square&logo=python&logoColor=white">
  <img alt="macOS, Linux, Windows" src="https://img.shields.io/badge/platform-macOS_·_Linux_·_Windows-111?style=flat-square">
  <a href="#ai-エージェントから使う"><img alt="MCP server" src="https://img.shields.io/badge/MCP-server-19b8d6?style=flat-square"></a>
  <a href="https://huggingface.co/google/embeddinggemma-2"><img alt="Model: EmbeddingGemma 2" src="https://img.shields.io/badge/model-EmbeddingGemma_2-ff9d00?style=flat-square&logo=huggingface&logoColor=white"></a>
  <a href="https://github.com/kanaworksai-hue/sonicproof/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/kanaworksai-hue/sonicproof/ci.yml?branch=main&style=flat-square&label=CI"></a>
  <a href="https://github.com/kanaworksai-hue/sonicproof/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/kanaworksai-hue/sonicproof?style=flat-square&logo=github&label=stars&color=f5c518"></a>
</p>

<p><a href="https://github.com/kanaworksai-hue"><img src="https://raw.githubusercontent.com/kanaworksai-hue/sonicproof/main/docs/brand/kanaworks-mark.png" height="22" alt="" align="absmiddle"></a>&nbsp;<a href="https://github.com/kanaworksai-hue"><b>A KanaWorks_AI project</b></a></p>

<p><a href="https://github.com/kanaworksai-hue/sonicproof/blob/main/README.md">English</a> · <a href="https://github.com/kanaworksai-hue/sonicproof/blob/main/README.zh-CN.md">简体中文</a> · <b>日本語</b></p>

[インストール](#インストール) · [使い方](#使い方) · [ワークベンチ](#ワークベンチ) · [AI エージェント](#ai-エージェントから使う) · [スコアの読み方](#スコアの読み方) · [⭐ Star](#気に入ったら-star-を)

</div>

**AI が生成した音、ちゃんと頼んだとおりになっていますか？**

Sonicproof は、生成オーディオのためのローカル QA ツールです。Stable Audio などで作った効果音・音楽・環境音を対象にします。
音を「聴く」のは多言語モデル 1 つだけです。Google の [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) で、
テキスト・音・画像・動画を同じ空間で扱います。

- **check**：このクリップはプロンプトどおりに聞こえるか？　プロンプトはどの言語でも書けます。
- **rank**：シードを 4 つ生成した。どれを残すべきか？
- **qc**：技術的にきれいか？　ラウドネス、トゥルーピーク、クリッピング、DC オフセット、無音、尻切れ、クリックノイズ、
  位相、ループのつなぎ目。問題ごとに直し方を一行で示します。
- **fit**：この音はこの映像に合うか？　1 フレーム、コンセプトアート、動画クリップ、または動画自身の音声トラック。
- **search**：手元の音源ライブラリを、言葉（どの言語でも）・音の例・画像で検索。
- **dedupe**：ライブラリに散らばる再エンコード・音量違い・頭尾カットのコピーを見つけ、どれを残すべきかを提案。

すべて手元のマシンで動きます。アップロードも API キーも不要です。ブラウザで開くローカルの[ワークベンチ](#ワークベンチ)、
[AI エージェント](#ai-エージェントから使う)（MCP サーバー + エージェントスキル）、コマンドライン、[Python](#python) から使えます。

![日本語表示の Sonicproof ワークベンチ](https://raw.githubusercontent.com/kanaworksai-hue/sonicproof/main/docs/images/workbench-ja.jpg)

```text
$ sp check dog_bark.wav "犬が吠えている"
╭─ dog_bark.wav ───────────────────────────────────────────────────────╮
│                                                                      │
│   STRONG   +3.64σ  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━                      │
│  clearly sounds like the prompt                                      │
│                                                                      │
│    prompt  “犬が吠えている”                                          │
│   ranking  #2 of 438 descriptions for this clip                      │
│  evidence  cos 0.764 · clip-side +4.20σ · prompt-side +3.08σ         │
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

コマンドラインの出力と音の説明リストは英語ですが、プロンプトは日本語のままで大丈夫です。

## インストール

```bash
uv tool install "sonicproof[model]"        # または：pip install "sonicproof[model]"
sonicproof doctor --load                   # モデルを一度だけダウンロードし、環境をチェック
```

- **Python 3.11 以上**（3.11〜3.13 でテスト済み）。Apple Silicon（MPS）、CUDA、CPU のどれでも動きます。`sp` は `sonicproof` の短縮コマンドです。
- **モデル**：[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2)（約 1.5 GB）は初回使用時に
  Hugging Face から通常のキャッシュへダウンロードされます。Hugging Face のアカウントやトークンは不要です。
- **ffmpeg** が必要なのは、M4A / AAC / MP4 など libsndfile が読めない形式と、`fit` / `search` で動画を扱うときだけです。
  WAV、FLAC、OGG、MP3 はなくても読めます。
- **Windows + NVIDIA GPU：** PyPI の Windows 版 PyTorch は CPU 専用です。先に
  [pytorch.org](https://pytorch.org/get-started/locally/) の手順で Python 環境に CUDA 版 PyTorch を入れ、同じ環境に
  `pip install "sonicproof[model]"` してください（`uv tool install` や `uvx` は別の環境を作るため、CPU 版が入ります）。
- **技術チェックだけなら：** `qc` はモデル不要です。`pip install sonicproof`（`[model]` なし）で足ります。
- **最新の `main` を使う：** `git clone https://github.com/kanaworksai-hue/sonicproof` し、その中で `uv sync --extra model` を実行します。
  コマンドは `uv run sp …`（別のフォルダからは `uv run --project /path/to/sonicproof sp …`）で実行します。

## 使い方

```bash
# プロンプト一致度。プロンプトを省くと Stable Audio の .json サイドカーから読みます。
sp check explosion.wav "破片が飛び散る巨大な爆発"
sp check thunder.wav

# ベストテイクを選ぶ
sp rank "木の扉がきしみながら開く" door_v1.wav door_v2.wav door_v3.wav door_v4.wav

# 技術チェック（モデル不要）。1 つでも不合格なら終了コード 1。スクリプトで便利です。
sp qc *.wav
sp qc ambience_loop.wav --loop --target-lufs -16

# すべてを 1 枚のスコアカードに：SHIP / REVIEW / REDO。--strict を付けると SHIP 以外は終了コード 1。
sp report footsteps.wav "砂利道を歩く足音"
sp report footsteps.wav "砂利道を歩く足音" --strict

# この音は映像に合う？シーンの説明は任意です。
sp fit sizzle.wav --image kitchen.png
sp fit rumble.wav --video storm.mp4 "雷が鳴る嵐の夜"
sp fit trailer.mp4                          # 動画自身の音声トラックを、その映像と照らし合わせる
```

### 手元の音源ライブラリ

```bash
sp index ~/SFX                              # 最初に一度。以降は新規・変更ファイルだけ読みます
sp search "トタン屋根に当たる雨" --in ~/SFX
sp search --like take_03.wav --in ~/SFX     # この音に似た音
sp search --image frame.png --in ~/SFX      # この画像に合う音
sp dedupe ~/SFX                             # 重複・ほぼ重複（何も削除しません）
```

ライブラリのフォルダは必ず明示します：`index DIR`、`search … --in DIR`、`dedupe DIR`（ホームフォルダ、ドライブのルート、システムフォルダは
拒否されます。音源が入っているフォルダを指定してください）。インデックスは `~/SFX/.sonicproof/` に保存されます。
`--in-cache` を付けると Sonicproof のキャッシュに置き、フォルダには何も書き込みません（読み取り専用フォルダでは自動的にキャッシュを使います）。
`search` と `dedupe` はその場でインデックスを更新するので、`sp index` は省略しても構いません。

### スクリプトから：`--json` と終了コード

分析系のコマンド（`check`、`rank`、`qc`、`report`、`fit`、`index`、`search`、`dedupe`）は `--json` に対応しています
（`doctor`、`serve`、`mcp`、`skill` は非対応）。`qc --json` はファイルが 1 つでも常に配列（1 ファイル 1 レコード）を出力します。
`qc` と `rank` では、読めないファイルは `"error"` になり、残りのファイルはそのまま処理されます。このときの終了コードは 2 です。

| 終了コード | 意味 |
|---|---|
| `0` | 完了。`check` / `rank` / `fit` は判定に関係なく 0 を返します。判定結果を確認してください。 |
| `1` | `qc` に不合格があった、`report --strict` の結論が SHIP ではなかった、または `doctor` のチェックが失敗した。 |
| `2` | 入力の問題：ファイルがない・読めない、音声が空、オプションの値が不正。 |
| `3` | 環境の問題：`model` extra（`sonicproof[model]`）が未インストール、モデル未ダウンロードかつオフライン、ffmpeg が必要、ポート使用中。 |

`--json` を付けるとエラーも JSON（stdout に `{"error": {"code": "not_found", "message": "…", "file": "…"}}`）で出力されるので、
スクリプトでトレースバックを解析する必要はありません。ただしコマンドラインの使い方の誤り（引数の不足、範囲外のオプション値）は例外で、
通常の使い方メッセージを表示して終了コード 2 で終わります。完全なトレースバックは `SONICPROOF_DEBUG=1` で表示できます。

## ワークベンチ

```bash
sonicproof serve                            # http://localhost:7788 を開きます
```

- **インボックス**：音声ファイルやフォルダごとドロップ。プロンプトは Stable Audio の `.json` サイドカーから読みます（自分で入力も可）。
  同じプロンプトのテイクはまとめて順位付けされ、いちばん良いものに **これを使う** が付きます。どれも合っていなければそう伝え、
  次に試す書き方を提案します。各カードには、スペクトログラムに重ねた波形、ひとことの判定（◎ そっくり · ○ 近い ·
  △ 少し近い · ✕ 違う）、0〜100 の一致度バー、技術チェックごとのドットが並び、技術的な指摘はやさしい言葉と直し方で示します。
- **ライブラリ**：フォルダを選んでインデックスを作り（進捗表示あり）、言葉・音のドロップ・画像で検索。**重複** タブにはコピーと
  残すべきファイルが並びます（何も削除しません）。ここで作るインデックスはキャッシュに置くので、フォルダには何も書き込みません。
- **映像**：画像や動画をドロップ：音の言葉で何が映っているかを見る、インボックスの音を採点する、ライブラリから合う音を探す、
  動画自身の音声トラックを判定する。

判定の裏にある生の数値は **詳細** を開けば見られます。**設定 → プロモード** にすると常に表示されます。
設定ではキャッシュの使用量の確認と消去もできます。インボックスから外したファイルは、ドロップ時に作ったコピーだけが削除されます
（元のファイルには一切触れません）。

画面表示は英語と日本語に対応しています。ブラウザの言語に合わせて表示され、切り替えもできます。音が鳴るのは再生ボタンを押したときだけです。
サーバーは `localhost` だけで待ち受け、他のサイトからのリクエストは拒否します。

![英語表示のワークベンチ：テイクをプロンプトごとにまとめ、各グループのベストに「Use this one」](https://raw.githubusercontent.com/kanaworksai-hue/sonicproof/main/docs/images/workbench.jpg)

## AI エージェントから使う

いちばん手早いのは、このリポジトリのリンクをエージェントに渡して「生成した音をチェックして」と頼むことです。
必要なことはこの節にすべて書いてあります。

**1. ターミナルでインストールし、モデルを一度ダウンロード**（[インストール](#インストール)参照）。MCP サーバーはツール呼び出しの中で
1.5 GB のダウンロードを始めません（フリーズしたように見えるため）。`sonicproof doctor --load` を一度実行しておいてください。
恒久的にインストールせずに済ませるなら：

```bash
uvx --from "sonicproof[model]" sonicproof doctor --load
```

**2. MCP サーバーを追加**：ツールは `check`、`rank`、`qc`、`report`、`fit`、`search`、`dedupe`、`index`。
どれも対応する CLI コマンドの `--json` 出力と同じレコードを返します。モデルは一度ロードしたら常駐します。

```bash
# Claude Code
claude mcp add sonicproof -e HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" -- uvx --from "sonicproof[model]" sonicproof mcp
# Codex
codex mcp add sonicproof --env HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" -- uvx --from "sonicproof[model]" sonicproof mcp
# どのクライアントでも：streamable HTTP（http://127.0.0.1:7789/mcp）
sonicproof mcp --http --port 7789
```

ほかのクライアントでは、JSON 設定に同じコマンドを書きます（キャッシュのパスは絶対パスで）：

```json
{ "mcpServers": { "sonicproof": {
    "command": "uvx", "args": ["--from", "sonicproof[model]", "sonicproof", "mcp"],
    "env": { "HF_HOME": "/Users/you/.cache/huggingface" } } } }
```

- **`HF_HOME` を渡す理由：** MCP クライアントは最小限の環境変数でサーバーを起動します。渡しておけばターミナルと同じ
  Hugging Face キャッシュを使い、モデルを二重にダウンロードしません。
- **ツール呼び出しでは絶対パスを。** 相対パスはサーバーの作業ディレクトリを基準に解決されますが、それは通常あなたの作業ディレクトリとは異なります。
- **HTTP モード**は `127.0.0.1` だけで待ち受け、localhost 以外から配信されたページからのリクエストは拒否しますが、認証はありません。
  同じマシン上のどのプロセスからも呼べます。気になる場合は stdio を使ってください。
- **`main` のクローンから動かす：** サーバーのコマンドは `uv run --project /absolute/path/to/sonicproof --extra model sonicproof mcp`
  （`--project` は呼び出し側の作業ディレクトリを保ちます。`--directory` だと変わってしまいます）。

**3. エージェントスキルを追加**　エージェントに次の流れを教え、生成音声を信頼できるものにします：「テイクを複数生成 → 順位付け →
ベストをチェック → どれも合わなければプロンプトを言い換えるかシードを変える → 選んだテイクをスコアカード付きで渡す」。
パッケージに同梱されています：

```bash
sp skill install              # → ~/.claude/skills/sonicproof/SKILL.md
sp skill install --project    # → ./.claude/skills/sonicproof/SKILL.md（このプロジェクトだけ）
sp skill show                 # 内容を表示（別の場所からスキルを読むエージェント向け）
```

クローンでは [`src/sonicproof/skill/SKILL.md`](https://github.com/kanaworksai-hue/sonicproof/blob/main/src/sonicproof/skill/SKILL.md) にあります。
[`examples/stable_audio_loop.py`](https://github.com/kanaworksai-hue/sonicproof/blob/main/examples/stable_audio_loop.py)
は同じ流れをスクリプトで回す例です（Stable Audio サーバー用と Sonicproof 用、2 つの MCP クライアントを使います）。

**このリポジトリを読んでいるエージェントへ：** 手順 1〜3 を済ませたら、スキル（`sp skill show` で表示）に従ってください。
MCP がなくても、分析コマンドはすべてシェルから `--json` 付きで使えます。例：`sp rank "prompt" /abs/a.wav /abs/b.wav --json`。
ローカルファイルの絶対パスを渡してください。音を再生して「聴こう」としないでください。Sonicproof があなたの耳です。

## スコアの読み方

すべてのクリップに 4 段階のどれかの判定が付きます。ワークベンチではやさしい言葉と 0〜100 の一致度バーで、
CLI と `--json` では判定名とその元になる生スコアで表示します。

| ワークベンチ | 判定 | 生スコア | 意味 |
|---|---|---|---|
| ◎ そっくり | **strong** | ≥ 2.0 | はっきりプロンプトどおりに聞こえる |
| ○ 近い | **plausible** | ≥ 1.1 | プロンプトどおりに聞こえる |
| △ 少し近い | **weak** | ≥ 0.3 | 少し似ている程度 |
| ✕ 違う | **miss** | < 0.3 | プロンプトどおりには聞こえない |

生スコアの出どころ：プロンプトとクリップの単純な類似度にはあまり情報がありません（どんな説明文も、どんなクリップに対しても
だいたい同じ値になります）。そこで Sonicproof は 2 つの相対的な問いを立て、その平均を取ります：

1. **クリップ側**：約 440 の身近な音の説明と比べて、「あなたのプロンプト」は「このクリップ」をどれだけうまく言い表しているか？
2. **プロンプト側**：110 本の参照音と比べて、「このクリップ」はどれだけ「あなたのプロンプト」に近いか？

どちらも基準から標準偏差いくつ分上かで測るため、CLI では `+3.79σ` のように表示されます。ワークベンチの一致度バーは
−1 … +3 を 0 … 100 に対応させたものです。

**聞こえ方（Heard as）** には、そのクリップが何に最も近く聞こえるかが並びます。✓ はプロンプトと同じ意味の説明です。
判定が意外なときは、たいていこのリストで理由がわかります。無音のクリップ（−60 dBFS を超える部分がない）は決して一致と判定されず、
`"verdict": "miss"` と `"note": "silent"` が付いて最下位になります。UI のクリック音のような短いワンショットは通常どおり判定します。

Sonicproof はあくまでセカンドオピニオンで、審判ではありません。短く具体的なプロンプト（「ガラスが割れる」）のほうが、
ミックスの指示を詰め込んだ長いプロンプトより正確に判定でき、ごく短いトランジェント音はいちばん区別が難しい音です。
採用前に必ず耳で確認してください。

### fit

`fit` は 2 つの根拠を組み合わせて判定します。どちらも同じ方法で校正され、同じ判定基準で読めます：

- **直接**：映像は、参照音よりも「このクリップ」にどれだけ近いか？
- **説明経由**：まず映像を言葉で説明し（**映っているもの / Pictured as**）、その説明をプロンプトとみなしてクリップをチェックします。
  シーンについての自分の説明を加えると、それも計算に入ります。

**映っているもの（Pictured as）** の ✓ はクリップの音にもあるもの、**聞こえ方（Heard as）** の ✓ は映像にも映っているものです。
判定が weak か miss のときは、映像から作ったプロンプトを提案します。再生成に便利です。
動画は 1 秒に 1 フレーム取り出します（最大 32 フレーム。長い動画では均等に間引きます）。

### dedupe

意味が近い 2 つの音が同じファイルだとは限りません（同じプロンプトを 4 つのシードで生成した音など）。そこで `dedupe` は埋め込みを
候補探しにだけ使い、信号そのもの（時間を合わせたスペクトル指紋）を比べます。これにより、形式・コーデック・音量・サンプルレート・
チャンネル数・頭尾カットが違うだけのコピーはまとめ、別テイクはまとめません。各グループでは残すべきコピーを 1 つ選んで提案し、
理由も示します。優先順位は、非可逆よりロスレス、クリッピングがない、高域まで実際に音が入っている（サンプルレートが高くても、
アップサンプルしただけのコピーは優先しません）、チャンネル数が多い、長い、音量を下げていない、の順です。

## Python

```python
from sonicproof.judge import Judge

judge = Judge(vision=True)    # 1 つのモデルですべて。Judge() だけなら軽いが画像は扱えない
result = judge.check("door.wav", "きしむ木の扉")
print(result.verdict, result.score, [a.text for a in result.heard_as])

fit = judge.fit("sizzle.wav", "kitchen.png", "image")
print(fit.verdict, [a.text for a in fit.pictured_as])

from sonicproof.library import Library, find_duplicates, search_text

lib = Library("~/SFX")
lib.update(lambda wave: judge.embedder.embed_audio([wave])[0])
for hit in search_text(lib, judge, "トタン屋根に当たる雨"):
    print(hit.path, hit.score, hit.verdict)
for group in find_duplicates(lib):
    print(group.keep, group.reason)
```

## 開発

```bash
uv sync --extra model --group dev && uv run pytest
cd web && npm ci && npm run dev             # ホットリロード付きの UI（:5173）。/api は :7788 の `sp serve` にプロキシ
npm run build                               #（web/ で）ビルドした UI を src/sonicproof/web に書き出す
```

UI の文言は `web/src/i18n/`（`en.ts`、`ja.ts`）にあり、型チェックで両者のキーが揃っていることを保証しています。
テスト、lint、プルリクエストについては [CONTRIBUTING.md](https://github.com/kanaworksai-hue/sonicproof/blob/main/CONTRIBUTING.md)（英語）をご覧ください。

## ライセンス

Sonicproof は [MIT ライセンス](https://github.com/kanaworksai-hue/sonicproof/blob/main/LICENSE)で公開しています（[NOTICE](https://github.com/kanaworksai-hue/sonicproof/blob/main/NOTICE) も参照）。
初回実行時に Hugging Face から EmbeddingGemma 2 をダウンロードします（モデルは同梱していません）。このモデルは Google が
Apache 2.0 で公開しており、モデルカードでは [Gemma 使用禁止ポリシー（Gemma Prohibited Use Policy）](https://ai.google.dev/gemma/prohibited_use_policy)に従った利用を求めています。

KanaWorks_AI の名称・ロゴ・キャラクター「Kana」は © KanaWorks_AI であり、MIT ライセンスの対象**外**です。
詳しくは [docs/brand](https://github.com/kanaworksai-hue/sonicproof/blob/main/docs/brand/README.md) をご覧ください。

## 気に入ったら Star を

<div align="center">

<a href="https://x.com/KanaWorks_AI"><img src="https://raw.githubusercontent.com/kanaworksai-hue/sonicproof/main/docs/brand/kana-wave.png" width="150" alt="手を振る KanaWorks_AI のマスコット、Kana"></a>

<p><b>Made with ♥ by <a href="https://x.com/KanaWorks_AI">KanaWorks_AI</a></b></p>

<p><sub>Sonicproof で試聴の手間が一回でも減ったら、⭐ をお願いします。ほかの人が見つけやすくなります。その他のプロジェクト：<a href="https://github.com/kanaworksai-hue">github.com/kanaworksai-hue</a></sub></p>

</div>
