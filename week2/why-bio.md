# Week 2 — BIO Tagging for Compound Segmentation

## 🎯 このタスクの目的

**Week 1**: 「単語が複合語かどうか」の判定 — Classification
**Week 2**: 「その複合語を**どこで分割するか**」の予測 — Sequence Labeling

Week 1 のモデルは `Kinderfahrradhelm` を見て「複合語である」と言えた。
Week 2 のモデルは同じ単語を見て `Kinder | fahrrad | helm` と切れるようにする。

技術的なゴール: **BIO (Begin / Inside / Outside) tagging framework** で、文字ごとにラベルを予測する sequence labeling を実装する。

---

## 📊 Week 1 vs Week 2 の位置付け

| | Week 1 | Week 2 |
|---|---|---|
| タスク | 判定 | 分割 |
| 入力 | 単語 | 単語 |
| 出力 | 0 or 1 | BIO の列 |
| 予測単位 | 単語 1つに 1 判定 | 文字 1つに 1 判定 |
| フレームワーク | Binary Classification | Sequence Labeling |
| モデル | Logistic Regression | LogReg / CRF / BiLSTM |
| 特徴的な難しさ | 独立に判定できる | 前後の文字の依存を考慮する必要 |

**共通点**: どちらも n-gram feature engineering を使う (Week 1 のコードが土台)

---

## 🔤 BIO Tagging とは

文字列の各位置に3つのタグのどれかを付ける記法:

- **B** = Beginning — 形態素の始まり
- **I** = Inside — 形態素の途中
- **O** = Outside — どの形態素にも属さない (今回はほぼ使わない)

### 例 1: `Kinderfahrradhelm` の BIO タグ

```
Position: 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16
Char:     K  i  n  d  e  r  f  a  h  r  r  a  d  h  e  l  m
Tag:      B  I  I  I  I  I  B  I  I  I  I  I  I  B  I  I  I
          └─── Kinder ────┘└──── fahrrad ────┘└─── helm ───┘
```

**B の位置 = 分割すべき位置**。モデルが BIO を予測できれば、`B` の直前にスペースを入れて分割できる。

### 例 2: `computerlinguistik`

```
Char:  c  o  m  p  u  t  e  r  l  i  n  g  u  i  s  t  i  k
Tag:   B  I  I  I  I  I  I  I  B  I  I  I  I  I  I  I  I  I
       └── computer ──┘└──── linguistik ──────┘
```

### 変換の全体フロー

```
Input word:  Kinderfahrradhelm
      ↓
文字ごとに feature 抽出 (周辺の n-gram)
      ↓
文字ごとの BIO 予測: [B, I, I, I, I, I, B, I, I, I, I, I, I, B, I, I, I]
      ↓
B の位置 (0, 6, 13) を境界とする
      ↓
Output:  Kinder | fahrrad | helm
```

---

## 🌍 実務ではどこで使われているか

BIO tagging は「情報抽出タスク全般で使われる universal framework」で、academic な記法ではなく **2026 年の今も production system の中枢で動いている**。

### 1. E コマース検索 (Zalando, Amazon.de, Otto)

商品タイトル:
```
Nike Air Max 90 Sneaker Herren Größe 42 schwarz
```

これを機能ごとに tagging (BIO + entity type):
```
Nike  Air   Max   90    Sneaker  Herren  Größe   42   schwarz
B-BR  I-BR  I-BR  B-MD  B-TYPE   B-GEN   B-SZK   B-SZ B-COL
```
(BR = Brand, MD = Model, TYPE, GEN = Gender, SZK = SizeKey, SZ = Size, COL = Color)

これができれば、ユーザーの検索クエリ「size 42 の Nike スニーカー、黒」に対して、structured なマッチングが可能になる。**BIO tagging がその実装基盤**。

### 2. Named Entity Recognition (NER)

ある文章から人名 / 組織名 / 場所を抜き出すタスク:

```
Text:  Ich   habe  bei   DeepL   in    Köln    gearbeitet
Tag:   O     O     O     B-ORG   O     B-LOC   O
```

これは BIO tagging そのもの (`B-` の後に entity type を付ける拡張形)。

**実際のユースケース**:
- LinkedIn 投稿から会社名 / スキル名を抽出 → レコメンド
- 履歴書から skill / 学歴を自動抽出
- ニュース記事から event / 人物 / 場所を抽出

### 3. RAG での Chunking と Entity-aware Retrieval

RAG (Retrieval-Augmented Generation) の pipeline:

```
[Document] → [Chunking] → [Embedding] → [Vector DB]
                                              ↓
[User Query] → [Entity Extract] → [Retrieval] → [LLM] → [Answer]
```

BIO tagging がここで登場する場所:

- **Chunking**: 単純な文字数 chunk ではなく、意味単位で切る → 境界予測は BIO 系の技術
- **Entity Extract**: Query から人名 / 会社名 / 商品名を抜き出す → NER = BIO
- **Entity-aware Retrieval**: Document 側でも entity を抽出しておいて、entity レベルでもマッチング

「Retrieval の品質は tokenization と entity extraction の品質で決まる」— BIO はこの pipeline の複数箇所で効いてくる。

### 4. LLM の Tokenization (BPE, WordPiece)

現代の LLM は BPE (Byte Pair Encoding) や WordPiece で単語を subword に分解する。

| | BIO Tagging | BPE / WordPiece |
|---|---|---|
| 方式 | Supervised (ラベルから学習) | Unsupervised (頻度で greedy merge) |
| 出力 | 意味的な形態素 | 統計的な subword |
| 学習に必要 | アノテーション | 生テキスト |
| 使われる場面 | NER, segmentation, chunking | LLM の tokenizer |

**技術的には別物だが、「文字列を意味のある単位に切る」という同じ問題領域**。両方を理解していると、「LLM が未知の複合語をどう扱っているか」を implementation レベルで語れる。

### 5. 日本語の形態素解析 (MeCab, Sudachi)

日本語には space がないので、あらゆる NLP タスクの前段で morphological analyzer が動く:

```
入力: 東京大学の学生
出力: 東京 | 大学 | の | 学生
```

内部では CRF や neural sequence model が使われているが、**根本のタスク設計は BIO tagging と同じ**。

**ここが USP になる**: ドイツ語で BIO を実装した経験は、日本語 morphological analyzer の内部動作の理解に直結する。「単語境界がない言語」を 2つ扱った経験は多言語 NLP の会社で強く効く。

---

## 🛠 モデルの選択肢 (階層)

### Level A: 各文字を独立に分類
- 各文字の周辺 n-gram を feature にして B/I を分類
- **Logistic Regression** で実装可能
- Week 1 の技術がほぼそのまま使える
- 前後の予測の依存関係を無視 → 精度に限界

### Level B: CRF (Conditional Random Field)
- 「前の文字が B なら次は I になりやすい」等の遷移を学習
- `sklearn-crfsuite` で実装可能
- 独立分類より精度が上がる典型例

### Level C: Neural Sequence Model (BiLSTM, Transformer)
- 文字 embedding + LSTM / Transformer
- SOTA 級の精度
- 実装コストが上がる

**方針**: Level A から入り、時間があれば Level B に進む。Level C は余裕があれば Week 4 で試す。

---

## 🔗 Week 1 との繋がり

- **N-gram feature extraction**: Week 1 のコードがそのまま使える (各文字の周辺 n-gram を feature 化)
- **Logistic Regression**: Level A の baseline に流用
- **fit / transform, sparse matrix**: Week 1 で身に付けた handling がそのまま活きる
- **Language-agnostic な approach**: Week 1 で確認した「言語知識に頼らない設計」の哲学が Week 2 でも一貫

---

## 🎯 面接での mental model

このプロジェクト全体を通じて言えるようになる ideal statement:

> Week 1 では複合語の判定を binary classification で解きました。Week 2 では同じ問題を **sequence labeling** として再定義し、BIO tagging framework で文字ごとに boundary を予測するようにしました。これにより Week 1 では出力できなかった「どこで切るか」の情報を出せるようになりました。
>
> このフレームワークは NER, chunking, product attribute extraction といった information extraction 全般で使われている standard な approach です。Zalando の product search でも、RAG system の entity extraction 部分でも、私が実装したものと本質的に同じ構造が動いています。
>
> さらに、この技術は日本語の morphological analyzer (MeCab, Sudachi) の内部動作と同じ発想です。ドイツ語と日本語という 2つの「word boundary が曖昧な言語」で同じ approach が有効であることは、多言語対応の NLP システムを設計する上での実践的な ground になります。

---

## 📝 このドキュメントを読む人が受け取れるもの

- Week 1 → Week 2 で **classification から sequence labeling へ** 抽象度を引き上げている
- 学生 exercise ではなく production NLP の core を実装している
- 多言語の視点 (Japanese + German) を意識した設計
- Modern LLM (BPE, WordPiece) や RAG との理論的接続を理解している
