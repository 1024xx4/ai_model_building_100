# ai_model_building_100

Python 実践 AI Model 構築 100本 Knock

---

#### Support page

https://www.shuwasystem.co.jp/book/9784798064406.html

#### Sample Data
https://www.shuwasystem.co.jp/support/7980html/6440.html


#### Versions

- Python 3.7.11
- pandas 1.1.5
- matplotlib 3.2.2
- seaborn 0.11.1
- numpy 1.19.5
- sklearn 0.22.2.post1
- scipy 1.4.1
- xgboost 0.90
- mlxtend 0.14.0
- umap 0.5.1
- shap 0.39.0
- pycaret 2.3.3

---

## 概要

- 実践的な力を身に付ける Concept
- AI Model に特化した内容
- 画像や言語などは取り扱わず、Table Data での Model 構築を行なう。
    - 価格の予測
    - 癌の診断  
      等
- 様々な特徴的な Data に対して複数の Algorithm を実践し、どのような Data の時に、どのような Algorithm を選択すべきかを学ぶ
- 直感的に Model の違いを理解

## 構成

1. 教師なし学習
2. 教師あり学習
3. 機械学習発展

の３部構成

### 第１部: 教師あり学習: Clustering, 次元削除

1. Clustering の基本
2. 特徴的な Data に対しての Algorithm の違い
3. 複数の Algorithm での次元削除

### 第２部: 教師あり学習: 回帰, 分類

4. 基本的な回帰 Model の構築
5. Algorithm の違い
6. 続 Algorithm の違い
7. 複数の Algorithm の取り扱い
8. 評価手法

### 第３部: 説明可能な AI, AutoML

9. SHAP を活用した予測に対する解釈性の取り扱い
10. AutoML 自動での Model 構築

どのような Data の時に、どのような Algorithm を選択していけば良いのか技術的な引出を増やす。

---

## 教師なし学習

- 正解を与えずに Data の傾向のみから Model を構築する。
- Data 分析と同時に使用することが多い技術。
- 複数の Algorithm が存在するが、どういった場合（Data）の時に、どの Algorithm を選択するかが重要。
- 正解 Data がないため、人間による解釈が必要であったり、評価が難しい。

### Clustering

Data の傾向から **Grouping** を行なう技術。

#### Algorithm 一覧

| 名称                  | 分類  | 推奨 Sample | 事前 Cluster数 | 特徴                                          |
|---------------------|-----|-----------|-------------|---------------------------------------------|
| KMeans              | 非階層 | 10K未満     | 要           | もっとも基本的な手法                                  |
| MiniBatch KMeans    | 非階層 | 10K以上     | 要           | KMeans を一定の Size ごとに実行。Sample数が多い場合はこちらを使う。 |
| Spectral Clustering | 非階層 | 10K未満     | 要           | Data 密度で Cluster を作成するため非線形でも機能する           |
| GMM                 | 非階層 | 10K 未満    | 要           | 傾いた楕円形で Cluster を作成できる                      |
| MeanShift           | 非階層 | 10K未満     | 任意          | Cluster 数の指定が不要                             |
| VBGMM               | 非階層 | 10K未満     | 任意          | Cluster 数の指定は不要                             |
| DBSCAN              | 非階層 | -         | 不要          | Data 密度で Cluster を作成する。外れ値を判定できる            |
| HDBSCAN             | 階層  | -         | 不要          | DBSCAN を階層型に拡張                              |

### 次元削減（次元圧縮）

複数の変数に対して、なるべく情報を落とさずに圧縮する技術

## 教師あり学習

- 正解 Data を用意する必要がある。
- 正解 Data さえあれば、Data の傾向から比較的精度の高い Model を構築できる。
- 正解 Data があることから客観的な精度評価が可能。
- Model 改善時の Parameter を Rule にのっとって機械的に行なえる。

### 回帰

数字を予測するもの

#### 回帰分析

- 教師あり学習のひとつ
- 目的変数を相関のある説明変数から説明・予測する

##### 線形回帰

回帰分析の中でも、目的変数と説明変数の関係性を線形で表す手法のこと。

###### Algorithm 一覧

| 名称  | 概要                       |
|-----|--------------------------|
| 重回帰 | 複数の説明変数を用いて１つの目的変数を予測する。 |
| LASSO 回帰 | 重回帰に過学習を抑えるための仕組みを導入したもの。最小二乗法の式に正規化項（L1-Norm）を加えている。 |
| Ridge 回帰 | 重回帰の過学習を抑えるための仕組みを導入したもの。最小二乗法の式に正規化項（L2-Norm）を加えている。 |

> **L1-Norm**<br>
> 特定の説明変数の重みを 0 にすることができるため解釈が容易になるが、全ての説明変数が重要である場合は適していない。

> **L2-Norm**<br>
> 説明変数の重みを 0 に近づけることができるが、完全に 0 にはならないため解釈が難しくなる。

### 分類

Category を分類予測する。

- 二値分類: 2 Category の場合
- 多値分類: Category が複数ある場合

### 教師あり学習の Model 構築手順

#### 1. Data の理解

与えられた Data の特徴を理解する。

- Data 件数
- 欠損値の有無

などの概観把握

- Data 同士の相関の確認

などの基礎的な分析。

#### 2. Data の加工

- 欠損値の補完
- Data-set を訓練用・Test 用に分割

など後続 Model の構築や評価に使える Data への加工。

#### 3. Model の構築

Data の特徴を学習させた Model を構築する。
学習する Algorithm により、Data の捉え方が変わる。

#### 4. Model の評価

- 客観的な精度評価指標
- 可視化 Graph

を用いて、構築した Model の精度を評価する。

#### 5. Model の保存

構築した Model をいつでも使用できるように保存する。

#### 6. Model の活用

保存した Model を用いて予測する仕組みを作っていく。