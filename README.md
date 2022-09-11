# ai_model_building_100

Python 実践 AI Model 構築 100本 Knock

---

#### Support page

https://www.shuwasystem.co.jp

#### Sample Data

https://www.shuwasystem.co.jp/book/9784798064406.html

#### Versions

- Python 3.7.11
- pandas 1.1.5
- matplotlib 3.2.2
- seaborn 0.11.1
- numpy 1.19.5
- sklearn 0.22.2 post1
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

### 次元削減（次元圧縮）
複数の変数に対して、なるべく情報を落とさずに圧縮する技術
 

