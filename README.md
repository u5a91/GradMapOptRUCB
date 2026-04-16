# Gradient Mapping Optimization

ユーザの二択のフィードバックをもとに, ベイズ最適化を用いてグラデーションマップを逐次的に最適化していく Web アプリです. BoTorch, FastAPI, Pillow, Numpy などを用いています.


## Overview
このアプリでは, アップロードされた画像をまずグレースケール化し, その明度に応じて HLS パレットから色を割り当てます.
処理の流れは次の通りです.

1. グラデーションマップを設定したい画像をアップロードする
2. サーバ側でグレースケール化する
3. 現在のグラデーションマップ `A` および相手の候補 `B` が表示される
4. ユーザが `A`, `B` のうち好ましいほうを選択する
5. 選択されたほうが次のグラデーションマップ `A` となる
6. 3--5 が `n_iter` 回繰り返される


## Features
- HLS 空間での線形補間, 明度昇順ソート
- Pairwise GP によるフィードバックの反映, Expected Utility of Best Option (EUBO) による候補の提案  
- H 成分の周期性を表す PeriodicKernel と、L/S 成分に対する RBFKernel を組み合わせたカスタムカーネル

## Requirements
- Python 3.12
- `torch` >=2.11.0,<3.0.0
- `botorch` >=0.17.2,<0.18.0
- `fastapi` >=0.135.3,<0.136.0
- `uvicorn` >=0.44.0,<0.45.0
- `pillow` >=12.2.0,<13.0.0
- `python-multipart` >=0.0.24,<0.0.25

## Getting Started
1. このリポジトリをクローンする.

   ```bash
   git clone https://github.com/u5a91/GradMapOpt.git
   cd GradMapOpt
   ```

2. **Docker を使う場合:**

    ```bash
    docker compose up
    ```

    **Docker を使わず, poetry を使う場合:**

    ```bash
    poetry install --no-root
    poetry run uvicorn main:app --host 0.0.0.0 --port 8000
    ```

    **Docker を使わず, poetry も使わない場合:**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```


## Resources
- BoTorch tutorial on preference-based Bayesian optimization  
  https://botorch.org/docs/tutorials/preference_bo/
- GPyTorch kernel documentation  
  https://docs.gpytorch.ai/en/stable/kernels.html