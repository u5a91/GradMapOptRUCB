from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import colorsys
import base64
import os
import uuid
import time
from dataclasses import dataclass, field

import torch

from botorch.fit import fit_gpytorch_mll
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.input import Normalize
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.optim import optimize_acqf
from gpytorch.kernels import ScaleKernel, RBFKernel, PeriodicKernel, AdditiveKernel
from botorch.exceptions.errors import ModelFittingError

SESSION_TTL_SECONDS = 30 * 60  # 30 minutes

DEFAULT_N_ITER = 10
DEFAULT_K = 5

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB
MAX_IMAGE_WIDTH = 4096
MAX_IMAGE_HEIGHT = 4096

ROOT_PATH = os.environ.get("ROOT_PATH", "")

app = FastAPI(
    title="Gradient Mapping Optimization",
    root_path=ROOT_PATH,
)


app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static",
)
templates = Jinja2Templates(directory="static")

@app.get("/", include_in_schema=False, response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "default_n_iter": DEFAULT_N_ITER,
            "default_k": DEFAULT_K,
        },
    )

@dataclass
class SessionState:
    gray_map: np.ndarray
    n_iter: int
    k: int

    X: torch.Tensor | None = None  # shape: (n, d)
    comps: torch.Tensor | None = None  # shape: (m, 2)

    current_pair: torch.Tensor | None = None  # shape: (2, d)
    incumbent: torch.Tensor | None = None  # shape: (d,)
    curr_iteration: int = 1
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)


sessions: dict[str, SessionState] = {}


def x_to_palette(x: torch.Tensor, k: int) -> np.ndarray:
    """
    shape (3 * k,) のテンソルをグラデーションマップの基本色 (パレット) の配列に変換する

    Args:
        x (torch.Tensor): shape (3 * k,) のテンソル
        k (int): グラデーションマップの基本色の数

    Returns:
        np.ndarray: shape (k, 3) の HLS 配列
    """
    arr = x.detach().cpu().numpy().reshape(k, 3)
    return np.clip(arr, 0.0, 1.0)


def cleanup_expired_sessions() -> None:
    """
    期限切れのセッションを削除する

    Returns:
        None: 期限切れセッションを sessions から削除
    """
    now = time.time()
    expired_ids = [
        session_id
        for session_id, state in sessions.items()
        if now - state.last_accessed > SESSION_TTL_SECONDS
    ]
    for session_id in expired_ids:
        del sessions[session_id]


def rgb_array_to_base64(arr: np.ndarray) -> str:
    """
    RGB 配列を PNG 形式の base64 エンコードし、URI として返す

    Args:
        arr (np.ndarray): 画像を表す shape (H, W, 3) の RGB 配列
    
    Returns:
        str: base 64 エンコードされた PNG 画像の URI
    """
    img = Image.fromarray((arr * 255).astype(np.uint8))
    with io.BytesIO() as buf:
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def sort_by_lightness(hls_pts: np.ndarray) -> np.ndarray:
    """
    HLS 配列 (パレット) を明度の昇順にソートする

    Args:
        hls_pts (np.ndarray): shape (k, 3) の HLS 配列

    Returns:
        np.ndarray: 明度の昇順にソートされた HLS 配列
    """
    return hls_pts[np.argsort(hls_pts[:, 1])]


def hls2rgb_array(hls_pts: np.ndarray) -> np.ndarray:
    """
    HLS 配列を RGB 配列に変換する

    Args:
        hls_pts (np.ndarray): shape (k, 3) の HLS 配列
    
    Returns:
        np.ndarray: shape (k, 3) の RGB 配列
    """
    return np.array([colorsys.hls_to_rgb(h, l, s) for h, l, s in hls_pts])


def apply_gradient_mixed(gray: np.ndarray, hls_pts: np.ndarray) -> np.ndarray:
    """
    グレースケール画像に対して線形補間に基づくグラデーションマップを適用する

    Args:
        gray (np.ndarray): shape (H, W) のグレースケール画像
        hls_pts (np.ndarray): shape (k, 3) の HLS 配列

    Returns:
        np.ndarray: shape (H, W, 3) の RGB 画像
    """
    h, w = gray.shape
    k = len(hls_pts)
    hls_sorted = sort_by_lightness(hls_pts)
    rgb = hls2rgb_array(hls_sorted)
    g = gray.ravel()
    seg = np.minimum((g * (k - 1)).astype(int), k - 2)
    t = g * (k - 1) - seg
    c0 = rgb[seg]
    c1 = rgb[seg + 1]
    out = (1 - t)[:, None] * c0 + t[:, None] * c1
    return out.reshape(h, w, 3)


def make_grad_bar_mixed(hls_pts: np.ndarray, width=300, height=20) -> np.ndarray:
    """
    パレットに対応するグラデーションバーの画像を生成する

    Args:
        hls_pts (np.ndarray): shape (k, 3) の HLS 配列
        width (int): グラデーションバーの幅
        height (int): グラデーションバーの高さ
    
    Returns:
        np.ndarray: shape (height, width, 3) の RGB 画像
    """
    k = len(hls_pts)
    hls_sorted = sort_by_lightness(hls_pts)
    rgb = hls2rgb_array(hls_sorted)

    positions = np.linspace(0, 1, width)
    x = positions * (k - 1)
    idx = np.minimum(x.astype(int), k - 2)
    dx = x - idx

    left_colors = rgb[idx]
    right_colors = rgb[idx + 1]
    bar = (1 - dx)[:, None] * left_colors + dx[:, None] * right_colors

    return np.tile(bar[None, :, :], (height, 1, 1))


def get_session_or_404(session_id: str) -> SessionState:
    """
        セッション ID からセッション状態を取得する

    Args:
        session_id (str): セッション ID
    
    Returns:
        SessionState: セッション状態
    """
    cleanup_expired_sessions()
    state = sessions.get(session_id)

    if state is None:
        raise HTTPException(
            status_code=404,
            detail="Invalid session_id. Please upload an image first.",
        )

    state.last_accessed = time.time()
    return state


def propose_challenger_eubo(
    model: PairwiseGP,
    incumbent: torch.Tensor,
    dim: int,
    num_restarts: int = 8,
    raw_samples: int = 128,
) -> torch.Tensor:
    """
    現在の最適候補 (incumbent) に対して
    Expected Utility of Best Option (EUBO) による相手候補を提案する

    Args:
        model (PairwiseGP): 選好ベース GP モデル
        incumbent (torch.Tensor): 現在の最適候補
        dim (int): 探索空間の次元
        num_restarts (int): 実際に局所探索を行う点数
        raw_samples (int): 初期サンプリング点数

    Returns:
        torch.Tensor: shape (dim,) の相手候補
    """
    bounds = torch.stack(
        [
            torch.zeros(dim, dtype=torch.double),
            torch.ones(dim, dtype=torch.double),
        ]
    )

    acq = AnalyticExpectedUtilityOfBestOption(pref_model=model)

    def acq_wrapper(x: torch.Tensor) -> torch.Tensor:
        batch_shape = x.shape[:-2]
        inc = incumbent.view(*([1] * len(batch_shape)), 1, dim).expand(
            *batch_shape, 1, dim
        )
        pair = torch.cat([inc, x], dim=-2)  # shape (..., 2, d)
        return acq(pair)

    X_next, _ = optimize_acqf(
        acq_function=acq_wrapper,
        bounds=bounds,
        q=1,    # 提案する (相手) 候補点数は 1
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )
    return X_next.squeeze(0)  # shape: (d,)


def build_hls_kernel(k: int) -> ScaleKernel:
    """
    HLS 空間上の候補点に対する共分散カーネルを構築する
    入力テンソルに対し
    H には周期性を反映する PeriodicKernel を
    L/S には標準の RBFKernel をそれぞれ適用し
    それらを AdditiveKernel で足し合わせたうえで
    ScaleKernel により全体をスケーリングする

    Args:
        k (int): パレットの色数

    Returns:
        ScaleKernel: 構築した HLS 用カーネル
    """

    hue_dims = list(range(0, 3 * k, 3))
    ls_dims = [i for i in range(3 * k) if i not in hue_dims]

    hue_kernel = PeriodicKernel(active_dims=hue_dims, ard_num_dims=len(hue_dims))
    ls_kernel = RBFKernel(active_dims=ls_dims, ard_num_dims=len(ls_dims))

    return ScaleKernel(AdditiveKernel(hue_kernel, ls_kernel))


def fit_pref_model(X: torch.Tensor, comps: torch.Tensor, k: int) -> PairwiseGP:
    """
    比較結果から選好 GP モデルをフィットする

    Args:
        X (torch.Tensor): shape (n, d) の候補点集合
        comps (torch.Tensor): shape (m, 2) の比較結果
        k (int): パレットの色数

    Returns:
        PairwiseGP: フィットされた選好 GP モデル
    """
    model = PairwiseGP(
        X,
        comps,
        covar_module=build_hls_kernel(k),
        input_transform=Normalize(d=X.shape[-1]),
    )
    mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model

def canonicalize_x(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    HLS テンソルを明度順ソートし [0, 1] にクリップして正規化

    Args:
        x (torch.Tensor): shape (3 * k,) の HLS テンソル
        k (int): パレットの色数

    Returns:
        torch.Tensor: 正規化された HLS テンソル
    """
    pal = x.detach().cpu().numpy().reshape(k, 3)
    pal = sort_by_lightness(np.clip(pal, 0.0, 1.0))
    return torch.tensor(pal.reshape(-1), dtype=torch.double, device=x.device)


@app.post("/upload/")
async def upload_image(
    file: UploadFile = File(...),
    n_iter_form: int = Form(DEFAULT_N_ITER),
    k_form: int = Form(DEFAULT_K),
):
    cleanup_expired_sessions()
    if not (1 <= n_iter_form <= 20):
        raise HTTPException(
            status_code=400,
            detail="n_iter must be between 1 and 20",
        )

    if not (2 <= k_form <= 6):
        raise HTTPException(
            status_code=400,
            detail="k must be between 2 and 6",
        )

    data = await file.read()

    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
        )

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        img = img.convert("RGB")
    except UnidentifiedImageError as e:
        raise HTTPException(
            status_code=400,
            detail="Failed to load image: unsupported or invalid image file",
        ) from e
    except OSError as e:
        raise HTTPException(
            status_code=400,
            detail="Failed to load image: corrupted image data",
        ) from e

    width, height = img.size
    if width > MAX_IMAGE_WIDTH or height > MAX_IMAGE_HEIGHT:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Image too large. Maximum resolution is "
                f"{MAX_IMAGE_WIDTH}x{MAX_IMAGE_HEIGHT}"
            ),
        )

    arr = np.asarray(img) / 255.0
    # RGB をグレースケールに変換 (ITU-R BT.601)
    gray = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])   # (H, W)

    dim = 3 * k_form    # d = 3 * k
    x0 = torch.stack(
        [canonicalize_x(x, k_form) for x in torch.rand(2, dim, dtype=torch.double)]
    )

    session_id = str(uuid.uuid4())
    sessions[session_id] = SessionState(
        gray_map=gray,
        n_iter=n_iter_form,
        k=k_form,
        X=x0.clone(),
        comps=torch.empty((0, 2), dtype=torch.long),
        current_pair=x0.clone(),
        incumbent=x0[0].clone(),
    )

    return {
        "message": f"Upload OK. n_iter={n_iter_form}, K={k_form}",
        "shape": gray.shape,
        "session_id": session_id,
    }


@app.get("/palette/next/")
def get_next_pair(session_id: str):
    state = get_session_or_404(session_id)

    if state.current_pair is None:
        return {
            "finished": True,
            "iteration": state.curr_iteration - 1,
            "total_iterations": state.n_iter,
        }

    palA = x_to_palette(state.current_pair[0], state.k)
    palB = x_to_palette(state.current_pair[1], state.k)

    imgA = apply_gradient_mixed(state.gray_map, palA)
    imgB = apply_gradient_mixed(state.gray_map, palB)

    barA = make_grad_bar_mixed(palA)
    barB = make_grad_bar_mixed(palB)

    return {
        "iteration": state.curr_iteration,
        "total_iterations": state.n_iter,
        "A": rgb_array_to_base64(imgA),
        "B": rgb_array_to_base64(imgB),
        "barA": rgb_array_to_base64(barA),
        "barB": rgb_array_to_base64(barB),
        "palA": sort_by_lightness(palA).tolist(),
        "palB": sort_by_lightness(palB).tolist(),
    }


def is_too_close(x: torch.Tensor, X: torch.Tensor, tol: float = 1e-4) -> bool:
    if X.numel() == 0:
        return False
    d2 = ((X - x) ** 2).sum(dim=1)
    return torch.any(d2 < tol ** 2).item()


def make_distinct_candidate(
    x: torch.Tensor,
    X: torch.Tensor,
    k: int,
    max_tries: int = 5,
    noise_scale: float = 1e-2,
) -> torch.Tensor:
    x = canonicalize_x(x, k)

    if not is_too_close(x, X):
        return x

    for _ in range(max_tries):
        y = x + noise_scale * torch.randn_like(x)
        y = canonicalize_x(y, k)
        if not is_too_close(y, X):
            return y

    dim = X.shape[1]
    return canonicalize_x(torch.rand(dim, dtype=torch.double, device=X.device), k)


@app.post("/palette/choice/")
def record_choice(
    session_id: str = Form(...),
    choice: str = Form(...),
):
    state = get_session_or_404(session_id)

    if choice not in ["A", "B"]:
        raise HTTPException(status_code=400, detail="choice must be either 'A' or 'B'")

    if state.X is None or state.comps is None or state.current_pair is None:
        raise HTTPException(status_code=400, detail="Session is not initialized")

    n = state.X.shape[0]
    idx_a, idx_b = n - 2, n - 1

    if choice == "A":
        new_comp = torch.tensor([[idx_a, idx_b]], dtype=torch.long)
        winner = state.current_pair[0].clone()
    else:
        new_comp = torch.tensor([[idx_b, idx_a]], dtype=torch.long)
        winner = state.current_pair[1].clone()

    state.comps = torch.cat([state.comps, new_comp], dim=0)
    state.incumbent = winner
    state.curr_iteration += 1

    if state.curr_iteration <= state.n_iter:
        try:
            model = fit_pref_model(state.X, state.comps, state.k)
            challenger = propose_challenger_eubo(
                model=model,
                incumbent=state.incumbent,
                dim=state.X.shape[-1],
            )
            challenger = make_distinct_candidate(
                challenger,
                state.X,
                state.k,
            )
        except ModelFittingError:
            challenger = make_distinct_candidate(
                torch.rand(state.X.shape[-1], dtype=torch.double, device=state.X.device),
                state.X,
                state.k,
            )

        next_pair = torch.stack([state.incumbent, challenger], dim=0)

        state.X = torch.cat([state.X, next_pair], dim=0)
        state.current_pair = next_pair
    else:
        state.current_pair = None

    return {"next_iteration": state.curr_iteration}


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host=host, port=port)
