from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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

SESSION_TTL_SECONDS = 30 * 60  # 30 minutes

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


@app.get("/", include_in_schema=False)  # docs に表示しない
async def root():
    return FileResponse("static/index.html")


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
    arr = x.detach().cpu().numpy().reshape(k, 3)
    return np.clip(arr, 0.0, 1.0)


def cleanup_expired_sessions() -> None:
    now = time.time()
    expired_ids = [
        session_id
        for session_id, state in sessions.items()
        if now - state.last_accessed > SESSION_TTL_SECONDS
    ]
    for session_id in expired_ids:
        del sessions[session_id]


def rgb_array_to_base64(arr: np.ndarray) -> str:
    img = Image.fromarray((arr * 255).astype(np.uint8))
    with io.BytesIO() as buf:
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def sort_by_lightness(hls_pts: np.ndarray) -> np.ndarray:
    return hls_pts[np.argsort(hls_pts[:, 1])]


def hls2rgb_array(hls_pts: np.ndarray) -> np.ndarray:
    return np.array([colorsys.hls_to_rgb(h, l, s) for h, l, s in hls_pts])


def apply_gradient_mixed(gray: np.ndarray, hls_pts: np.ndarray) -> np.ndarray:
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
    k = len(hls_pts)
    hls_sorted = sort_by_lightness(hls_pts)
    rgb = hls2rgb_array(hls_sorted)
    xs = np.linspace(0, 1, width)
    seg = np.minimum((xs * (k - 1)).astype(int), k - 2)
    t = xs * (k - 1) - seg
    c0 = rgb[seg]
    c1 = rgb[seg + 1]
    bar = (1 - t)[:, None] * c0 + t[:, None] * c1
    return np.tile(bar[None, :, :], (height, 1, 1))


def get_session_or_404(session_id: str) -> SessionState:
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
        q=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )
    return X_next.squeeze(0)  # shape: (d,)


def fit_pref_model(X: torch.Tensor, comps: torch.Tensor) -> PairwiseGP:
    model = PairwiseGP(
        X,
        comps,
        input_transform=Normalize(d=X.shape[-1]),
    )
    mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def canonicalize_x(x: torch.Tensor, k: int) -> torch.Tensor:
    pal = x.detach().cpu().numpy().reshape(k, 3)
    pal = sort_by_lightness(np.clip(pal, 0.0, 1.0))
    return torch.tensor(pal.reshape(-1), dtype=torch.double, device=x.device)


@app.post("/upload/")
async def upload_image(
    file: UploadFile = File(...),
    n_iter_form: int = Form(20),
    k_form: int = Form(5),
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
    gray = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])

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
        model = fit_pref_model(state.X, state.comps)

        challenger = propose_challenger_eubo(
            model=model,
            incumbent=state.incumbent,
            dim=state.X.shape[-1],
        )
        challenger = canonicalize_x(challenger, state.k)

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
