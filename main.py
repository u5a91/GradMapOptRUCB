from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
from dataclasses import dataclass, field


ROOT_PATH = os.environ.get("ROOT_PATH", "")

app = FastAPI(
    title="Gradient Mapping Optimization RUCB",
    root_path=ROOT_PATH, 
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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
    current_palette: np.ndarray | None = None
    challenger_palette: np.ndarray | None = None
    plays: dict = field(default_factory=dict)
    wins: dict = field(default_factory=dict)
    curr_iteration: int = 1

sessions: dict[str, SessionState] = {}

def rgb_array_to_base64(arr: np.ndarray) -> str:
    img = Image.fromarray((arr * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def sort_by_lightness(hls_pts: np.ndarray) -> np.ndarray:
    return hls_pts[np.argsort(hls_pts[:,1])]

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

def palette_id(hls_pts: np.ndarray) -> tuple:
    return tuple(map(tuple, sort_by_lightness(hls_pts)))

def record_duel(
    plays: dict, 
    wins: dict, 
    i: tuple,
    j: tuple,
    winner: tuple
):  
    key = (i, j) if winner == i else (j, i)
    wins.setdefault(key, 0)
    plays.setdefault((i, j), 0)
    plays.setdefault((j, i), 0)

    wins[key] += 1
    plays[(i, j)] += 1
    plays[(j, i)] += 1

def rucb_score(
    plays: dict, 
    wins: dict, 
    i: tuple,
    j: tuple,
    t: int,
    alpha: float = 1.0
) -> float:
    nij = plays.get((i, j), 0)
    if nij == 0:
        return 1.0

    pij = wins.get((i, j), 0) / nij
    return pij + np.sqrt(alpha * np.log(t) / nij)

def select_challenger_local(
    curr_hls: np.ndarray,
    plays: dict,
    wins: dict,
    t: int,
    n_iter: int,
    M: int = 50,
    sigma0: float = 0.3,
) -> np.ndarray:
    k = curr_hls.shape[0]
    t_eff = min(t, n_iter)
    sigma_t = sigma0 * (1 - (t_eff - 1) / n_iter)

    cands = curr_hls[None] + np.random.normal(scale=sigma_t, size=(M, k, 3))
    cands = np.clip(cands, 0.0, 1.0)

    best, best_score = None, -np.inf
    curr_id = palette_id(curr_hls)
    for h in cands:
        h_id = palette_id(h)
        score = rucb_score(plays, wins, curr_id, h_id, t_eff)
        if score > best_score:
            best_score, best = score, h

    if best is None:
        raise HTTPException(status_code=500, detail="Failed to generate challenger palette")
    
    return best

def get_session_or_404(session_id: str) -> SessionState:
    state = sessions.get(session_id)
    if state is None:
        raise HTTPException(
            status_code=404,
            detail="Invalid session_id. Please upload an image first.",
        )
    return state

@app.post("/upload/")
async def upload_image(
    file: UploadFile = File(...),
    n_iter_form: int = Form(20),
    k_form: int = Form(5),
):
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
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

    arr = np.asarray(img) / 255.0
    gray = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])

    session_id = str(uuid.uuid4())
    sessions[session_id] = SessionState(
        gray_map=gray,
        n_iter=n_iter_form,
        k=k_form,
    )

    return {
        "message": f"Upload OK. n_iter={n_iter_form}, K={k_form}",
        "shape": gray.shape,
        "session_id": session_id,
    }


@app.get("/palette/next/")
def get_next_pair(session_id: str):
    state = get_session_or_404(session_id)

    if state.current_palette is None:
        state.current_palette = np.random.rand(state.k, 3)
        state.challenger_palette = select_challenger_local(
            state.current_palette,
            state.plays,
            state.wins,
            state.curr_iteration,
            state.n_iter,
        )

    if state.challenger_palette is None:
        raise HTTPException(status_code=500, detail="Challenger palette is not initialized")

    imgA = apply_gradient_mixed(state.gray_map, state.current_palette)
    imgB = apply_gradient_mixed(state.gray_map, state.challenger_palette)

    barA = make_grad_bar_mixed(state.current_palette)
    barB = make_grad_bar_mixed(state.challenger_palette)
    palA = sort_by_lightness(state.current_palette)
    palB = sort_by_lightness(state.challenger_palette)

    return {
        "iteration": state.curr_iteration,
        "total_iterations": state.n_iter,
        "A": rgb_array_to_base64(imgA),
        "B": rgb_array_to_base64(imgB),
        "barA": rgb_array_to_base64(barA),
        "barB": rgb_array_to_base64(barB),
        "palA": palA.tolist(),
        "palB": palB.tolist(),
    }

@app.post("/palette/choice/")
def record_choice(
    session_id: str = Form(...),
    choice: str = Form(...),
):
    state = get_session_or_404(session_id)

    if choice not in ["A", "B"]:
        raise HTTPException(
            status_code=400,
            detail="choice must be either 'A' or 'B'",
        )

    if state.current_palette is None or state.challenger_palette is None:
        raise HTTPException(
            status_code=400,
            detail="Please call /palette/next/ before submitting a choice",
        )

    if choice == "A":
        winner = state.current_palette
        loser = state.challenger_palette
    else:
        winner = state.challenger_palette
        loser = state.current_palette

    winner_id = palette_id(winner)
    loser_id = palette_id(loser)

    record_duel(
        state.plays,
        state.wins,
        winner_id,
        loser_id,
        winner=winner_id,
    )

    state.curr_iteration += 1
    state.current_palette = winner
    if state.curr_iteration <= state.n_iter:
        state.challenger_palette = select_challenger_local(
            state.current_palette,
            state.plays,
            state.wins,
            state.curr_iteration,
            state.n_iter,
        )
    else:
        state.challenger_palette = None

    return {"next_iteration": state.curr_iteration}

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host=host, port=port)


'''
@app.get("/items/{item_id}")
def read_item(
    item_id: int,
    q: str | None = Query(None),
):
    if item_id < 0:
        raise HTTPException(400, "Invalid item_id")
    return {"item_id": item_id, "q": q}

@app.post("/upload/")
async def upload_file(
    file: UploadFile = File(...),
    count: int = Form(1),
):
    data = await file.read()
    return {"filename": file.filename, "size": len(data), "count": count}

'''