from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
from PIL import Image
import io
import colorsys
import base64

# ============================================================================
# 1) FastAPI インスタンス
# ============================================================================
app = FastAPI(
    title="Color Gradient RUCB API",
    description="画像アップロード→パレット生成→RUCB更新を行うAPI",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory=".", html=True), name="static")

from fastapi.responses import FileResponse

# ─── ルートアクセスで index.html を返す ───
@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("index.html")

# ============================================================================
# 2) グローバル状態
# ============================================================================
gray_map: np.ndarray | None = None
current_palette: np.ndarray | None = None
challenger_palette: np.ndarray | None = None
n_iter: int = 20
K: int = 5
plays: dict = {}
wins: dict = {}
curr_iteration: int = 1

# ============================================================================
# 3) ユーティリティ関数
# ============================================================================
def rgb_array_to_base64(arr: np.ndarray) -> str:
    img = Image.fromarray((arr * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def sort_by_lightness(hsl_pts: np.ndarray) -> np.ndarray:
    return hsl_pts[np.argsort(hsl_pts[:,1])]

def hls2rgb_array(hsl_pts: np.ndarray) -> np.ndarray:
    return np.array([colorsys.hls_to_rgb(h, l, s) for h, l, s in hsl_pts])

def apply_gradient_mixed(gray: np.ndarray, hsl_pts: np.ndarray) -> np.ndarray:
    H, W = gray.shape
    k = len(hsl_pts)
    hsl_sorted = sort_by_lightness(hsl_pts)
    rgb = hls2rgb_array(hsl_sorted)
    g = gray.ravel()
    seg = np.minimum((g * (k - 1)).astype(int), k - 2)
    t = g * (k - 1) - seg
    c0 = rgb[seg]
    c1 = rgb[seg + 1]
    out = (1 - t)[:, None] * c0 + t[:, None] * c1
    return out.reshape(H, W, 3)

def make_grad_bar_mixed(hsl_pts: np.ndarray, width=300, height=20) -> np.ndarray:
    k = len(hsl_pts)
    hsl_sorted = sort_by_lightness(hsl_pts)
    rgb = hls2rgb_array(hsl_sorted)
    xs = np.linspace(0, 1, width)
    seg = np.minimum((xs * (k - 1)).astype(int), k - 2)
    t = xs * (k - 1) - seg
    c0 = rgb[seg]
    c1 = rgb[seg + 1]
    bar = (1 - t)[:, None] * c0 + t[:, None] * c1
    return np.tile(bar[None, :, :], (height, 1, 1))

def record_duel(i, j, winner):
    key = (i, j) if winner == i else (j, i)
    wins.setdefault(key, 0)
    plays.setdefault((i, j), 0)
    plays.setdefault((j, i), 0)
    wins[key] += 1
    plays[(i, j)] += 1
    plays[(j, i)] += 1

def rucb_score(i, j, t, alpha=1.0):
    nij = plays.get((i, j), 0)
    if nij == 0:
        return 1.0
    pij = wins.get((i, j), 0) / nij
    return pij + np.sqrt(alpha * np.log(t) / nij)

def select_challenger_local(curr_hsl: np.ndarray, t: int, n_iter: int, M=50, sigma0=0.3) -> np.ndarray:
    k = curr_hsl.shape[0]
    t_eff = min(t, n_iter)
    sigma_t = sigma0 * (1 - (t_eff - 1) / n_iter)
    cands = curr_hsl[None] + np.random.normal(scale=sigma_t, size=(M, k, 3))
    cands = np.clip(cands, 0.0, 1.0)
    best, best_s = None, -np.inf
    curr_id = tuple(map(tuple, sort_by_lightness(curr_hsl)))
    for h in cands:
        h_id = tuple(map(tuple, sort_by_lightness(h)))
        s = rucb_score(curr_id, h_id, t_eff)
        if s > best_s:
            best_s, best = s, h
    return best

# ============================================================================
# 4) エンドポイント定義
# ============================================================================
@app.post("/upload/")
async def upload_image(
    file: UploadFile = File(...),
    n_iter_form: int = Form(20),
    k_form: int = Form(5),
):
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except:
        raise HTTPException(400, "画像の読み込みに失敗しました")
    arr = np.asarray(img) / 255.0
    gray = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])

    global gray_map, current_palette, challenger_palette
    global curr_iteration, wins, plays, n_iter, K

    # パラメータ更新
    n_iter = n_iter_form
    K      = k_form

    # 状態リセット
    gray_map           = gray
    current_palette     = None
    challenger_palette = None
    curr_iteration      = 1
    wins                = {}
    plays               = {}

    return {"message": f"Upload OK. n_iter={n_iter}, K={K}", "shape": gray.shape}

@app.get("/palette/next/")
def get_next_pair():
    global gray_map, current_palette, challenger_palette, curr_iteration, n_iter, K

    if gray_map is None:
        raise HTTPException(400, "先に /upload/ で画像をアップロードしてください")

    if current_palette is None:
        current_palette     = np.random.rand(K, 3)
        challenger_palette = select_challenger_local(current_palette, curr_iteration, n_iter)

    imgA = apply_gradient_mixed(gray_map, current_palette)
    imgB = apply_gradient_mixed(gray_map, challenger_palette)

    # グラデーションバー & ソート済パレット
    barA = make_grad_bar_mixed(current_palette)
    barB = make_grad_bar_mixed(challenger_palette)
    palA = sort_by_lightness(current_palette)

    return {
        "iteration":        curr_iteration,
        "total_iterations": n_iter,
        "A":                rgb_array_to_base64(imgA),
        "B":                rgb_array_to_base64(imgB),
        "barA":             rgb_array_to_base64(barA),
        "barB":             rgb_array_to_base64(barB),
        "palette":          palA.tolist(),
    }

@app.post("/palette/choice/")
def record_choice(choice: str = Form(...)):
    global current_palette, challenger_palette, curr_iteration, n_iter, K

    if choice not in ["A", "B"]:
        raise HTTPException(400, "choice は 'A' か 'B' を指定してください")

    winner = current_palette if choice == "A" else challenger_palette
    loser  = challenger_palette if choice == "A" else current_palette

    record_duel(
        tuple(map(tuple, sort_by_lightness(winner))),
        tuple(map(tuple, sort_by_lightness(loser))),
        winner=tuple(map(tuple, sort_by_lightness(winner)))
    )

    curr_iteration += 1
    current_palette = winner
    if curr_iteration <= n_iter:
        challenger_palette = select_challenger_local(current_palette, curr_iteration, n_iter)

    return {"next_iteration": curr_iteration}

# ============================================================================
# 5) サーバ起動
# ============================================================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
