from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
from PIL import Image
import io
import colorsys
import base64
import os

# 1. API definition
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
# CORS configuration: allow all origins, all methods, all headers


from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

static_path = (ROOT_PATH + "/static").rstrip("/")
app.mount(
    static_path,
    StaticFiles(directory=".", html=True),
    name="static",
)
# Serve all frontend assets (HTML, CSS, JS, images) directly from the project root
# under the /static URL, so everything (UI and API) is hosted from one server/directory.

@app.get(ROOT_PATH + "/", include_in_schema=True)
async def root():
    return FileResponse("index.html")
# Assign the root URL ("/") to return index.html as the main entry point,
# keeping this UI route out of the auto-generated API docs.

# 2. Global state
# Type annotation: np.ndarray or None (Optional[np.ndarray])
# None indicates "not yet uploaded"
gray_map: np.ndarray | None = None
current_palette: np.ndarray | None = None
challenger_palette: np.ndarray | None = None
n_iter: int = 20
K: int = 5
plays: dict = {}
wins: dict = {}
curr_iteration: int = 1

# 3 Utility function
def rgb_array_to_base64(arr: np.ndarray) -> str:
    img = Image.fromarray((arr * 255).astype(np.uint8))
    # Convert floating-point RGB array (0.0–1.0) to uint8 (0–255) and create a PIL (Python Imaging Library) Image
    buf = io.BytesIO()
    # Prepare an in-memory buffer (a “file” in RAM)
    img.save(buf, format="JPEG")
    # Write the image data into the buffer in JPEG format
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    # Retrieve all bytes from the buffer and Base64-encode them.
    # By returning a data URL, the browser can reconstruct the image solely from this string.

def sort_by_lightness(hsl_pts: np.ndarray) -> np.ndarray:
    # hsl_pts: Array of HSL color points (hue, lightness, saturation)
    # Sort the HSL palette by the lightness component in ascending order.
    # This ensures a smooth gradient from darkest to brightest colors.
    return hsl_pts[np.argsort(hsl_pts[:,1])]

def hls2rgb_array(hsl_pts: np.ndarray) -> np.ndarray:
    # Convert each HSL (hue, lightness, saturation) point to an RGB tuple
    # using colorsys.hls_to_rgb, and return as a NumPy array.
    # The resulting RGB values are in the 0.0–1.0 range, ready for display.
    return np.array([colorsys.hls_to_rgb(h, l, s) for h, l, s in hsl_pts])

def apply_gradient_mixed(gray: np.ndarray, hsl_pts: np.ndarray) -> np.ndarray:
    # Given a grayscale image (H×W float array in range 0.0–1.0) and
    # an HSL palette array (hue, lightness, saturation),
    # returns an RGB image with the gradient applied.
    H, W = gray.shape
    k = len(hsl_pts)
    hsl_sorted = sort_by_lightness(hsl_pts)
    # Sort the palette by lightness,
    rgb = hls2rgb_array(hsl_sorted)
    # Convert sorted HSL to RGB values (0.0–1.0)
    g = gray.ravel()
    # Flatten the image to simplify per-pixel processing
    seg = np.minimum((g * (k - 1)).astype(int), k - 2)
    # Map brightness g to a [0, k−1] scale and compute integer segment index seg
    t = g * (k - 1) - seg
    # Compute the fractional position t within each segment (0.0–1.0)
    c0 = rgb[seg]
    c1 = rgb[seg + 1]
    # Lookup lower color c0 and upper color c1 for each seg
    out = (1 - t)[:, None] * c0 + t[:, None] * c1
    # Linearly interpolate final color: (1−t)*c0 + t*c1
    return out.reshape(H, W, 3)
    # Reshape back to (H, W, 3) and return as RGB image


def make_grad_bar_mixed(hsl_pts: np.ndarray, width=300, height=20) -> np.ndarray:
    # Create an RGB gradient bar image of given width and height from an HSL palette.
    k = len(hsl_pts)
    hsl_sorted = sort_by_lightness(hsl_pts)
    # Sort the palette by lightness,
    rgb = hls2rgb_array(hsl_sorted)
    # Convert sorted HSL to RGB values (0.0–1.0)
    xs = np.linspace(0, 1, width)
    # xs is a 1D array of length 'width', evenly spaced from 0.0 to 1.0
    seg = np.minimum((xs * (k - 1)).astype(int), k - 2)
    # Map brightness g to a [0, k−1] scale and compute integer segment index seg
    t = xs * (k - 1) - seg
    # Compute the fractional position t within each segment (0.0–1.0)
    c0 = rgb[seg]
    c1 = rgb[seg + 1]
    # Lookup lower color c0 and upper color c1 for each seg
    bar = (1 - t)[:, None] * c0 + t[:, None] * c1
    # Linearly interpolate final color: (1−t)*c0 + t*c1
    return np.tile(bar[None, :, :], (height, 1, 1))
    # np.tile: tile 'bar[None, :, :]' vertically 'height' times,
    # resulting in a gradient bar image of shape (height, width, 3)

def record_duel(i, j, winner):
    # Record the duel result between palettes i and j, updating wins and plays.
    key = (i, j) if winner == i else (j, i)
    wins.setdefault(key, 0)
    plays.setdefault((i, j), 0)
    plays.setdefault((j, i), 0)
    # Initialize wins[key], plays[(i, j)], and plays[(j, i)] to 0 if not present
    wins[key] += 1
    plays[(i, j)] += 1
    plays[(j, i)] += 1

def rucb_score(i, j, t, alpha=1.0):
    nij = plays.get((i, j), 0)
    # plays.get(key, default):  
    #   - returns the value for ‘key’ if it exists in the dict,  
    #   - otherwise returns ‘default’  
    if nij == 0:
        return 1.0
    pij = wins.get((i, j), 0) / nij
    return pij + np.sqrt(alpha * np.log(t) / nij)

def select_challenger_local(curr_hsl: np.ndarray, t: int, n_iter: int, M=50, sigma0=0.3) -> np.ndarray:
    # Generate M perturbed palette candidates around curr_hsl,
    # evaluate each with the R-UCB score, and return the best one.
    k = curr_hsl.shape[0]
    t_eff = min(t, n_iter)
    sigma_t = sigma0 * (1 - (t_eff - 1) / n_iter)
    # sigma_t: compute decayed noise stddev
    cands = curr_hsl[None] + np.random.normal(scale=sigma_t, size=(M, k, 3))
    # cands: Mcreate M candidates by adding Gaussian noise and clip to [0,1]
    #   curr_hsl[None]: expand to shape (1, k, 3) so it can be broadcast against (M, k, 3)
    cands = np.clip(cands, 0.0, 1.0)
    # np.clip(..., 0.0, 1.0): clamp values to the valid HSL range [0.0, 1.0] (below -> 0.0, above -> 1.0)
    best, best_s = None, -np.inf
    curr_id = tuple(map(tuple, sort_by_lightness(curr_hsl)))
    for h in cands:
        h_id = tuple(map(tuple, sort_by_lightness(h)))
        s = rucb_score(curr_id, h_id, t_eff)
        if s > best_s:
            best_s, best = s, h
    return best

# 4. Endpoint definition
# POST /upload/: receive image file and parameters, convert to gray, reset state
# Upload endpoint definition
@app.post("/upload/")
async def upload_image(
    # Initial processing function after upload (asynchronous: optional)
    file: UploadFile = File(...),
    # Uploaded image file (required)
    n_iter_form: int = Form(20),
    # Total number of iterations (from form, default=20)
    k_form: int = Form(5),
    # Number of colors in palette (from form, default=5)
):
    data = await file.read()
    # await: Asynchronously read the uploaded file into bytes
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        # Try to open the bytes as an image and convert to RGB
    except:
        raise HTTPException(400, "画像の読み込みに失敗しました / Failed to load image")
        # If image loading fails, return HTTP 400 error

    # Convert the PIL Image to a NumPy array and normalize pixel values to [0.0, 1.0]
    arr = np.asarray(img) / 255.0
    gray = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])
    # Note: using luminance (a linear, weighted sum), not lightness

    global gray_map, current_palette, challenger_palette
    global curr_iteration, wins, plays, n_iter, K

    # Parameter update
    n_iter = n_iter_form
    K = k_form

    # State reset
    gray_map = gray
    current_palette = None
    challenger_palette = None
    curr_iteration = 1
    wins = {}
    plays = {}

    return {"message": f"Upload OK. n_iter={n_iter}, K={K}", "shape": gray.shape}

@app.get("/palette/next/")
def get_next_pair():
    # Retrieve the next palette candidates (A/B) images and metadata
    global gray_map, current_palette, challenger_palette, curr_iteration, n_iter, K

    if gray_map is None:
        raise HTTPException(400, "先に /upload/ で画像をアップロードしてください / Please upload an image via the /upload/ endpoint first")

    if current_palette is None:
        current_palette = np.random.rand(K, 3)
        challenger_palette = select_challenger_local(current_palette, curr_iteration, n_iter)

    imgA = apply_gradient_mixed(gray_map, current_palette)
    imgB = apply_gradient_mixed(gray_map, challenger_palette)

    # Gradient bars & sorted palette
    barA = make_grad_bar_mixed(current_palette)
    barB = make_grad_bar_mixed(challenger_palette)
    palA = sort_by_lightness(current_palette)
    palB = sort_by_lightness(challenger_palette)

    return {
        "iteration": curr_iteration,
        "total_iterations": n_iter,
        "A": rgb_array_to_base64(imgA),
        "B": rgb_array_to_base64(imgB),
        "barA": rgb_array_to_base64(barA),
        "barB": rgb_array_to_base64(barB),
        "palA": palA.tolist(),
        "palB": palB.tolist(),
    }

@app.post("/palette/choice/")
def record_choice(choice: str = Form(...)):
    # Receive user choice (A/B), record the duel result, and prepare for the next iteration
    global current_palette, challenger_palette, curr_iteration, n_iter, K

    if choice not in ["A", "B"]:
        raise HTTPException(400, "choice は 'A' か 'B' を指定してください / Please specify choice as 'A' or 'B'")

    if choice == "A":
        winner = current_palette
        loser = challenger_palette
    else:
        winner = challenger_palette
        loser = current_palette

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

# 5. For server startup
if __name__ == "__main__":
    # Use PORT environment variable if set by hosting service, default to 8000
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    # Launch the FastAPI app with Uvicorn, listening on all interfaces
    uvicorn.run(app, host=host, port=port)

# 6. Memo

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