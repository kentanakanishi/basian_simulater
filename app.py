"""
Bayesian Optimization Visualizer
GS-TS: Gaussian Process + Thompson Sampling
"""

import numpy as np
from flask import Flask, jsonify, render_template
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

app = Flask(__name__)

# ──────────────────────────────────────────────
# 定数
# ──────────────────────────────────────────────
DOMAIN = (-3.0, 3.0)
GRID_SIZE = 60          # 60×60 グリッド
N_INIT = 3              # 初期観測点数
N_GP_RESTARTS = 3       # GP カーネルパラメータ最適化の再試行数

# ──────────────────────────────────────────────
# グローバル状態
# ──────────────────────────────────────────────
state: dict = {}


# ──────────────────────────────────────────────
# 関数生成
# ──────────────────────────────────────────────
def _make_sin(rng):
    a, b = rng.uniform(-2.0, 2.0, 2)
    c = rng.uniform(-np.pi, np.pi)
    w = rng.uniform(-1.5, 1.5)
    return {"type": "sin", "w": w, "a": a, "b": b, "c": c}


def _make_cos(rng):
    a, b = rng.uniform(-2.0, 2.0, 2)
    c = rng.uniform(-np.pi, np.pi)
    w = rng.uniform(-1.5, 1.5)
    return {"type": "cos", "w": w, "a": a, "b": b, "c": c}


def _make_gauss(rng):
    cx, cy = rng.uniform(-2.0, 2.0, 2)
    sigma = rng.uniform(0.5, 2.0)
    w = rng.uniform(-1.5, 1.5)
    return {"type": "gauss", "w": w, "cx": cx, "cy": cy, "sigma": sigma}


def _make_tanh(rng):
    a, b = rng.uniform(-2.0, 2.0, 2)
    w = rng.uniform(-1.5, 1.5)
    return {"type": "tanh", "w": w, "a": a, "b": b}


def _make_poly(rng):
    a, b, c = rng.uniform(-0.4, 0.4, 3)
    w = rng.uniform(-1.5, 1.5)
    return {"type": "poly", "w": w, "a": a, "b": b, "c": c}


def generate_terms(rng) -> list[dict]:
    """ランダムな初等関数の線形結合を生成する。"""
    makers = [_make_sin, _make_cos, _make_gauss, _make_tanh, _make_poly]
    n = rng.integers(4, 8)
    chosen = rng.choice(makers, size=n)
    return [m(rng) for m in chosen]


def eval_terms(terms: list[dict], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """terms で定義された関数を (x, y) 配列で評価する。"""
    z = np.zeros_like(x, dtype=float)
    for t in terms:
        tp = t["type"]
        w = t["w"]
        if tp == "sin":
            z += w * np.sin(t["a"] * x + t["b"] * y + t["c"])
        elif tp == "cos":
            z += w * np.cos(t["a"] * x + t["b"] * y + t["c"])
        elif tp == "gauss":
            z += w * np.exp(
                -((x - t["cx"]) ** 2 + (y - t["cy"]) ** 2) / t["sigma"] ** 2
            )
        elif tp == "tanh":
            z += w * np.tanh(t["a"] * x + t["b"] * y)
        elif tp == "poly":
            z += w * (t["a"] * x ** 2 + t["b"] * y ** 2 + t["c"] * x * y)
    return z


def terms_to_formula(terms: list[dict]) -> str:
    """terms を人間が読める数式文字列に変換する。"""
    parts = []
    for t in terms:
        w = t["w"]
        tp = t["type"]
        if tp == "sin":
            inner = f"{t['a']:.2f}x{t['b']:+.2f}y{t['c']:+.2f}"
            parts.append(f"{w:+.2f}·sin({inner})")
        elif tp == "cos":
            inner = f"{t['a']:.2f}x{t['b']:+.2f}y{t['c']:+.2f}"
            parts.append(f"{w:+.2f}·cos({inner})")
        elif tp == "gauss":
            dx = f"x{-t['cx']:+.2f}"
            dy = f"y{-t['cy']:+.2f}"
            parts.append(f"{w:+.2f}·exp(-({dx})²+({dy})²)/{t['sigma']:.2f}²)")
        elif tp == "tanh":
            inner = f"{t['a']:.2f}x{t['b']:+.2f}y"
            parts.append(f"{w:+.2f}·tanh({inner})")
        elif tp == "poly":
            parts.append(f"{w:+.2f}·({t['a']:.2f}x²{t['b']:+.2f}y²{t['c']:+.2f}xy)")
    formula = " ".join(parts)
    # 先頭の '+' を除去
    if formula.startswith("+"):
        formula = formula[1:]
    return "f(x,y) = " + formula


# ──────────────────────────────────────────────
# GP フィット & Thompson Sampling
# ──────────────────────────────────────────────
def fit_gp_and_sample() -> None:
    """GP を観測点でフィットし、Thompson Sampling で次の候補点を決定する。"""
    X_obs = state["X_obs"]   # shape (n, 2)
    y_obs = state["y_obs"]   # shape (n,)
    X_grid = state["X_grid"]  # shape (G*G, 2)

    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
        * RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0))
        + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-10, 0.1))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=N_GP_RESTARTS,
        normalize_y=True,
    )
    gp.fit(X_obs, y_obs)

    mu, sigma = gp.predict(X_grid, return_std=True)

    # Thompson Sampling: 事後サンプルの argmax を次の提案点とする
    sample = mu + sigma * np.random.randn(len(X_grid))
    next_idx = int(np.argmax(sample))

    G = GRID_SIZE
    state["mu_2d"] = mu.reshape(G, G).tolist()
    state["sigma_2d"] = sigma.reshape(G, G).tolist()
    state["next_point"] = X_grid[next_idx].tolist()


# ──────────────────────────────────────────────
# 状態初期化
# ──────────────────────────────────────────────
def reset_state() -> None:
    """新しいランダム関数を生成し、初期観測点でGPを初期化する。"""
    global state
    rng = np.random.default_rng()

    # グリッド構築
    xs = np.linspace(DOMAIN[0], DOMAIN[1], GRID_SIZE)
    ys = np.linspace(DOMAIN[0], DOMAIN[1], GRID_SIZE)
    XX, YY = np.meshgrid(xs, ys)
    x_flat, y_flat = XX.ravel(), YY.ravel()
    X_grid = np.column_stack([x_flat, y_flat])

    # 関数生成
    terms = generate_terms(rng)
    f_flat = eval_terms(terms, x_flat, y_flat)
    f_2d = f_flat.reshape(GRID_SIZE, GRID_SIZE)

    # 初期観測点 (ランダム)
    init_x = rng.uniform(DOMAIN[0], DOMAIN[1], N_INIT)
    init_y = rng.uniform(DOMAIN[0], DOMAIN[1], N_INIT)
    X_obs = np.column_stack([init_x, init_y])
    y_obs = eval_terms(terms, init_x, init_y)

    state = {
        "terms": terms,
        "formula": terms_to_formula(terms),
        "xs": xs.tolist(),
        "ys": ys.tolist(),
        "X_grid": X_grid,
        "f_2d": f_2d.tolist(),
        "X_obs": X_obs,
        "y_obs": y_obs,
        "mu_2d": None,
        "sigma_2d": None,
        "next_point": None,
        "step": 0,
    }

    fit_gp_and_sample()


# ──────────────────────────────────────────────
# レスポンス構築
# ──────────────────────────────────────────────
def build_response() -> dict:
    X_obs = state["X_obs"]
    y_obs = state["y_obs"]
    best_idx = int(np.argmax(y_obs))
    return {
        "formula": state["formula"],
        "xs": state["xs"],
        "ys": state["ys"],
        "f_grid": state["f_2d"],
        "mu_grid": state["mu_2d"],
        "sigma_grid": state["sigma_2d"],
        "obs_x": X_obs[:, 0].tolist(),
        "obs_y": X_obs[:, 1].tolist(),
        "obs_f": y_obs.tolist(),
        "best_point": X_obs[best_idx].tolist(),
        "best_f": float(y_obs[best_idx]),
        "next_point": state["next_point"],
        "step": state["step"],
    }


# ──────────────────────────────────────────────
# ルート
# ──────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/reset")
def api_reset():
    reset_state()
    return jsonify(build_response())


@app.route("/api/step", methods=["POST"])
def api_step():
    if not state or state.get("next_point") is None:
        return jsonify({"error": "先に /api/reset を呼んでください"}), 400

    nx, ny = state["next_point"]
    nf = float(
        eval_terms(state["terms"], np.array([nx]), np.array([ny]))[0]
    )

    state["X_obs"] = np.vstack([state["X_obs"], [[nx, ny]]])
    state["y_obs"] = np.append(state["y_obs"], nf)
    state["step"] += 1

    fit_gp_and_sample()
    return jsonify(build_response())


# ──────────────────────────────────────────────
if __name__ == "__main__":
    reset_state()
    app.run(debug=True, port=5000)
