import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Rutas robustas
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DATA_DIR = os.path.join(REPO_ROOT, "data")
FIG_DIR  = os.path.join(REPO_ROOT, "figures")
RES_DIR  = os.path.join(REPO_ROOT, "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

PATTERNS = {
    "PI":      os.path.join(DATA_DIR, "pi_run_*.csv"),
    "PID":     os.path.join(DATA_DIR, "pid_run_*.csv"),
    "LQR/LQI": os.path.join(DATA_DIR, "lqr_run_*.csv"),
}

SETTLING_BAND = 0.05   # ±5%
STEADY_FRAC   = 0.20   # último 20%
COMMON_FS_HZ  = 10.0   # 0.1s si logueas así
T_WINDOW_AFTER_STEP = 12.0

# =========================
# Utilidades de detección
# =========================
def _is_monotonic_increasing(x):
    dx = np.diff(x)
    return np.all(dx >= -1e-9) and np.sum(dx > 0) > 3

def _score_time_candidate(x):
    # bueno si es monotónico y rango razonable
    if not _is_monotonic_increasing(x):
        return -1e9
    rng = float(np.nanmax(x) - np.nanmin(x))
    # penaliza si es demasiado pequeño o demasiado grande
    return -abs(rng - 15.0)  # típicamente ~10-20s

def _find_step_index(sig):
    d = np.abs(np.diff(sig))
    idx = np.where(d > 1e-6)[0]
    return int(idx[0] + 1) if len(idx) > 0 else 0

def _score_ref_candidate(x):
    # ref suele ser piecewise-constant con un escalón fuerte
    idx = _find_step_index(x)
    if idx == 0:
        return -1e9
    # tiene pocos valores únicos -> bueno
    uniq = len(np.unique(np.round(x, 3)))
    return -uniq  # menos únicos = mejor

def _score_u_candidate(x):
    # u normalmente está en [-1,1] (o muy cerca)
    mx = float(np.nanmax(np.abs(x)))
    return -abs(mx - 1.0) if mx <= 1.5 else -1e9


def read_run(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None)
    if raw.shape[1] < 4:
        raise ValueError(f"{os.path.basename(path)} no tiene 4 columnas.")

    # Tomamos solo las primeras 4 columnas por si hay extras
    A = raw.iloc[:, :4].copy()
    A = A.replace([np.inf, -np.inf], np.nan).dropna()
    for c in A.columns:
        A[c] = A[c].astype(float)

    cols = [A.iloc[:, i].values for i in range(4)]

    # 1) Detectar u por rango [-1,1]
    u_idx = int(np.argmax([_score_u_candidate(c) for c in cols]))

    # 2) Detectar t por monotonicidad
    time_scores = [_score_time_candidate(c) for c in cols]
    # Evitar escoger u como tiempo
    time_scores[u_idx] = -1e9
    t_idx = int(np.argmax(time_scores))

    # 3) Quedan 2 columnas: ref y y. Elegir ref por “escalón”
    rem = [i for i in range(4) if i not in (u_idx, t_idx)]
    ref_scores = [_score_ref_candidate(cols[i]) for i in rem]
    ref_idx = rem[int(np.argmax(ref_scores))]
    y_idx = [i for i in rem if i != ref_idx][0]

    df = pd.DataFrame({
        "t":   cols[t_idx],
        "ref": cols[ref_idx],
        "y":   cols[y_idx],
        "u":   cols[u_idx],
    })

    # Limpieza mínima: ordenar por tiempo
    df = df.sort_values("t").reset_index(drop=True)
    return df


def step_time(df: pd.DataFrame) -> float:
    ref = df["ref"].values
    t = df["t"].values
    idx = np.where(np.diff(ref) != 0)[0]
    return float(t[idx[0] + 1]) if len(idx) > 0 else float(t[0])


def steady_value(df: pd.DataFrame) -> float:
    y = df["y"].values
    n = len(y)
    k0 = int((1.0 - STEADY_FRAC) * n)
    return float(np.mean(y[k0:]))


def compute_metrics(df: pd.DataFrame) -> dict:
    t = df["t"].values
    ref = df["ref"].values
    y = df["y"].values

    t0 = step_time(df)
    mask = t >= t0
    t2, ref2, y2 = t[mask], ref[mask], y[mask]

    ref_final = float(np.mean(ref2[int(0.8 * len(ref2)):]))

    y_ss = steady_value(df)
    e_ss = ref_final - y_ss

    y_peak = float(np.max(y2))
    overshoot = 0.0
    if abs(ref_final) > 1e-9:
        overshoot = max(0.0, (y_peak - ref_final) / abs(ref_final) * 100.0)

    rmse = float(np.sqrt(np.mean((ref2 - y2) ** 2)))

    band = SETTLING_BAND * abs(ref_final)
    ts = np.nan
    if band >= 1e-9:
        within = np.abs(y2 - ref_final) <= band
        for i in range(len(within)):
            if np.all(within[i:]):
                ts = float(t2[i] - t0)
                break

    return {
        "t_step": t0,
        "ref_final": ref_final,
        "y_ss": y_ss,
        "e_ss": e_ss,
        "overshoot_pct": overshoot,
        "rmse": rmse,
        "settling_time_s": ts
    }


def common_time_grid() -> np.ndarray:
    dt = 1.0 / COMMON_FS_HZ
    return np.arange(0.0, T_WINDOW_AFTER_STEP + 1e-9, dt)


def interp_to_grid_rel(df: pd.DataFrame, tgrid_rel: np.ndarray) -> dict:
    t0 = step_time(df)
    t_rel = df["t"].values - t0
    out = {}
    for col in ["ref", "y", "u"]:
        out[col] = np.interp(tgrid_rel, t_rel, df[col].values)
    return out


def plot_overlay_and_mean(name: str, runs: list[pd.DataFrame], out_prefix: str):
    tgrid = common_time_grid()

    Rs, Ys, Us = [], [], []
    for df in runs:
        itp = interp_to_grid_rel(df, tgrid)
        Rs.append(itp["ref"])
        Ys.append(itp["y"])
        Us.append(itp["u"])

    Rm = np.mean(np.vstack(Rs), axis=0)
    Ym = np.mean(np.vstack(Ys), axis=0)
    Um = np.mean(np.vstack(Us), axis=0)

    # y(t)
    plt.figure()
    for y in Ys:
        plt.plot(tgrid, y, alpha=0.25)
    plt.plot(tgrid, Rm, label="Referencia (promedio)")
    plt.plot(tgrid, Ym, label=f"Salida {name} (promedio)", linewidth=2.0)
    plt.xlabel("Tiempo relativo al escalón (s)")
    plt.ylabel("Velocidad (counts/s)")
    plt.title(f"{name}: Respuesta al escalón (20 ensayos)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{out_prefix}_{name.lower().replace('/','_')}_step.png"), dpi=200)
    plt.close()

    # u(t)
    plt.figure()
    for u in Us:
        plt.plot(tgrid, u, alpha=0.25)
    plt.plot(tgrid, Um, label=f"u {name} (promedio)", linewidth=2.0)
    plt.xlabel("Tiempo relativo al escalón (s)")
    plt.ylabel("u (normalizada)")
    plt.title(f"{name}: Señal de control u(t) (20 ensayos)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{out_prefix}_{name.lower().replace('/','_')}_u.png"), dpi=200)
    plt.close()

    return {"tgrid": tgrid, "ref_mean": Rm, "y_mean": Ym, "u_mean": Um}


def plot_three_controller_means(mean_curves: dict):
    first_key = next(iter(mean_curves.keys()))
    tgrid = mean_curves[first_key]["tgrid"]

    # step compare
    plt.figure()
    plt.plot(tgrid, mean_curves[first_key]["ref_mean"], label="Referencia")
    for ctrl, s in mean_curves.items():
        plt.plot(tgrid, s["y_mean"], label=f"Salida {ctrl}")
    plt.xlabel("Tiempo relativo al escalón (s)")
    plt.ylabel("Velocidad (counts/s)")
    plt.title("Comparación de respuesta promedio (20 ensayos por controlador)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "step_compare_pi_pid_lqr_mean.png"), dpi=200)
    plt.close()

    # u compare
    plt.figure()
    for ctrl, s in mean_curves.items():
        plt.plot(tgrid, s["u_mean"], label=f"u {ctrl}")
    plt.xlabel("Tiempo relativo al escalón (s)")
    plt.ylabel("u (normalizada)")
    plt.title("Comparación de señal de control promedio (20 ensayos por controlador)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "u_compare_pi_pid_lqr_mean.png"), dpi=200)
    plt.close()


def main():
    all_metrics_rows = []
    mean_curves = {}

    for ctrl, pat in PATTERNS.items():
        files = sorted(glob.glob(pat))
        print(f"{ctrl}: {len(files)} archivos encontrados en data/")

    for ctrl_name, pattern in PATTERNS.items():
        files = sorted(glob.glob(pattern))
        if len(files) == 0:
            raise FileNotFoundError(f"No se encontraron CSV para {ctrl_name} con patrón: {pattern}")

        runs = [read_run(p) for p in files]

        # métricas por run
        ctrl_metrics = []
        for p, df in zip(files, runs):
            m = compute_metrics(df)
            all_metrics_rows.append({"controlador": ctrl_name, "archivo": os.path.basename(p), **m})
            ctrl_metrics.append(m)

        dfm = pd.DataFrame(ctrl_metrics)
        summary = dfm.agg(["mean", "std"])

        out_sum_txt = os.path.join(RES_DIR, f"{ctrl_name.lower().replace('/','_')}_summary_20runs.txt")
        with open(out_sum_txt, "w", encoding="utf-8") as f:
            f.write(f"=== RESUMEN 20 ENSAYOS: {ctrl_name} ===\n\n")
            for col in ["e_ss", "overshoot_pct", "rmse", "settling_time_s"]:
                mu = summary.loc["mean", col]
                sd = summary.loc["std", col]
                if np.isnan(mu):
                    f.write(f"{col}: N/A\n")
                else:
                    f.write(f"{col}: {mu:.3f} ± {sd:.3f}\n")

        mean_curves[ctrl_name] = plot_overlay_and_mean(ctrl_name, runs, out_prefix="overlay")

    df_all = pd.DataFrame(all_metrics_rows)
    df_all.to_csv(os.path.join(RES_DIR, "metrics_all_runs.csv"), index=False)

    # resumen global robusto
    summary_rows = []
    for ctrl in PATTERNS.keys():
        sub = df_all[df_all["controlador"] == ctrl]
        row = {"controlador": ctrl}
        for col in ["e_ss", "overshoot_pct", "rmse", "settling_time_s"]:
            vals = sub[col].values.astype(float)
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                row[f"{col}_mean"] = np.nan
                row[f"{col}_std"]  = np.nan
            else:
                row[f"{col}_mean"] = float(np.mean(vals))
                row[f"{col}_std"]  = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        summary_rows.append(row)

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(os.path.join(RES_DIR, "metrics_summary_20runs.csv"), index=False)

    plot_three_controller_means(mean_curves)

    out_report = os.path.join(RES_DIR, "metrics_summary_20runs.txt")
    with open(out_report, "w", encoding="utf-8") as f:
        f.write("=== RESUMEN GLOBAL (20 ENSAYOS POR CONTROLADOR) ===\n\n")
        for _, r in df_sum.iterrows():
            ctrl = r["controlador"]
            f.write(f"[{ctrl}]\n")
            if np.isnan(r["e_ss_mean"]):
                f.write("No hay datos válidos para este controlador.\n\n")
                continue
            f.write(f"e_ss: {r['e_ss_mean']:.3f} ± {r['e_ss_std']:.3f} counts/s\n")
            f.write(f"Overshoot: {r['overshoot_pct_mean']:.3f} ± {r['overshoot_pct_std']:.3f} %\n")
            f.write(f"RMSE: {r['rmse_mean']:.3f} ± {r['rmse_std']:.3f} counts/s\n")
            if np.isnan(r["settling_time_s_mean"]):
                f.write("Ts (±5%): N/A\n\n")
            else:
                f.write(f"Ts (±5%): {r['settling_time_s_mean']:.3f} ± {r['settling_time_s_std']:.3f} s\n\n")

    print("\nListo ✅")
    print("Figuras -> figures")
    print("Resultados -> results")


if __name__ == "__main__":
    main()
