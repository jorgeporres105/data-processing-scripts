import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
DATA_GLOB = "../data/pid_run_*.csv"      # ajusta si tus nombres difieren
OUT_FIG_DIR = "../figures"
OUT_DOC_DIR = "../docs"

SETTLING_BAND = 0.05    # ±5%
FINAL_FRACTION = 0.20   # último 20% para ref_final y métricas

os.makedirs(OUT_FIG_DIR, exist_ok=True)
os.makedirs(OUT_DOC_DIR, exist_ok=True)

# =========================
# UTILIDADES
# =========================
def detect_step_time(t, ref):
    ref0 = np.median(ref[:max(10, len(ref)//50)])
    eps = max(1.0, 0.005 * max(1.0, abs(ref0)))
    idx = np.where(np.abs(ref - ref0) > eps)[0]
    return t[idx[0]] if len(idx) else t[0]

def settling_time(t, y, ref_final, band_frac, t_step):
    low = ref_final * (1.0 - band_frac)
    high = ref_final * (1.0 + band_frac)

    mask = t >= t_step
    t2 = t[mask]
    y2 = y[mask]
    inside = (y2 >= low) & (y2 <= high)

    Ts = None
    for i in range(len(t2)):
        if np.all(inside[i:]):
            Ts = t2[i] - t_step
            break
    return Ts, low, high

def analyze_one(csv_path):
    df = pd.read_csv(csv_path, names=["t","ref","y","u","dcnt"])

    t   = df["t"].astype(float).values
    ref = df["ref"].astype(float).values
    y   = df["y"].astype(float).values
    u   = df["u"].astype(float).values

    t_step = detect_step_time(t, ref)

    post = df[df["t"] >= t_step].copy()
    ref_post = post["ref"].astype(float).values
    y_post   = post["y"].astype(float).values

    n_final = max(10, int(len(ref_post) * FINAL_FRACTION))
    ref_final = float(np.mean(ref_post[-n_final:]))
    y_final   = float(np.mean(y_post[-n_final:]))

    steady_state_error = ref_final - y_final
    y_peak = float(np.max(y_post))
    overshoot_pct = ((y_peak - ref_final) / ref_final) * 100.0 if ref_final != 0 else np.nan
    rmse = float(np.sqrt(np.mean((ref_post - y_post) ** 2)))

    Ts, low_band, high_band = settling_time(t, y, ref_final, SETTLING_BAND, t_step)

    return {
        "t_step": t_step,
        "ref_final": ref_final,
        "y_final": y_final,
        "steady_state_error": steady_state_error,
        "overshoot_pct": overshoot_pct,
        "rmse": rmse,
        "Ts": Ts,
        "low_band": low_band,
        "high_band": high_band,
        "df": df
    }

def save_figure(run_id, res):
    df = res["df"]
    t = df["t"].values
    ref = df["ref"].values
    y = df["y"].values

    plt.figure(figsize=(9, 4.5))
    plt.plot(t, ref, "--", label="Referencia")
    plt.plot(t, y, label="Salida")
    plt.axvline(res["t_step"], linestyle=":", label="Inicio escalón")
    plt.hlines([res["low_band"], res["high_band"]],
               xmin=res["t_step"], xmax=t[-1], linestyles=":", linewidth=1)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Velocidad [counts/s]")
    plt.title(f"Respuesta al escalón – PI ({run_id})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(OUT_FIG_DIR, f"{run_id}_response.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    return fig_path

def save_txt(run_id, csv_path, fig_path, res):
    txt_path = os.path.join(OUT_DOC_DIR, f"{run_id}_results.txt")

    lines = []
    lines.append(f"=== RESULTADOS PID ({run_id}) ===\n")
    lines.append(f"Archivo: {csv_path}\n")
    lines.append(f"Figura:  {fig_path}\n\n")

    lines.append(f"t_step (inicio escalón): {res['t_step']:.3f} s\n")
    lines.append(f"Referencia final (último {int(FINAL_FRACTION*100)}%): {res['ref_final']:.2f} counts/s\n")
    lines.append(f"Banda asentamiento: ±{SETTLING_BAND*100:.1f}% -> [{res['low_band']:.2f}, {res['high_band']:.2f}] counts/s\n\n")

    lines.append("=== MÉTRICAS ===\n")
    lines.append(f"Error estacionario (ref_final - y_final): {res['steady_state_error']:.2f} counts/s\n")
    lines.append(f"Overshoot: {res['overshoot_pct']:.2f} %\n")
    lines.append(f"RMSE (post-escalón): {res['rmse']:.2f} counts/s\n")
    if res["Ts"] is None:
        lines.append("Tiempo de asentamiento (Ts): No se asentó dentro de la banda en el intervalo medido.\n")
    else:
        lines.append(f"Tiempo de asentamiento (Ts, ±{SETTLING_BAND*100:.1f}%): {res['Ts']:.3f} s\n")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return txt_path

# =========================
# MAIN
# =========================
csv_files = sorted(glob.glob(DATA_GLOB))
if not csv_files:
    raise FileNotFoundError(f"No se encontraron CSV con el patrón: {DATA_GLOB}")

summary_rows = []

for csv_path in csv_files:
    run_id = os.path.splitext(os.path.basename(csv_path))[0]

    res = analyze_one(csv_path)
    fig_path = save_figure(run_id, res)
    txt_path = save_txt(run_id, csv_path, fig_path, res)

    summary_rows.append({
        "run": run_id,
        "ref_final": res["ref_final"],
        "steady_state_error": res["steady_state_error"],
        "overshoot_pct": res["overshoot_pct"],
        "rmse": res["rmse"],
        "Ts": res["Ts"] if res["Ts"] is not None else np.nan
    })

    print(f"OK: {run_id} -> fig + txt")

# Guardar resumen global
summary = pd.DataFrame(summary_rows)

summary_txt = os.path.join(OUT_DOC_DIR, "pid_summary.txt")
with open(summary_txt, "w", encoding="utf-8") as f:
    f.write("=== RESUMEN GLOBAL PID (todos los ensayos) ===\n\n")
    f.write(f"Ensayos analizados: {len(summary)}\n")
    f.write(f"Banda Ts: ±{SETTLING_BAND*100:.1f}%\n\n")

    def wstat(name, series):
        f.write(f"{name}:\n")
        f.write(f"  mean = {np.nanmean(series):.3f}\n")
        f.write(f"  std  = {np.nanstd(series):.3f}\n")
        f.write(f"  min  = {np.nanmin(series):.3f}\n")
        f.write(f"  max  = {np.nanmax(series):.3f}\n\n")

    wstat("Error estacionario (counts/s)", summary["steady_state_error"].values)
    wstat("Overshoot (%)", summary["overshoot_pct"].values)
    wstat("RMSE (counts/s)", summary["rmse"].values)
    wstat("Ts (s)", summary["Ts"].values)

print(f"\nResumen guardado en: {summary_txt}")
print("Listo ✅")
