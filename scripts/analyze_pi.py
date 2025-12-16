import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIGURACIÓN
# =========================
CSV_PATH = "../data/pi_run_01.csv"
FIG_PATH = "../figures/pi_response.png"
TXT_PATH = "../docs/pi_results.txt"

# Banda para tiempo de asentamiento (ej: 0.05 = ±5%)
SETTLING_BAND = 0.05

# Fracción final para estimar referencia final y métricas de régimen (ej: 0.2 = último 20%)
FINAL_FRACTION = 0.20


# =========================
# UTILIDADES
# =========================
def detect_step_time(t, ref):
    """
    Detecta el instante del escalón buscando el primer punto donde ref
    se aleja significativamente del valor inicial.
    """
    ref0 = np.median(ref[:max(10, len(ref)//50)])  # baseline robusto
    eps = max(1.0, 0.005 * max(1.0, abs(ref0)))   # umbral mínimo
    idx = np.where(np.abs(ref - ref0) > eps)[0]
    if len(idx) == 0:
        return t[0]
    return t[idx[0]]

def settling_time(t, y, ref_final, band_frac, t_step):
    """
    Tiempo de asentamiento: primer instante donde y permanece dentro de
    [ref_final*(1-band), ref_final*(1+band)] hasta el final.
    Devuelve Ts relativo al escalón.
    """
    low = ref_final * (1.0 - band_frac)
    high = ref_final * (1.0 + band_frac)

    # usar solo datos después del escalón
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


# =========================
# CARGA DE DATOS
# =========================
df = pd.read_csv(CSV_PATH, names=["t", "ref", "y", "u", "dcnt"])

t   = df["t"].values.astype(float)
ref = df["ref"].values.astype(float)
y   = df["y"].values.astype(float)
u   = df["u"].values.astype(float)

# Crear carpetas si no existen
os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
os.makedirs(os.path.dirname(TXT_PATH), exist_ok=True)

# =========================
# DETECCIÓN DE ESCALÓN + VALORES FINALES
# =========================
t_step = detect_step_time(t, ref)

# Tomar datos después del escalón
post = df[df["t"] >= t_step].copy()
t_post = post["t"].values.astype(float)
ref_post = post["ref"].values.astype(float)
y_post = post["y"].values.astype(float)

# Referencia final (promedio del último tramo)
n_final = max(10, int(len(ref_post) * FINAL_FRACTION))
ref_final = float(np.mean(ref_post[-n_final:]))

# =========================
# MÉTRICAS (post-escalón)
# =========================
# Error estacionario (promedio del último tramo)
y_final = float(np.mean(y_post[-n_final:]))
steady_state_error = ref_final - y_final

# Overshoot (%): pico sobre ref_final
y_peak = float(np.max(y_post))
overshoot_pct = ((y_peak - ref_final) / ref_final) * 100.0 if ref_final != 0 else np.nan

# RMSE (post-escalón completo)
rmse = float(np.sqrt(np.mean((ref_post - y_post) ** 2)))

# Tiempo de asentamiento
Ts, low_band, high_band = settling_time(t, y, ref_final, SETTLING_BAND, t_step)

# =========================
# FIGURA (comparación)
# =========================
plt.figure(figsize=(9, 4.5))
plt.plot(t, ref, "--", label="Referencia")
plt.plot(t, y, label="Salida")

# Marcar escalón y bandas
plt.axvline(t_step, linestyle=":", label="Inicio escalón")
plt.hlines([low_band, high_band], xmin=t_step, xmax=t[-1], linestyles=":", linewidth=1)

plt.xlabel("Tiempo [s]")
plt.ylabel("Velocidad [counts/s]")
plt.title("Respuesta al escalón – Control PI")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(FIG_PATH, dpi=300)
plt.show()

# =========================
# GUARDAR RESULTADOS TXT
# =========================
lines = []
lines.append("=== RESULTADOS CONTROL PI (desde CSV) ===\n")
lines.append(f"Archivo: {CSV_PATH}\n")
lines.append(f"t_step (inicio del escalón): {t_step:.3f} s\n")
lines.append(f"Referencia final (promedio último {int(FINAL_FRACTION*100)}%): {ref_final:.2f} counts/s\n")
lines.append(f"Banda de asentamiento: ±{SETTLING_BAND*100:.1f}% -> [{low_band:.2f}, {high_band:.2f}] counts/s\n\n")

lines.append("=== MÉTRICAS ===\n")
lines.append(f"Error estacionario (ref_final - y_final): {steady_state_error:.2f} counts/s\n")
lines.append(f"Overshoot: {overshoot_pct:.2f} %\n")
lines.append(f"RMSE (post-escalón): {rmse:.2f} counts/s\n")

if Ts is None:
    lines.append("Tiempo de asentamiento (Ts): No se asentó dentro de la banda en el intervalo medido.\n")
else:
    lines.append(f"Tiempo de asentamiento (Ts, ±{SETTLING_BAND*100:.1f}%): {Ts:.3f} s\n")

lines.append(f"\nFigura guardada en: {FIG_PATH}\n")

with open(TXT_PATH, "w", encoding="utf-8") as f:
    f.writelines(lines)

print("OK ✅")
print(f"- Figura: {FIG_PATH}")
print(f"- Resultados: {TXT_PATH}")
