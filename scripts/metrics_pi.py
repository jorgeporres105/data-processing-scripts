import pandas as pd
import numpy as np

# =============================
# Cargar datos
# =============================
df = pd.read_csv(
    "../data/pi_run_01.csv",
    names=["t", "ref", "y", "u", "dcnt"]
)

# =============================
# Usar solo la parte en régimen
# (descartamos los primeros 2 s)
# =============================
df_ss = df[df["t"] > 2.0]

ref = df_ss["ref"].values
y   = df_ss["y"].values

# =============================
# Métricas
# =============================

# Error estacionario (promedio)
steady_state_error = np.mean(ref - y)

# Overshoot (%)
overshoot = (np.max(y) - np.mean(ref)) / np.mean(ref) * 100.0

# RMSE
rmse = np.sqrt(np.mean((ref - y)**2))

# =============================
# Resultados
# =============================
print("=== MÉTRICAS CONTROL PI ===")
print(f"Error estacionario promedio: {steady_state_error:.2f} counts/s")
print(f"Overshoot: {overshoot:.2f} %")
print(f"RMSE: {rmse:.2f} counts/s")
