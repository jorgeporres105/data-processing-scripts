import pandas as pd
import matplotlib.pyplot as plt

# Leer CSV generado desde STM32
df = pd.read_csv(
    "../data/pi_run_01.csv",
    names=["t","ref","y","u","dcnt"]
)

# Gráfica referencia vs salida
plt.figure(figsize=(8,4))
plt.plot(df["t"], df["ref"], "--", label="Referencia")
plt.plot(df["t"], df["y"], label="Salida")
plt.xlabel("Tiempo [s]")
plt.ylabel("Velocidad [counts/s]")
plt.title("Respuesta al escalón – Control PI")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Guardar figura para el informe
plt.savefig("../figures/pi_step_response.png", dpi=300)
plt.show()
