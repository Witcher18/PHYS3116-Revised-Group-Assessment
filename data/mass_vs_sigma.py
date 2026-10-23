import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import linregress

gama = pd.read_csv("data/InputCatGAMADR3.csv")
clusters = pd.read_csv("data/samiDR3InputCatClusters.csv")
skinedata = pd.read_csv("data/samiDR3StelKin.csv")
visualdata = pd.read_csv("data/samiDR3VisualMorphology.csv")

photodata = pd.concat([gama, clusters], ignore_index=True).drop_duplicates(subset=["catid"])

velocity_dispersion = "sigma_re"

unified = (
    photodata
    .merge(skinedata, on="catid")
    .merge(visualdata, on="catid")
    .loc[lambda df: df["bad_class"].isin([0, 5, 8])]
    .loc[lambda df: df[velocity_dispersion].notna()]
    .loc[lambda df: df[velocity_dispersion] > 35]
    .loc[lambda df: df[velocity_dispersion + "_err"] < df[velocity_dispersion] * 0.1 + 25]
    .loc[lambda df: df["type"].between(0, 0.5)]
    .loc[lambda df: df["mstar"].notna()]
)

print(f"Sample size: {len(unified)} galaxies")

x_vals = np.log10(unified[velocity_dispersion])
y_vals = unified["mstar"]
res = linregress(x_vals, y_vals)

fig, ax = plt.subplots(figsize=(7, 5))

ax.scatter(x_vals, y_vals, s=12, color="purple", alpha=0.6, edgecolors="none", label="SAMI Early-Types")

x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
sign = "-" if res.intercept < 0 else "+"
label_fit = rf"$\log_{{10}}(M_*) = {res.slope:.2f}\,\log_{{10}}(\sigma) {sign} {abs(res.intercept):.2f}$ ($R^2={res.rvalue**2:.2f}$)"

ax.plot(x_line, res.slope * x_line + res.intercept, color="crimson", linewidth=2, label=label_fit)

ax.set_xlabel(r"$\log_{10}(\sigma_{\mathrm{re}}\ [\mathrm{km/s}])$", fontsize=11)
ax.set_ylabel(r"$\log_{10}(M_* / M_\odot)$", fontsize=11)
ax.set_title(r"Faber-Jackson Relation: Stellar Mass vs Velocity Dispersion", fontsize=12)
ax.legend(frameon=True, loc="upper left")
ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

print(f"Slope:     {res.slope:.3f} ± {res.stderr:.3f}")
print(f"Intercept: {res.intercept:.3f} ± {res.intercept_stderr:.3f}")
print(f"R²:        {res.rvalue**2:.3f}")