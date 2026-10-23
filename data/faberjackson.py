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
    .loc[lambda df: df["m_r"].notna()]
)

if len(unified) == 0:
    raise ValueError("No galaxies matched the selection criteria. Check your file paths or filters!")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

x_mag = unified["m_r"]
y_mag = np.log10(unified[velocity_dispersion])
res_mag = linregress(x_mag, y_mag)

axes[0].scatter(x_mag, y_mag, s=12, color="royalblue", alpha=0.6, edgecolors="none")
x_line_mag = np.linspace(x_mag.min(), x_mag.max(), 100)

sign_mag = "-" if res_mag.intercept < 0 else "+"
axes[0].plot(
    x_line_mag,
    res_mag.slope * x_line_mag + res_mag.intercept,
    color="crimson",
    linewidth=2,
    label=rf"$\log_{{10}}(\sigma) = {res_mag.slope:.2f}\,M_r {sign_mag} {abs(res_mag.intercept):.2f}$ ($R^2={res_mag.rvalue**2:.2f}$)"
)

axes[0].invert_xaxis()
axes[0].set_xlabel(r"$M_r$ (mags)", fontsize=11)
axes[0].set_ylabel(r"$\log_{10}(\sigma_{\mathrm{re}}\ [\mathrm{km/s}])$", fontsize=11)
axes[0].set_title(r"Faber-Jackson: $\sigma$ vs Magnitude", fontsize=12)
axes[0].legend(frameon=True)
axes[0].grid(True, linestyle="--", alpha=0.5)

x_mass = unified["mstar"]
y_mass = np.log10(unified[velocity_dispersion])
res_mass = linregress(x_mass, y_mass)

axes[1].scatter(x_mass, y_mass, s=12, color="purple", alpha=0.6, edgecolors="none")
x_line_mass = np.linspace(x_mass.min(), x_mass.max(), 100)

sign_mass = "-" if res_mass.intercept < 0 else "+"
axes[1].plot(
    x_line_mass,
    res_mass.slope * x_line_mass + res_mass.intercept,
    color="crimson",
    linewidth=2,
    label=rf"$\log_{{10}}(\sigma) = {res_mass.slope:.2f}\,\log_{{10}}(M_*) {sign_mass} {abs(res_mass.intercept):.2f}$ ($R^2={res_mass.rvalue**2:.2f}$)"
)

axes[1].set_xlabel(r"$\log_{10}(M_* / M_\odot)$", fontsize=11)
axes[1].set_ylabel(r"$\log_{10}(\sigma_{\mathrm{re}}\ [\mathrm{km/s}])$", fontsize=11)
axes[1].set_title(r"Faber-Jackson: $\sigma$ vs Stellar Mass", fontsize=12)
axes[1].legend(frameon=True)
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

print(f"Sample size: {len(unified)} galaxies")
print(f"Magnitude Fit: Slope = {res_mag.slope:.3f} ± {res_mag.stderr:.3f}, Intercept = {res_mag.intercept:.2f}, R² = {res_mag.rvalue**2:.3f}")
print(f"Mass Fit:      Slope = {res_mass.slope:.3f} ± {res_mass.stderr:.3f}, Intercept = {res_mass.intercept:.2f}, R² = {res_mass.rvalue**2:.3f}")