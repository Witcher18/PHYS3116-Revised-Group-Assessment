import pandas as pd
from matplotlib import pyplot as plt

gama = pd.read_csv("data/InputCatGAMADR3.csv")
clusters = pd.read_csv("data/samiDR3InputCatClusters.csv")
skinedata = pd.read_csv("data/samiDR3StelKin.csv")
visualdata = pd.read_csv("data/samiDR3VisualMorphology.csv")

photodata = pd.concat([gama, clusters], ignore_index=True).drop_duplicates(subset=["catid"])

sigma = "sigma_re"

unified = (
    photodata
    .merge(skinedata, on="catid")
    .merge(visualdata, on="catid")
    .loc[lambda df: df["bad_class"].isin([0, 5, 8])]
    .loc[lambda df: df[sigma].notna()]
    .loc[lambda df: df[sigma + "_err"] < df[sigma] * 0.1 + 25]
    .loc[lambda df: df["m_r"].notna()]
)

fig, ax = plt.subplots()
ax.hist(unified["m_r"], bins=40, density=True, histtype="step", color="blue")

ax.set_xlabel("$M_r$ (mags)")
ax.set_ylabel("Density")
plt.title("Distribution of $M_r$ (SAMI DR3)")
plt.grid(True)

plt.show()