import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

os.makedirs("charts", exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#f9f9f9",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

CLASES = ["true", "false", "mixture", "unproven"]
COLORES_CLASE = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]

results = {
    "BERT baseline":                  {"macro": 0.5769, "true": 0.84, "false": 0.71, "mixture": 0.35, "unproven": 0.41},
    "BERT + synonyms":                {"macro": 0.5811, "true": 0.82, "false": 0.67, "mixture": 0.41, "unproven": 0.42},
    "BERT + back-translation":        {"macro": 0.5994, "true": 0.85, "false": 0.71, "mixture": 0.43, "unproven": 0.41},
    "BERT + random swap":             {"macro": 0.5280, "true": 0.82, "false": 0.69, "mixture": 0.32, "unproven": 0.28},
    "BERT + head-tail":               {"macro": 0.6093, "true": 0.89, "false": 0.74, "mixture": 0.50, "unproven": 0.32},
    "BERT + synonyms + HT":           {"macro": 0.6172, "true": 0.87, "false": 0.73, "mixture": 0.47, "unproven": 0.40},
    "BERT + back-transl. + HT":       {"macro": 0.6366, "true": 0.88, "false": 0.75, "mixture": 0.53, "unproven": 0.38},
    "BERT + random swap + HT":        {"macro": 0.6306, "true": 0.89, "false": 0.79, "mixture": 0.50, "unproven": 0.34},
    "BERT + 3 augs + HT":             {"macro": 0.6077, "true": 0.88, "false": 0.76, "mixture": 0.46, "unproven": 0.33},
    "PubMed baseline":                {"macro": 0.5910, "true": 0.86, "false": 0.71, "mixture": 0.41, "unproven": 0.39},
    "PubMed + synonyms":              {"macro": 0.5562, "true": 0.83, "false": 0.68, "mixture": 0.40, "unproven": 0.32},
    "PubMed + back-translation":      {"macro": 0.5929, "true": 0.83, "false": 0.69, "mixture": 0.41, "unproven": 0.44},
    "PubMed + random swap":           {"macro": 0.5437, "true": 0.84, "false": 0.71, "mixture": 0.29, "unproven": 0.34},
    "PubMed + head-tail":             {"macro": 0.6600, "true": 0.87, "false": 0.76, "mixture": 0.53, "unproven": 0.47},
    "PubMed + synonyms + HT":         {"macro": 0.6172, "true": 0.87, "false": 0.73, "mixture": 0.47, "unproven": 0.40},
    "PubMed + back-transl. + HT":     {"macro": 0.6370, "true": 0.88, "false": 0.76, "mixture": 0.54, "unproven": 0.37},
    "PubMed + random swap + HT":      {"macro": 0.6547, "true": 0.89, "false": 0.77, "mixture": 0.52, "unproven": 0.44},
    "PubMed + 3 augs + HT":           {"macro": 0.6047, "true": 0.86, "false": 0.75, "mixture": 0.48, "unproven": 0.33},
    "RoBERTa baseline":               {"macro": 0.6162, "true": 0.86, "false": 0.72, "mixture": 0.44, "unproven": 0.45},
    "RoBERTa + synonyms":             {"macro": 0.6100, "true": 0.85, "false": 0.73, "mixture": 0.43, "unproven": 0.43},
    "RoBERTa + back-translation":     {"macro": 0.6129, "true": 0.86, "false": 0.73, "mixture": 0.43, "unproven": 0.43},
    "RoBERTa + random swap":          {"macro": 0.5427, "true": 0.85, "false": 0.73, "mixture": 0.36, "unproven": 0.23},
    "RoBERTa + head-tail":            {"macro": 0.6600, "true": 0.90, "false": 0.79, "mixture": 0.55, "unproven": 0.43},
    "RoBERTa + synonyms + HT":        {"macro": 0.6400, "true": 0.88, "false": 0.78, "mixture": 0.50, "unproven": 0.40},
    "RoBERTa + back-transl. + HT":    {"macro": 0.6510, "true": 0.89, "false": 0.78, "mixture": 0.51, "unproven": 0.43},
    "RoBERTa + random swap + HT":     {"macro": 0.6557, "true": 0.88, "false": 0.75, "mixture": 0.50, "unproven": 0.49},
    "RoBERTa + 3 augs + HT":          {"macro": 0.6606, "true": 0.88, "false": 0.79, "mixture": 0.53, "unproven": 0.44},
    "Longformer baseline":            {"macro": 0.3427, "true": 0.76, "false": 0.61, "mixture": 0.00, "unproven": 0.00},
    "Longformer + synonyms":          {"macro": 0.3681, "true": 0.77, "false": 0.60, "mixture": 0.11, "unproven": 0.00},
    "Longformer + random swap":       {"macro": 0.3988, "true": 0.78, "false": 0.55, "mixture": 0.27, "unproven": 0.00},
    "Longformer + 3 augs":            {"macro": 0.2991, "true": 0.59, "false": 0.49, "mixture": 0.02, "unproven": 0.10},
}

modelos = list(results.keys())
macros   = [results[m]["macro"] for m in modelos]
matriz   = np.array([[results[m][c] for c in CLASES] for m in modelos])


# 1. MACRO-F1 BY MODEL
fig, ax = plt.subplots(figsize=(12, 14))
colores_modelo = []
for m in modelos:
    if "Longformer" in m: colores_modelo.append("#E53935")
    elif "PubMed"   in m: colores_modelo.append("#FB8C00")
    elif "RoBERTa"  in m: colores_modelo.append("#1E88E5")
    else:                  colores_modelo.append("#43A047")

idx = np.argsort(macros)
bars = ax.barh([modelos[i] for i in idx], [macros[i] for i in idx],
               color=[colores_modelo[i] for i in idx], height=0.65)

for bar, val in zip(bars, [macros[i] for i in idx]):
    ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9)

ax.set_xlim(0, 0.80)
ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.set_xlabel("Macro F1-score", fontsize=11)
ax.set_title("Macro F1 by Model Configuration", fontsize=13, fontweight="bold", pad=12)

leyenda = [
    mpatches.Patch(color="#43A047", label="BERT"),
    mpatches.Patch(color="#FB8C00", label="PubMed-BERT"),
    mpatches.Patch(color="#1E88E5", label="RoBERTa"),
    mpatches.Patch(color="#E53935", label="Longformer"),
]
ax.legend(handles=leyenda, loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("charts/1_macro_f1_by_model.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 1 saved")


# 2. F1 HEATMAP BY MODEL AND CLASS
fig, ax = plt.subplots(figsize=(8, 16))
im = ax.imshow(matriz, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

ax.set_xticks(range(4))
ax.set_xticklabels(CLASES, fontsize=11)
ax.set_yticks(range(len(modelos)))
ax.set_yticklabels(modelos, fontsize=8)
ax.set_title("F1-score by Model and Class", fontsize=13, fontweight="bold", pad=12)

for i in range(len(modelos)):
    for j in range(4):
        val = matriz[i, j]
        color_txt = "white" if val < 0.35 or val > 0.75 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=8, color=color_txt, fontweight="bold")

plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04, label="F1-score")
plt.tight_layout()
plt.savefig("charts/2_heatmap_f1.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 2 saved")


# 3. HEAD-TAIL IMPACT BY CLASS
ht_compare = {
    "BERT":    {"without": results["BERT baseline"],    "with": results["BERT + head-tail"]},
    "PubMed":  {"without": results["PubMed baseline"],  "with": results["PubMed + head-tail"]},
    "RoBERTa": {"without": results["RoBERTa baseline"], "with": results["RoBERTa + head-tail"]},
}

x = np.arange(4)
width = 0.12
fig, ax = plt.subplots(figsize=(11, 6))
colores_base = {"BERT": "#66BB6A", "PubMed": "#FFA726", "RoBERTa": "#42A5F5"}
colores_ht   = {"BERT": "#1B5E20", "PubMed": "#E65100", "RoBERTa": "#0D47A1"}

offsets = [-2, -1, 0, 1, 2, 3]
i = 0
for modelo, datos in ht_compare.items():
    vals_sin = [datos["without"][c] for c in CLASES]
    vals_con = [datos["with"][c] for c in CLASES]
    ax.bar(x + offsets[i]*width, vals_sin, width, label=f"{modelo} w/o HT",
           color=colores_base[modelo], alpha=0.6, edgecolor="white")
    ax.bar(x + offsets[i+1]*width, vals_con, width, label=f"{modelo} + HT",
           color=colores_ht[modelo], edgecolor="white")
    i += 2

ax.set_xticks(x)
ax.set_xticklabels(CLASES, fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_ylabel("F1-score", fontsize=11)
ax.set_title("Impact of Head-Tail Truncation by Model and Class", fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=8, ncol=3, loc="upper right")
ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
plt.tight_layout()
plt.savefig("charts/3_headtail_impact.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 3 saved")


# 4. RADAR CHART
radar_modelos = {
    "BERT + HT":          [results["BERT + head-tail"][c]       for c in CLASES],
    "PubMed + HT":        [results["PubMed + head-tail"][c]     for c in CLASES],
    "RoBERTa + HT":       [results["RoBERTa + head-tail"][c]    for c in CLASES],
    "RoBERTa + 3augs+HT": [results["RoBERTa + 3 augs + HT"][c] for c in CLASES],
    "Longformer baseline":[results["Longformer baseline"][c]    for c in CLASES],
}

N = len(CLASES)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
colores_radar = ["#43A047", "#FB8C00", "#1E88E5", "#5C6BC0", "#E53935"]

for (nombre, vals), color in zip(radar_modelos.items(), colores_radar):
    vals_cierre = vals + vals[:1]
    ax.plot(angles, vals_cierre, "o-", linewidth=2, color=color, label=nombre, markersize=4)
    ax.fill(angles, vals_cierre, alpha=0.07, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(CLASES, fontsize=12)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], fontsize=8, color="gray")
ax.set_title("F1 per Class - Main Models", fontsize=13, fontweight="bold", pad=18)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
plt.tight_layout()
plt.savefig("charts/4_radar_models.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 4 saved")


# 5. DATASET DISTRIBUTION
labels_dist = ["true", "false", "mixture", "unproven"]
original    = [5078, 3001, 1434, 291]
augmentado  = [5078, 3001, 5078, 5078]

x = np.arange(len(labels_dist))
width = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - width/2, original,   width, label="Original",   color="#90CAF9", edgecolor="white")
b2 = ax.bar(x + width/2, augmentado, width, label="Augmented",  color="#1E88E5", edgecolor="white")

for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            str(int(bar.get_height())), ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(labels_dist, fontsize=11)
ax.set_ylabel("Number of samples", fontsize=11)
ax.set_title("Dataset Distribution: Original vs. Augmented", fontsize=13,
             fontweight="bold", pad=12)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("charts/5_dataset_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 5 saved")


# 6. BEST MODEL PER CLASS
best_clases   = ["true", "false", "mixture", "unproven"]
best_f1       = [0.90, 0.79, 0.55, 0.49]
best_modelos  = [
    "RoBERTa\n+ head-tail",
    "3-way tie\nRoBERTa+HT\nBERT+rand+HT\nRoBERTa+3augs+HT",
    "RoBERTa\n+ head-tail",
    "RoBERTa\n+ random swap\n+ head-tail"
]
colores_best = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(best_clases, best_f1, color=colores_best,
              width=0.5, edgecolor="white", linewidth=1.2)

for bar, f1, modelo in zip(bars, best_f1, best_modelos):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"F1 = {f1}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
            modelo, ha="center", va="center", fontsize=8.5,
            color="white", fontweight="bold")

ax.set_ylim(0, 1.05)
ax.set_ylabel("F1-score", fontsize=11)
ax.set_title("Best Model per Class - PubHealth Fact-Checking",
             fontsize=13, fontweight="bold", pad=12)
ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
plt.tight_layout()
plt.savefig("charts/6_best_model_per_class.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 6 saved")


# 7. MACRO-F1 WITH/WITHOUT HEAD-TAIL
grupos = ["BERT", "PubMed-BERT", "RoBERTa"]
sin_ht = [results["BERT baseline"]["macro"],
          results["PubMed baseline"]["macro"],
          results["RoBERTa baseline"]["macro"]]
con_ht = [results["BERT + head-tail"]["macro"],
          results["PubMed + head-tail"]["macro"],
          results["RoBERTa + head-tail"]["macro"]]

x = np.arange(len(grupos))
width = 0.3
fig, ax = plt.subplots(figsize=(8, 5))
b1 = ax.bar(x - width/2, sin_ht, width, label="Without head-tail",
            color=["#A5D6A7","#FFCC80","#90CAF9"], edgecolor="white")
b2 = ax.bar(x + width/2, con_ht, width, label="With head-tail",
            color=["#2E7D32","#E65100","#0D47A1"], edgecolor="white")

for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10)

for i, (s, c) in enumerate(zip(sin_ht, con_ht)):
    mejora = c - s
    ax.annotate(f"+{mejora:.3f}", xy=(x[i]+width/2, c),
                xytext=(x[i]+width/2+0.05, c+0.025),
                fontsize=9, color="darkgreen", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(grupos, fontsize=11)
ax.set_ylim(0.45, 0.75)
ax.set_ylabel("Macro F1-score", fontsize=11)
ax.set_title("Head-Tail Impact on Macro F1 by Model",
             fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("charts/7_headtail_macro_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 7 saved")


# 8. AUGMENTATION BY MODEL (without head-tail)
aug_data = {
    "BERT": {
        "no aug":           0.5769,
        "synonyms":         0.5811,
        "back-translation": 0.5994,
        "random swap":      0.5280,
    },
    "PubMed": {
        "no aug":           0.5910,
        "synonyms":         0.5562,
        "back-translation": 0.5929,
        "random swap":      0.5437,
    },
    "RoBERTa": {
        "no aug":           0.6162,
        "synonyms":         0.6100,
        "back-translation": 0.6129,
        "random swap":      0.5427,
    },
}

categorias = ["no aug", "synonyms", "back-translation", "random swap"]
modelos_aug = list(aug_data.keys())
x = np.arange(len(categorias))
width = 0.25
colores_aug = ["#43A047", "#FB8C00", "#1E88E5"]

fig, ax = plt.subplots(figsize=(10, 6))
for i, (modelo, color) in enumerate(zip(modelos_aug, colores_aug)):
    vals = [aug_data[modelo][c] for c in categorias]
    bars = ax.bar(x + i*width, vals, width, label=modelo,
                  color=color, edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

ax.set_xticks(x + width)
ax.set_xticklabels(categorias, fontsize=11)
ax.set_ylim(0.45, 0.70)
ax.set_ylabel("Macro F1-score", fontsize=11)
ax.set_title("Impact of Augmentation Strategy by Model (without head-tail)",
             fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=10)
ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
plt.tight_layout()
plt.savefig("charts/8_augmentation_by_model.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 8 saved")


# 9. AUGMENTATION WITH AND WITHOUT HEAD-TAIL
aug_ht_data = {
    "BERT": {
        "no aug":                0.5769,
        "synonyms":              0.5811,
        "back-translation":      0.5994,
        "random swap":           0.5280,
        "no aug + HT":           0.6093,
        "synonyms + HT":         0.6172,
        "back-transl. + HT":     0.6366,
        "random swap + HT":      0.6306,
        "3 augs + HT":           0.6077,
    },
    "PubMed": {
        "no aug":                0.5910,
        "synonyms":              0.5562,
        "back-translation":      0.5929,
        "random swap":           0.5437,
        "no aug + HT":           0.6600,
        "synonyms + HT":         0.6172,
        "back-transl. + HT":     0.6370,
        "random swap + HT":      0.6547,
        "3 augs + HT":           0.6047,
    },
    "RoBERTa": {
        "no aug":                0.6162,
        "synonyms":              0.6100,
        "back-translation":      0.6129,
        "random swap":           0.5427,
        "no aug + HT":           0.6600,
        "synonyms + HT":         0.6400,
        "back-transl. + HT":     0.6510,
        "random swap + HT":      0.6557,
        "3 augs + HT":           0.6606,
    },
}

categorias_ht = ["no aug", "synonyms", "back-translation", "random swap",
                 "no aug + HT", "synonyms + HT", "back-transl. + HT",
                 "random swap + HT", "3 augs + HT"]
colores_bert    = ["#A5D6A7","#C8E6C9","#DCEDC8","#F0F4C3",
                   "#1B5E20","#2E7D32","#388E3C","#43A047","#66BB6A"]
colores_pubmed  = ["#FFCCBC","#FFAB91","#FF8A65","#FF7043",
                   "#BF360C","#D84315","#E64A19","#F4511E","#FF5722"]
colores_roberta = ["#BBDEFB","#90CAF9","#64B5F6","#42A5F5",
                   "#0D47A1","#1565C0","#1976D2","#1E88E5","#2196F3"]

all_colors = [colores_bert, colores_pubmed, colores_roberta]
modelos_ht = list(aug_ht_data.keys())
x = np.arange(len(categorias_ht))
width = 0.25

fig, ax = plt.subplots(figsize=(16, 6))
for i, (modelo, colores) in enumerate(zip(modelos_ht, all_colors)):
    vals = [aug_ht_data[modelo][c] for c in categorias_ht]
    bars = ax.bar(x + i*width, vals, width, label=modelo,
                  color=colores, edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=90)

ax.set_xticks(x + width)
ax.set_xticklabels(categorias_ht, fontsize=9, rotation=20, ha="right")
ax.set_ylim(0.45, 0.75)
ax.set_ylabel("Macro F1-score", fontsize=11)
ax.set_title("Augmentation with and without Head-Tail by Model",
             fontsize=13, fontweight="bold", pad=12)
ax.axvline(x=3.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
ax.text(1.5, 0.74, "without head-tail", ha="center", fontsize=9, color="gray")
ax.text(5.5, 0.74, "with head-tail", ha="center", fontsize=9, color="gray")

legend_handles = [
    mpatches.Patch(color="#43A047", label="BERT"),
    mpatches.Patch(color="#F4511E", label="PubMed"),
    mpatches.Patch(color="#1E88E5", label="RoBERTa"),
]
ax.legend(handles=legend_handles, fontsize=10)
ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
plt.tight_layout()
plt.savefig("charts/9_augmentation_headtail_full.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 9 saved")


# 10. LINE CHART - MACRO F1 EVOLUTION
lineas_data = {
    "BERT":       [0.5769, 0.5811, 0.5994, 0.5280, 0.6093, 0.6172, 0.6366, 0.6306, 0.6077],
    "PubMed":     [0.5910, 0.5562, 0.5929, 0.5437, 0.6600, 0.6172, 0.6370, 0.6547, 0.6047],
    "RoBERTa":    [0.6162, 0.6100, 0.6129, 0.5427, 0.6600, 0.6400, 0.6510, 0.6557, 0.6606],
    "Longformer": [0.3427, 0.3681, 0.3414, 0.3988, None,   None,   None,   None,   0.2991],
}

etiquetas = [
    "no aug", "synonyms", "back-transl.", "random swap",
    "HT", "synonyms+HT", "back-transl.+HT", "random+HT", "3augs+HT"
]

colores_linea = {"BERT": "#43A047", "PubMed": "#FB8C00", "RoBERTa": "#1E88E5", "Longformer": "#E53935"}
markers_l = {"BERT": "o", "PubMed": "s", "RoBERTa": "^", "Longformer": "D"}

fig, ax = plt.subplots(figsize=(13, 6))
for modelo, vals in lineas_data.items():
    xs = [etiquetas[i] for i, v in enumerate(vals) if v is not None]
    ys = [v for v in vals if v is not None]
    ax.plot(xs, ys, marker=markers_l[modelo], linewidth=2,
            markersize=7, label=modelo, color=colores_linea[modelo])
    for x_label, val in zip(xs, ys):
        ax.text(etiquetas.index(x_label), val + 0.004, f"{val:.3f}", ha="center",
                va="bottom", fontsize=7.5, color=colores_linea[modelo])

ax.axvline(x=3.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
ax.text(1.5, 0.725, "without head-tail", ha="center", fontsize=9, color="gray")
ax.text(6.0, 0.725, "with head-tail", ha="center", fontsize=9, color="gray")
ax.set_ylim(0.25, 0.74)
ax.set_ylabel("Macro F1-score", fontsize=11)
ax.set_title("Macro F1 Evolution by Model and Augmentation Configuration",
             fontsize=13, fontweight="bold", pad=12)
ax.tick_params(axis="x", rotation=25)
ax.legend(fontsize=10)
ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
plt.tight_layout()
plt.savefig("charts/10_lines_augmentation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 10 saved")


# 11. F1 PER CLASS (4 SUBPLOTS)
por_clase = {
    "BERT": {
        "true":    [0.84, 0.82, 0.85, 0.82, 0.89, 0.87, 0.88, 0.89, 0.88],
        "false":   [0.71, 0.67, 0.71, 0.69, 0.74, 0.73, 0.75, 0.79, 0.76],
        "mixture": [0.35, 0.41, 0.43, 0.32, 0.50, 0.47, 0.53, 0.50, 0.46],
        "unproven":[0.41, 0.42, 0.41, 0.28, 0.32, 0.40, 0.38, 0.34, 0.33],
    },
    "PubMed": {
        "true":    [0.86, 0.83, 0.83, 0.84, 0.87, 0.87, 0.88, 0.89, 0.86],
        "false":   [0.71, 0.68, 0.69, 0.71, 0.76, 0.73, 0.76, 0.77, 0.75],
        "mixture": [0.41, 0.40, 0.41, 0.29, 0.53, 0.47, 0.54, 0.52, 0.48],
        "unproven":[0.39, 0.32, 0.44, 0.34, 0.47, 0.40, 0.37, 0.44, 0.33],
    },
    "RoBERTa": {
        "true":    [0.86, 0.85, 0.86, 0.85, 0.90, 0.88, 0.89, 0.88, 0.88],
        "false":   [0.72, 0.73, 0.73, 0.73, 0.79, 0.78, 0.78, 0.75, 0.79],
        "mixture": [0.44, 0.43, 0.43, 0.36, 0.55, 0.50, 0.51, 0.50, 0.53],
        "unproven":[0.45, 0.43, 0.43, 0.23, 0.43, 0.40, 0.43, 0.49, 0.44],
    },
    "Longformer": {
        "true":    [0.76, 0.77, 0.75, 0.78, None, None, None, None, 0.59],
        "false":   [0.61, 0.60, 0.57, 0.55, None, None, None, None, 0.49],
        "mixture": [0.00, 0.11, 0.01, 0.27, None, None, None, None, 0.02],
        "unproven":[0.00, 0.00, 0.03, 0.00, None, None, None, None, 0.10],
    },
}

etiquetas_c = [
    "no aug", "synonyms", "back-transl.", "random swap",
    "HT", "synonyms+HT", "back-transl.+HT", "random+HT", "3augs+HT"
]

colores_c = {"BERT": "#43A047", "PubMed": "#FB8C00", "RoBERTa": "#1E88E5", "Longformer": "#E53935"}
markers_c = {"BERT": "o", "PubMed": "s", "RoBERTa": "^", "Longformer": "D"}
clases_plot = ["true", "false", "mixture", "unproven"]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for ax, clase in zip(axes, clases_plot):
    for modelo in por_clase:
        vals = por_clase[modelo][clase]
        xs = [etiquetas_c[i] for i, v in enumerate(vals) if v is not None]
        ys = [v for v in vals if v is not None]
        ax.plot(xs, ys, marker=markers_c[modelo], linewidth=2,
                markersize=6, label=modelo, color=colores_c[modelo])
        for x_label, val in zip(xs, ys):
            ax.text(etiquetas_c.index(x_label), val + 0.008,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=6.5, color=colores_c[modelo])

    ax.axvline(x=3.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_title(f"Class: {clase}", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1-score", fontsize=10)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.legend(fontsize=8)
    ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.text(1.5, 1.02, "w/o HT", ha="center", fontsize=8, color="gray")
    ax.text(6.0, 1.02, "with HT", ha="center", fontsize=8, color="gray")

fig.suptitle("F1 Evolution per Class, Model and Augmentation Configuration",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("charts/11_f1_per_class.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 11 saved")

print("\nAll charts saved in 'charts/' folder")
