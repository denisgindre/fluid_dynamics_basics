"""
Animation de l'effet Venturi dans une canalisation.

- Les particules (colorées selon leur vitesse) accélèrent dans le rétrécissement
  (conservation du débit : v = Q / S).
- Les tubes manométriques montrent la chute de pression au col
  (Bernoulli : p + ½ρv² = constante).
- Le graphique du bas trace la vitesse et la pression le long de la conduite.

Usage :
    python venturi.py            # affiche l'animation
    python venturi.py --save     # enregistre aussi venturi.gif
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

# ---------------- Paramètres physiques ----------------
L = 12.0            # longueur de la conduite
R1 = 1.0            # demi-hauteur de la conduite large
R2 = 0.45           # demi-hauteur au col
X0, X1 = 2.0, 10.0   # début / fin du convergent-divergent
Q = 1.0             # débit (par unité de profondeur) -> v1 = Q / (2 R1)
RHO = 1.0
P0 = 1.6            # pression (manométrique) en entrée

def rayon(x):
    """Profil de la conduite : rétrécissement en cosinus entre X0 et X1."""
    x = np.asarray(x, dtype=float)
    r = np.full_like(x, R1)
    m = (x > X0) & (x < X1)
    s = (x[m] - X0) / (X1 - X0)                 # 0 -> 1
    r[m] = R1 - (R1 - R2) * 0.5 * (1 - np.cos(2 * np.pi * s))
    return r

def vitesse(x):
    return Q / (2 * rayon(x))                    # continuité

def pression(x):
    v1 = Q / (2 * R1)
    return P0 + 0.5 * RHO * (v1**2 - vitesse(x)**2)   # Bernoulli

# ---------------- Particules ----------------
rng = np.random.default_rng(0)
N = 600
px = rng.uniform(0, L, N)
peta = rng.uniform(-0.92, 0.92, N)               # position relative dans la section

v_min, v_max = vitesse(0.0), vitesse((X0 + X1) / 2)

# ---------------- Figure ----------------
fig, (ax, ax2) = plt.subplots(
    2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [2.2, 1]})
fig.suptitle("Effet Venturi : le fluide accélère et la pression chute au rétrécissement",
             fontsize=13)

xs = np.linspace(0, L, 400)
rs = rayon(xs)
ax.fill_between(xs, rs, rs + 0.25, color="0.35")      # paroi haute
ax.fill_between(xs, -rs - 0.25, -rs, color="0.35")    # paroi basse
ax.fill_between(xs, -rs, rs, color="#e8f2fb")         # fluide

# Tubes manométriques
tubes = [1.8, (X0 + X1) / 2, 10.2]
noms = ["entrée", "col", "sortie"]
echelle = 1.6          # hauteur de colonne par unité de pression
largeur = 0.35
colonnes = []
for xt, nom in zip(tubes, noms):
    base = rayon(xt) + 0.25
    ax.add_patch(Rectangle((xt - largeur / 2, base), largeur, 3.2,
                           fill=False, lw=1.5, ec="0.3"))
    col = Rectangle((xt - largeur / 2, base), largeur, 0, color="#2b7bba")
    ax.add_patch(col)
    colonnes.append((col, xt))
    ax.text(xt, -R1 - 0.55, nom, ha="center", fontsize=10)

scat = ax.scatter(px, peta * rayon(px), c=vitesse(px), cmap="plasma",
                  vmin=v_min, vmax=v_max, s=9)
cb = fig.colorbar(scat, ax=ax, pad=0.01)
cb.set_label("vitesse")

ax.set_xlim(0, L)
ax.set_ylim(-R1 - 0.8, R1 + 4.6)
ax.set_aspect("equal")
ax.axis("off")
info = ax.text(L / 2, R1 + 4.5, "", fontsize=10, va="top", ha="center")

# Graphique vitesse / pression
ax2.plot(xs, vitesse(xs), color="#d1495b", lw=2, label="vitesse v")
ax2.set_ylabel("vitesse", color="#d1495b")
ax2.set_xlim(0, L)
ax2.set_xlabel("position le long de la conduite")
ax3 = ax2.twinx()
ax3.plot(xs, pression(xs), color="#2b7bba", lw=2, label="pression p")
ax3.set_ylabel("pression", color="#2b7bba")
ax2.axvspan(X0, X1, color="0.9", zorder=0)
curseur = ax2.axvline(tubes[1], color="0.4", ls="--", lw=1)
fig.legend(loc="lower center", ncol=2, frameon=False)
fig.tight_layout(rect=(0, 0.04, 1, 0.96))

# ---------------- Animation ----------------
DT = 0.06
def maj(frame):
    global px
    # Démarrage progressif des manomètres sur les premières images
    k = min(1.0, frame / 40)
    for col, xt in colonnes:
        col.set_height(k * echelle * float(pression(xt)))

    px = px + vitesse(px) * DT * 2.5
    sortie = px > L
    px[sortie] -= L
    peta[sortie] = rng.uniform(-0.92, 0.92, sortie.sum())

    scat.set_offsets(np.column_stack([px, peta * rayon(px)]))
    scat.set_array(vitesse(px))

    xc = (frame * 0.08) % L                     # curseur qui balaie la conduite
    curseur.set_xdata([xc, xc])
    info.set_text(f"x = {xc:4.1f}   v = {float(vitesse(xc)):.2f}   "
                  f"p = {float(pression(xc)):.2f}\n"
                  f"S·v = constante   |   p + ½ρv² = constante")
    return [scat, curseur, info] + [c for c, _ in colonnes]

anim = FuncAnimation(fig, maj, frames=150, interval=40, blit=False)

#if __name__ == "__main__":
#    if "--save" in sys.argv:
anim.save("venturi2.gif", writer=PillowWriter(fps=25))
print("Animation enregistrée dans venturi.gif")
plt.show()
