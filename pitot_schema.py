"""
Animation du tube de Pitot, d'après le schéma classique :

    - en A : tube piézométrique (prise latérale) -> pression statique
             le liquide monte jusqu'en A'  (hauteur h_A)
    - en B : tube de Pitot coudé, ouverture face à l'écoulement -> le fluide
             s'arrête en B (point d'arrêt) : pression totale
             le liquide monte jusqu'en B'  (hauteur h_B)

    Bernoulli entre A et B (même altitude, v_B = 0) :
        p_B = p_A + ½ρv²   =>   h_B − h_A = v² / (2g)   =>   v = √(2g·Δh)

La vitesse dans la canalisation varie dans le temps : on voit B' monter et
descendre, alors que A' reste fixe.

Usage :
    python pitot_schema.py           # affiche l'animation
    python pitot_schema.py --save    # enregistre aussi pitot_schema.gif
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle, Polygon

save_animation = True

# ---------------- Physique ----------------
G = 9.81
V_MIN, V_MAX = 0.6, 2.0      # vitesse dans la canalisation (m/s)
H_A = 0.30                   # hauteur piézométrique en A (m), constante
ECH = 2.5                    # unités de dessin par mètre de colonne
N_FRAMES = 220

def v_de_t(frame):
    s = 0.5 * (1 - np.cos(2 * np.pi * frame / N_FRAMES))
    return V_MIN + (V_MAX - V_MIN) * s

# ---------------- Géométrie (unités de dessin) ----------------
X0, X1 = 0.05, 3.4           # extrémités de la canalisation
Y_BAS, Y_HAUT = 0.0, 1.35    # parois inférieure / supérieure
Y_TOP = 3.05                 # sommet des tubes (air libre)
XA_G, XA_D = 0.80, 1.10      # tube piézométrique en A
XA = 0.95
YAB = 0.98                   # altitude des points A et B
XB = 1.90                    # ouverture du tube de Pitot (point B)
YP_BAS, YP_HAUT = 0.85, 1.15 # branche horizontale du tube de Pitot
XP_G, XP_D = 2.10, 2.40      # branche verticale du tube de Pitot

BLEU = "#9fcdf5"
NOIR = "k"
LW = 3

# ---------------- Écoulement autour du tube de Pitot ----------------
# Le nez du tube est modélisé par un ovale de Rankine (source + puits) placé
# sur l'axe de la branche horizontale : point d'arrêt exactement en B.
YC = 0.5 * (YP_BAS + YP_HAUT)
L_OV, H_OV = 0.27, 0.5 * (YP_HAUT - YP_BAS)
XC = XB + L_OV

def psi_unit(x, y, c, k):
    return y + k * (np.arctan2(y, x + c) - np.arctan2(y, x - c))

def demi_epaisseur(c, k):
    lo, hi = 1e-6, 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if psi_unit(0.0, mid, c, k) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

best = None
for c in np.linspace(0.01, L_OV * 0.999, 400):
    k = (L_OV**2 - c**2) / (2 * c)
    err = abs(demi_epaisseur(c, k) - H_OV)
    if best is None or err < best[0]:
        best = (err, c, k)
_, C_OV, K_OV = best

def champ(x, y, U):
    xr, yr = x - XC, y - YC
    r1 = (xr + C_OV)**2 + yr**2 + 1e-9
    r2 = (xr - C_OV)**2 + yr**2 + 1e-9
    k = K_OV * U
    u = U + k * ((xr + C_OV) / r1 - (xr - C_OV) / r2)
    v = k * (yr / r1 - yr / r2)
    return u, v

def dans_ovale(x, y):
    return psi_unit(x - XC, np.abs(y - YC), C_OV, K_OV) < 0

# ---------------- Particules ----------------
rng = np.random.default_rng(3)
N = 550
px = rng.uniform(X0, X1, N)
py = rng.uniform(Y_BAS + 0.03, Y_HAUT - 0.03, N)
ok = ~dans_ovale(px, py)
px, py = px[ok], py[ok]

# ---------------- Figure ----------------
fig, (ax, axg) = plt.subplots(1, 2, figsize=(13, 6.2),
                              gridspec_kw={"width_ratios": [1.25, 1]})
fig.suptitle("Tube de Pitot : h_B − h_A = v² / 2g", fontsize=14)

# Liquide dans la canalisation
ax.add_patch(Rectangle((X0, Y_BAS), X1 - X0, Y_HAUT - Y_BAS, color=BLEU, zorder=0))

# Particules (derrière les tubes : en 3D, le fluide contourne les tubes)
scat = ax.scatter(px, py, c=np.ones_like(px), cmap="viridis", vmin=0, vmax=1.3,
                  s=10, zorder=2)

# Tube de Pitot : liquide au repos à l'intérieur (masque les particules)
interieur = Polygon([(XB, YP_BAS), (XP_D, YP_BAS), (XP_D, Y_HAUT),
                     (XP_G, Y_HAUT), (XP_G, YP_HAUT), (XB, YP_HAUT)],
                    closed=True, color=BLEU, zorder=3)
ax.add_patch(interieur)

# Colonnes de liquide dans les tubes (animées)
col_A = Rectangle((XA_G, Y_HAUT), XA_D - XA_G, 0, color=BLEU, zorder=3)
col_B = Rectangle((XP_G, Y_HAUT), XP_D - XP_G, 0, color=BLEU, zorder=3)
ax.add_patch(col_A); ax.add_patch(col_B)

# Parois
kw = dict(color=NOIR, lw=LW, zorder=5, solid_capstyle="butt")
ax.plot([X0, X1], [Y_BAS, Y_BAS], **kw)
ax.plot([X0, XA_G], [Y_HAUT, Y_HAUT], **kw)
ax.plot([XA_D, XP_G], [Y_HAUT, Y_HAUT], **kw)
ax.plot([XP_D, X1], [Y_HAUT, Y_HAUT], **kw)
ax.plot([XA_G, XA_G], [Y_HAUT, Y_TOP], **kw)
ax.plot([XA_D, XA_D], [Y_HAUT, Y_TOP], **kw)
ax.plot([XB, XP_G, XP_G], [YP_HAUT, YP_HAUT, Y_TOP], **kw)
ax.plot([XB, XP_D, XP_D], [YP_BAS, YP_BAS, Y_TOP], **kw)

ax.text(1.55, Y_TOP + 0.12, "air libre", ha="center", fontsize=14, style="italic")

# Points A et B, ligne de courant AB
ax.plot([X0, XB - 0.02], [YAB, YAB], color=NOIR, lw=0.8, zorder=4)
for x, nom in [(XA, "A"), (XB - 0.1, "B")]:
    ax.plot(x, YAB, "x", color=NOIR, ms=8, mew=1.6, zorder=6)
    ax.text(x, YAB - 0.17, nom, ha="center", va="top", fontsize=13, zorder=6)

# Repères A', B' et cotes h_A, h_B (animés)
mA, = ax.plot([], [], "x", color=NOIR, ms=8, mew=1.6, zorder=6)
mB, = ax.plot([], [], "x", color=NOIR, ms=8, mew=1.6, zorder=6)
tA = ax.text(0, 0, "A′", ha="center", fontsize=13, zorder=6)
tB = ax.text(0, 0, "B′", ha="center", fontsize=13, zorder=6)
pA, = ax.plot([], [], ":", color="0.3", lw=1, zorder=6)
pB, = ax.plot([], [], ":", color="0.3", lw=1, zorder=6)
XHA, XHB = 1.40, 1.80
fA = ax.annotate("", xy=(XHA, Y_HAUT), xytext=(XHA, Y_HAUT),
                 arrowprops=dict(arrowstyle="-|>", lw=1.6, color=NOIR))
fB = ax.annotate("", xy=(XHB, Y_HAUT), xytext=(XHB, Y_HAUT),
                 arrowprops=dict(arrowstyle="-|>", lw=1.6, color=NOIR))
lA = ax.text(XHA - 0.05, 0, "$h_\\mathrm{A}$", fontsize=14, va="center", ha="right")
lB = ax.text(XHB + 0.05, 0, "$h_\\mathrm{B}$", fontsize=14, va="center")

# Flèche de vitesse (longueur proportionnelle à v)
fv = ax.annotate("", xy=(1.1, 0.33), xytext=(1.1, 0.33), zorder=6,
                 arrowprops=dict(arrowstyle="-|>", lw=3, color="#ee4b1e",
                                 mutation_scale=22))
tv = ax.text(1.6, 0.08, "$\\vec{v}$", color="#ee4b1e", fontsize=17, ha="center")

ax.set_xlim(0, 3.45)
ax.set_ylim(-0.55, Y_TOP + 0.4)
ax.set_aspect("equal")
ax.axis("off")

cax = ax.inset_axes([0.55, 0.05, 0.4, 0.022])
cb = fig.colorbar(scat, cax=cax, orientation="horizontal")
cb.set_label("vitesse locale / v", fontsize=9)
cb.ax.tick_params(labelsize=8)

# Graphique Δh = v²/2g
vv = np.linspace(0, V_MAX * 1.05, 200)
axg.plot(vv, 100 * vv**2 / (2 * G), color="#2b6cb0", lw=2.5)
pt, = axg.plot([], [], "o", color="#ee4b1e", ms=10, zorder=5)
gv, = axg.plot([], [], "--", color="#ee4b1e", lw=1)
axg.set_xlim(0, V_MAX * 1.05)
axg.set_ylim(0, 100 * (V_MAX * 1.05)**2 / (2 * G))
axg.set_xlabel("vitesse v dans la canalisation (m/s)")
axg.set_ylabel("Δh = h_B − h_A (cm)")
axg.grid(alpha=0.3)
info = axg.text(0.04, 0.96, "", transform=axg.transAxes, va="top", fontsize=11,
                family="monospace",
                bbox=dict(fc="white", ec="0.8", boxstyle="round,pad=0.5"))
axg.text(0.96, 0.04,
         "Bernoulli de A à B (même altitude, v_B = 0) :\n"
         "p_B = p_A + ½ρv²   ⟹   Δh = v²/2g",
         transform=axg.transAxes, ha="right", va="bottom", fontsize=9.5, color="0.25")
fig.tight_layout(rect=(0, 0, 1, 0.95))

# ---------------- Animation ----------------
DT = 0.02
def maj(frame):
    global px, py
    U = v_de_t(frame)

    # Advection RK2
    u1, v1 = champ(px, py, U)
    xm, ym = px + 0.5 * DT * u1, py + 0.5 * DT * v1
    u2, v2 = champ(xm, ym, U)
    px = px + DT * u2
    py = np.clip(py + DT * v2, Y_BAS + 0.02, Y_HAUT - 0.02)

    sp = np.hypot(*champ(px, py, U))
    bad = (px > X1) | dans_ovale(px, py) | ((sp < 0.04 * U) & (np.abs(px - XB) < 0.15))
    n = bad.sum()
    px[bad] = X0 + rng.uniform(0, 0.05, n)
    py[bad] = rng.uniform(Y_BAS + 0.03, Y_HAUT - 0.03, n)
    scat.set_offsets(np.column_stack([px, py]))
    scat.set_array(np.hypot(*champ(px, py, U)) / U)

    # Niveaux dans les tubes
    dh = U**2 / (2 * G)
    hA_d = ECH * H_A
    hB_d = ECH * (H_A + dh)
    yA, yB = Y_HAUT + hA_d, Y_HAUT + hB_d
    col_A.set_height(hA_d)
    col_B.set_height(hB_d)
    mA.set_data([XA], [yA]);            tA.set_position((XA, yA + 0.08))
    mB.set_data([0.5 * (XP_G + XP_D)], [yB]); tB.set_position((0.5 * (XP_G + XP_D), yB + 0.08))
    pA.set_data([XA + 0.07, XHA + 0.03], [yA, yA])
    pB.set_data([XHB - 0.03, XP_G + 0.08], [yB, yB])
    fA.xy = (XHA, yA); fA.set_position((XHA, Y_HAUT))
    fB.xy = (XHB, yB); fB.set_position((XHB, Y_HAUT))
    lA.set_position((XHA - 0.05, Y_HAUT + 0.5 * hA_d))
    lB.set_position((XHB + 0.05, Y_HAUT + 0.5 * hB_d))

    # Flèche vitesse
    lg = 0.55 * U
    fv.xy = (1.1 + lg, 0.33); fv.set_position((1.1, 0.33))
    tv.set_position((1.1 + lg / 2, 0.06))

    # Graphique
    pt.set_data([U], [100 * dh])
    gv.set_data([U, U, 0], [0, 100 * dh, 100 * dh])
    info.set_text(f"v        = {U:4.2f} m/s\n"
                  f"h_A      = {100 * H_A:5.1f} cm\n"
                  f"h_B      = {100 * (H_A + dh):5.1f} cm\n"
                  f"Δh       = {100 * dh:5.1f} cm\n"
                  f"√(2g·Δh) = {np.sqrt(2 * G * dh):4.2f} m/s")
    return []

anim = FuncAnimation(fig, maj, frames=N_FRAMES, interval=40, blit=False)

#if __name__ == "__main__":
#    if "--save" in sys.argv:
if save_animation == True:
    anim.save("pitot_schema.gif", writer=PillowWriter(fps=25))
    print("Animation enregistrée dans pitot_schema.gif")
    plt.show()
