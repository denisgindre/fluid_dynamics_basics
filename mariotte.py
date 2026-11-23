"""
Vase de Mariotte : la vitesse d'écoulement en C ne dépend que de la hauteur H
du point B (extrémité du tube d'entrée d'air), pas du niveau du liquide.

    En B, des bulles d'air entrent : p_B = p_atm (et v_B ≈ 0).
    En C, le jet sort à l'air libre : p_C = p_atm, z_C = 0.
    Bernoulli de B à C :  p_atm + ρgH = p_atm + ½ρv_C²   =>   v_C = √(2gH)

L'animation place le point B à plusieurs hauteurs successives, pendant que le
vase se vide :
    - v_C change quand on déplace B,
    - v_C reste constante pendant que le niveau du liquide baisse.

Usage :
    python mariotte.py           # affiche l'animation
    python mariotte.py --save    # enregistre aussi mariotte.gif
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter, writers
from matplotlib.patches import Rectangle

save_animation = True

G = 9.81
BLEU = "#a9cdee"
LW = 2

# ---------------- Géométrie (mètres, C en z = 0) ----------------
XT_G, XT_D = 0.90, 2.40        # parois du vase
Z_FOND, Z_HAUT = 0.90, 2.35
W = 0.08                       # diamètre intérieur des tubes
# Tube d'entrée d'air (en U) : branche gauche (A, ouverte en haut) + branche intérieure (B)
XG = 0.30                      # branche gauche
XI = 1.45                      # branche intérieure (dans le vase)
Z_U = 0.45                     # bas du U
Z_TUBE_HAUT = 2.75
# Tube de sortie
XS = 1.64
Z_S_HAUT = 1.02                # entrée du tube de sortie, dans le vase
XA = XG + W / 2

# ---------------- Scénario ----------------
PALIERS = [1.95, 1.70, 1.45, 1.20]   # hauteurs successives de B (m)
N_PAL, N_TRANS = 70, 22
H_list = []
for i, h in enumerate(PALIERS):
    if i > 0:
        h0 = PALIERS[i - 1]
        s = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, N_TRANS)))
        H_list += list(h0 + (h - h0) * s)
    H_list += [h] * N_PAL
H_t = np.array(H_list)
N_FRAMES = len(H_t)
V_t = np.sqrt(2 * G * H_t)
L0 = 2.28
K_VIDANGE = 0.00032                    # baisse du niveau par image et par m/s
L_t = L0 - K_VIDANGE * np.cumsum(V_t)  # niveau libre dans le vase
temps = np.arange(N_FRAMES) / 25       # s (pour le graphique)

# ---------------- Figure ----------------
fig = plt.figure(figsize=(13, 7.5))
gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1], hspace=0.35, wspace=0.2)
ax = fig.add_subplot(gs[:, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, 1])
fig.suptitle("Vase de Mariotte : $v_C = \\sqrt{2gH}$ ne dépend que de la hauteur de B",
             fontsize=14)

# Vase
liq = Rectangle((XT_G, Z_FOND), XT_D - XT_G, 0, color=BLEU, zorder=1)
ax.add_patch(liq)
ax.plot([XT_G, XT_G, XT_D, XT_D, XT_G], [Z_FOND, Z_HAUT, Z_HAUT, Z_FOND, Z_FOND],
        color="k", lw=LW + 0.5, zorder=4)
ax.add_patch(Rectangle((XI - 0.08, Z_FOND - 0.07), XS + W - XI + 0.16, 0.14,
                       fc="0.75", ec="k", hatch="....", lw=1, zorder=5))  # bouchon

# Tube de sortie (rempli, particules)
ax.add_patch(Rectangle((XS, 0), W, Z_S_HAUT, color=BLEU, zorder=2))
ax.plot([XS, XS], [0, Z_S_HAUT], color="k", lw=LW, zorder=6)
ax.plot([XS + W, XS + W], [0, Z_S_HAUT], color="k", lw=LW, zorder=6)

# Tube d'entrée d'air (géométrie mise à jour avec H)
fill_G = Rectangle((XG, Z_U), W, 0, color=BLEU, zorder=2)
fill_U = Rectangle((XG, Z_U), XI + W - XG, W, color=BLEU, zorder=2)
fill_I = Rectangle((XI, Z_U), W, 0, color=BLEU, zorder=2)
for p in (fill_G, fill_U, fill_I):
    ax.add_patch(p)
ext, = ax.plot([], [], color="k", lw=LW, zorder=6)
intl, = ax.plot([], [], color="k", lw=LW, zorder=6)

# Particules dans le tube de sortie et jet
rng = np.random.default_rng(2)
NT = 30
tz = rng.uniform(0, Z_S_HAUT, NT)
tx = XS + rng.uniform(0.015, W - 0.015, NT)
tube_pts = ax.scatter(tx, tz, s=7, color="#1d4f91", zorder=3)
NJ = 40
jz = rng.uniform(-0.45, 0, NJ)
jx = XS + W / 2 + rng.normal(0, 0.008, NJ)
jet_pts = ax.scatter(jx, jz, s=10, color="#3b7dd8", zorder=3)

# Bulles
bulles = []    # [x, z]
bul_sc = ax.scatter([], [], s=30, facecolor="white", edgecolor="0.4", lw=0.8, zorder=3)

# Repères
ax.plot([XS + W / 2], [0], "x", color="k", ms=7, mew=1.5, zorder=7)
ax.text(XS - 0.05, 0.02, "C", ha="right", va="bottom", fontsize=13, zorder=7)
mA, = ax.plot([], [], "x", color="k", ms=7, mew=1.5, zorder=7)
mB, = ax.plot([], [], "x", color="k", ms=7, mew=1.5, zorder=7)
tA = ax.text(0, 0, "A", fontsize=13, ha="right", va="bottom", zorder=7)
tB = ax.text(0, 0, "B", fontsize=13, ha="left", va="bottom", zorder=7)
ligne_H, = ax.plot([], [], ":", color="0.25", lw=1, zorder=7)
XZ = 2.75
ax.annotate("", xy=(XZ, Z_TUBE_HAUT), xytext=(XZ, -0.05),
            arrowprops=dict(arrowstyle="-|>", lw=1.2, color="k"))
ax.text(XZ + 0.05, Z_TUBE_HAUT, "z", fontsize=13, style="italic", va="top")
ax.plot([XS + W, XZ], [0, 0], ":", color="0.25", lw=1)
ax.text(XZ + 0.07, 0, "0", va="center", fontsize=12)
tH = ax.text(XZ + 0.07, 0, "H", va="center", fontsize=13, style="italic")
cote = ax.annotate("", xy=(XZ - 0.12, 1), xytext=(XZ - 0.12, 0),
                   arrowprops=dict(arrowstyle="<->", lw=1, color="#c0392b"))
fv = ax.annotate("", xy=(XS + W + 0.12, -0.1), xytext=(XS + W + 0.12, -0.1),
                 arrowprops=dict(arrowstyle="-|>", lw=3, color="#e8491d",
                                 mutation_scale=20))
tv = ax.text(XS + W + 0.2, -0.2, "$v_C$", color="#e8491d", fontsize=14)
info = ax.text(0.15, -0.35, "", fontsize=11, family="monospace", va="top")

ax.set_xlim(0.1, 3.0)
ax.set_ylim(-0.65, 3.0)
ax.set_aspect("equal")
ax.axis("off")

# Graphique 1 : v_C en fonction de H
Hs = np.linspace(0, 2.2, 200)
ax1.plot(Hs, np.sqrt(2 * G * Hs), color="#2b6cb0", lw=2.5, label="$v_C=\\sqrt{2gH}$")
pt, = ax1.plot([], [], "o", color="#e8491d", ms=10, zorder=5)
gd, = ax1.plot([], [], "--", color="#e8491d", lw=1)
ax1.set_xlim(0, 2.2); ax1.set_ylim(0, 7)
ax1.set_xlabel("hauteur H du point B (m)")
ax1.set_ylabel("$v_C$ (m/s)")
ax1.grid(alpha=0.3)
ax1.legend(loc="lower right", frameon=False)

# Graphique 2 : évolution temporelle
lv, = ax2.plot([], [], color="#e8491d", lw=2.2, label="$v_C$")
ax2.set_xlim(0, temps[-1]); ax2.set_ylim(0, 7)
ax2.set_xlabel("temps (s)")
ax2.set_ylabel("$v_C$ (m/s)", color="#e8491d")
ax2.grid(alpha=0.3)
ax2b = ax2.twinx()
lL, = ax2b.plot([], [], color="#2b6cb0", lw=1.8, label="niveau du liquide")
lH, = ax2b.plot([], [], color="0.3", lw=1.8, ls="--", label="hauteur H de B")
ax2b.set_ylim(0, 2.4)
ax2b.set_ylabel("z (m)")
h1, l1 = ax2.get_legend_handles_labels()
h2, l2 = ax2b.get_legend_handles_labels()
ax2.legend(h1 + h2, l1 + l2, loc="lower left", frameon=False, fontsize=9, ncol=3)
ax2.set_title("le niveau baisse mais $v_C$ ne change que lorsque B bouge",
              fontsize=10, color="0.3")
fig.subplots_adjust(left=0.02, right=0.94, top=0.92, bottom=0.08)

# ---------------- Animation ----------------
VIS = 0.012      # facteur d'échelle visuel pour le mouvement des particules
compteur = [0.0]

def maj(f):
    global tz, jz, bulles
    if f == 0:
        bulles = []
    H, v, L = H_t[f], V_t[f], L_t[f]

    liq.set_height(L - Z_FOND)

    # Tube d'entrée d'air : liquide jusqu'à H dans les deux branches
    fill_G.set_height(H - Z_U)
    fill_I.set_height(H - Z_U)
    ext.set_data([XG, XG, XI + W, XI + W], [Z_TUBE_HAUT, Z_U, Z_U, H])
    intl.set_data([XG + W, XG + W, XI, XI], [Z_TUBE_HAUT, Z_U + W, Z_U + W, H])

    mA.set_data([XA], [H]); tA.set_position((XA - 0.06, H + 0.02))
    mB.set_data([XI + W / 2], [H]); tB.set_position((XI + W / 2 + 0.06, H + 0.02))
    ligne_H.set_data([XA - 0.15, XZ], [H, H])
    tH.set_position((XZ + 0.07, H))
    cote.xy = (XZ - 0.12, H); cote.set_position((XZ - 0.12, 0))

    # Particules dans le tube et jet
    tz = tz - v * VIS
    tz[tz < 0] += Z_S_HAUT
    jz = jz - v * VIS * 1.3
    jz[jz < -0.45] += 0.45
    tube_pts.set_offsets(np.column_stack([tx, tz]))
    jet_pts.set_offsets(np.column_stack([jx, jz]))

    # Bulles : émises en B à un rythme proportionnel au débit
    compteur[0] += v * 0.04
    while compteur[0] >= 1:
        compteur[0] -= 1
        bulles.append([XI + W / 2 + rng.uniform(-0.02, 0.02), H + 0.02])
    for b in bulles:
        b[1] += 0.025
        b[0] += rng.normal(0, 0.006)
    bulles = [b for b in bulles if b[1] < L - 0.02]
    bul_sc.set_offsets(np.array(bulles) if bulles else np.empty((0, 2)))

    # Flèche de vitesse
    lg = 0.08 * v
    fv.xy = (XS + W + 0.12, -0.05 - lg); fv.set_position((XS + W + 0.12, -0.05))

    info.set_text(f"H   = {H:4.2f} m\n"
                  f"v_C = √(2gH) = {v:4.2f} m/s\n"
                  f"niveau = {L:4.2f} m")

    pt.set_data([H], [v])
    gd.set_data([H, H, 0], [0, v, v])
    lv.set_data(temps[:f + 1], V_t[:f + 1])
    lL.set_data(temps[:f + 1], L_t[:f + 1])
    lH.set_data(temps[:f + 1], H_t[:f + 1])
    return []

anim = FuncAnimation(fig, maj, frames=N_FRAMES, interval=40, blit=False)

#if __name__ == "__main__":
    #if "--save" in sys.argv:
if save_animation == True:
    # ffmpeg (si disponible) écrit les images au fil de l'eau : moins de mémoire
    w = FFMpegWriter(fps=25) if writers.is_available("ffmpeg") else PillowWriter(fps=25)
    anim.save("mariotte.gif", writer=w, dpi=80)
    print("Animation enregistrée dans mariotte.gif")
    plt.show()
