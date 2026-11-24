"""
Animation du ludion (plongeur de Descartes).

Principe :
  1. On appuie sur la bouteille : la pression de l'eau augmente.
  2. L'air emprisonné dans le ludion est comprimé (loi de Boyle-Mariotte :
     p·V = constante), de l'eau entre dans le ludion.
  3. Le ludion (ampoule + eau qu'elle contient) devient plus lourd, alors que
     la poussée d'Archimède, fixée par son volume extérieur, ne change pas.
  4. Quand le poids devient supérieur à la poussée, le ludion coule.
  5. On relâche : l'air se détend et chasse l'eau, le poids redevient
     inférieur à la poussée, le ludion remonte.

Deux façons équivalentes de faire le bilan (réglage SYSTEME) :
  - "ampoule+eau" : système = verre + air + eau intérieure
        poids      = (m + ρ·V_eau_int)·g      -> augmente quand l'eau entre
        Archimède  = ρ·g·(V_verre + V_int)    -> constante
  - "verre+air"   : système = verre + air seulement
        poids      = m·g                      -> constant
        Archimède  = ρ·g·(V_verre + V_air)    -> diminue quand l'air est comprimé
  La force totale Archimède − poids est identique dans les deux cas,
  puisque V_int = V_air + V_eau_int.

La trajectoire est calculée en intégrant  m·a = Archimède − poids − traînée.

Usage :
    python ludion.py           # affiche l'animation
    python ludion.py --save    # enregistre aussi ludion.gif
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter, writers
from matplotlib.patches import Polygon, Rectangle, FancyBboxPatch

# ---------------- Paramètres physiques ----------------
G, RHO, P0 = 9.81, 1000.0, 100e3   # SI
M_VERRE = 2.5e-3                   # masse du ludion (kg)
V_VERRE = 1.0e-6                   # volume du verre (m³)
V_REF = 1.57e-6                    # volume d'air à la pression P0 (m³)
PROF = 0.25                        # hauteur d'eau dans la bouteille (m)
DP_MAX = 8e3                       # surpression due aux mains (Pa)
CD, AIRE = 1.0, 2e-4               # traînée
FPS = 25
EXAGERE = 5                        # amplification visuelle du volume d'air
V_INT = 2.2e-6                     # volume intérieur de l'ampoule (m³)
SYSTEME = "ampoule+eau"            # ou "verre+air" (voir en-tête)

# Géométrie du dessin
Y_FOND, Y_SURF = 0.03, 0.95
ECH = (Y_SURF - Y_FOND) / PROF     # unités de dessin par mètre
LUD_L, LUD_H = 0.10, 0.21         # largeur / hauteur du ludion (dessin)
D_MAX = PROF - LUD_H / ECH         # profondeur max (ludion posé au fond)

def v_air(dp, z):
    return V_REF * P0 / (P0 + dp + RHO * G * z)

def forces(dp, z):
    va = v_air(dp, z)
    return RHO * G * (V_VERRE + va), M_VERRE * G, va

# ---------------- Pré-calcul du scénario ----------------
dt = 1 / FPS / 20
t, z, v = 0.0, 0.0, 0.0
phase, t_evt, dp = "repos", 0.0, 0.0
T_list, Z_list, DP_list, VA_list, B_list, PH_list = [], [], [], [], [], []
DUREE_RAMPE = 0.6
step = 0
while True:
    # Scénario : repos -> appui -> maintien jusqu'au fond (+1 s) -> relâche -> remontée
    if phase == "repos" and t > 1.5:
        phase, t_evt = "appui", t
    if phase == "appui":
        dp = DP_MAX * min(1, (t - t_evt) / DUREE_RAMPE)
        if z >= D_MAX - 1e-9:
            phase, t_evt = "fond", t
    if phase == "fond" and t - t_evt > 1.0:
        phase, t_evt = "relache", t
    if phase == "relache":
        dp = DP_MAX * max(0, 1 - (t - t_evt) / DUREE_RAMPE)
        if z <= 1e-9 and dp == 0:
            phase, t_evt = "fin", t
    if phase == "fin" and t - t_evt > 1.5:
        break

    B, W, va = forces(dp, z)
    m_eff = M_VERRE + 0.5 * RHO * (V_VERRE + va)
    F = W - B - 0.5 * RHO * CD * AIRE * v * abs(v)   # z vers le bas
    v += F / m_eff * dt
    z += v * dt
    if z < 0: z, v = 0.0, 0.0
    if z > D_MAX: z, v = D_MAX, 0.0

    if step % 20 == 0:
        T_list.append(t); Z_list.append(z); DP_list.append(dp)
        VA_list.append(va); B_list.append(B); PH_list.append(phase)
    t += dt
    step += 1

T = np.array(T_list); Z = np.array(Z_list); DP = np.array(DP_list)
VA = np.array(VA_list); BB = np.array(B_list)
N_FRAMES = len(T)
W_N = M_VERRE * G
if SYSTEME == "ampoule+eau":
    ARCH = np.full_like(VA, RHO * G * (V_VERRE + V_INT))   # constante
    POIDS = W_N + RHO * G * (V_INT - VA)                   # augmente
else:
    ARCH = BB                                              # diminue
    POIDS = np.full_like(VA, W_N)                          # constant

# ---------------- Figure ----------------
fig = plt.figure(figsize=(13, 7.5))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15], hspace=0.38, wspace=0.18)
ax = fig.add_subplot(gs[:, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, 1])
fig.suptitle("Le ludion : l'air est comprimé, l'eau entre, le ludion s'alourdit"
             if SYSTEME == "ampoule+eau" else
             "Le ludion : la pression comprime l'air, la poussée d'Archimède diminue",
             fontsize=14)

EAU = "#9fcbee"
XB = 0.22                      # demi-largeur de la bouteille

def contour_bouteille(dp):
    """Parois latérales enfoncées proportionnellement à la pression."""
    creux = 0.045 * dp / DP_MAX
    ys = np.linspace(Y_FOND - 0.03, 1.0, 60)
    prof = np.where((ys > 0.15) & (ys < 0.85),
                    np.sin(np.pi * (ys - 0.15) / 0.7), 0)
    xd = XB - creux * prof
    return ys, xd

eau = Polygon(np.zeros((3, 2)), closed=True, color=EAU, zorder=1)
ax.add_patch(eau)
paroi_g, = ax.plot([], [], color="0.25", lw=3, zorder=4)
paroi_d, = ax.plot([], [], color="0.25", lw=3, zorder=4)
ax.plot([-XB, XB], [Y_FOND - 0.03] * 2, color="0.25", lw=3, zorder=4)
# épaule, goulot, bouchon
ax.plot([-XB, -0.06, -0.06], [1.0, 1.1, 1.18], color="0.25", lw=3, zorder=4)
ax.plot([XB, 0.06, 0.06], [1.0, 1.1, 1.18], color="0.25", lw=3, zorder=4)
ax.add_patch(Rectangle((-0.075, 1.18), 0.15, 0.07, color="#c0392b", zorder=4))
niveau, = ax.plot([], [], color="#3a7bbf", lw=1.2, zorder=2)

# Ludion : tube de verre fermé en haut, ouvert en bas
verre = FancyBboxPatch((0, 0), LUD_L, LUD_H, boxstyle="round,pad=0,rounding_size=0.025",
                       fc="none", ec="0.15", lw=2, zorder=6)
air = Rectangle((0, 0), LUD_L, 0, fc="white", ec="none", zorder=5)
eau_int = Rectangle((0, 0), LUD_L, 0, fc="#6aaee6", ec="none", zorder=5)
for p in (eau_int, air, verre):
    ax.add_patch(p)
lest = Rectangle((0, 0), LUD_L + 0.012, 0.018, color="0.3", zorder=7)  # bague de lestage
ax.add_patch(lest)

# Forces sur le ludion
K_F = 70.0     # amplification de l'écart Archimède − poids sur les flèches
fP = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", lw=2.5, color="#c0392b", mutation_scale=16), zorder=8)
fA = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", lw=2.5, color="#1f5fa8", mutation_scale=16), zorder=8)
tP = ax.text(0, 0, "Poids", color="#c0392b", fontsize=10, zorder=8)
tA = ax.text(0, 0, "Archimède", color="#1f5fa8", fontsize=10, zorder=8)

# Mains (flèches de pression)
mains = [ax.annotate("", xy=(0, 0.5), xytext=(0, 0.5),
                     arrowprops=dict(arrowstyle="simple", color="#e67e22",
                                     mutation_scale=28), zorder=8) for _ in range(2)]
t_main = ax.text(0, 0.30, "", ha="center", color="#e67e22", fontsize=11, zorder=8,
                 fontweight="bold")
etat = ax.text(0, -0.17, "", ha="center", va="top", fontsize=10.5,
               bbox=dict(fc="white", ec="0.75", boxstyle="round,pad=0.5"))

ax.set_xlim(-0.48, 0.48)
ax.set_ylim(-0.42, 1.3)
ax.set_aspect("equal")
ax.axis("off")

# Graphique 1 : pression et volume d'air
l_dp, = ax1.plot([], [], color="#e67e22", lw=2.2, label="surpression Δp")
ax1.set_xlim(0, T[-1]); ax1.set_ylim(0, DP_MAX / 1e3 * 1.15)
ax1.set_ylabel("Δp (kPa)", color="#e67e22")
ax1.set_xlabel("temps (s)")
ax1.grid(alpha=0.3)
ax1b = ax1.twinx()
l_va, = ax1b.plot([], [], color="0.2", lw=2, ls="--", label="volume d'air")
ax1b.set_ylim(VA.min() * 1e6 - 0.02, VA.max() * 1e6 + 0.02)
ax1b.set_ylabel("volume d'air (cm³)")
h1, lb1 = ax1.get_legend_handles_labels(); h2, lb2 = ax1b.get_legend_handles_labels()
ax1.legend(h1 + h2, lb1 + lb2, loc="upper right", frameon=False, fontsize=9)
ax1.set_title("p·V = constante : quand p augmente, V diminue", fontsize=10, color="0.3")

# Graphique 2 : forces
l_P, = ax2.plot([], [], color="#c0392b", lw=2.2, label="poids")
l_B, = ax2.plot([], [], color="#1f5fa8", lw=2.2, label="poussée d'Archimède")
ax2.set_xlim(0, T[-1])
ax2.set_ylim(min(ARCH.min(), POIDS.min()) * 1e3 - 0.4, max(ARCH.max(), POIDS.max()) * 1e3 + 0.4)
ax2.set_ylabel("force (mN)")
ax2.set_xlabel("temps (s)")
ax2.grid(alpha=0.3)
ax2c = ax2.twinx()
l_z, = ax2c.plot([], [], color="0.5", lw=1.5, ls=":", label="profondeur")
ax2c.set_ylim(PROF * 100 * 1.05, -1)
ax2c.set_ylabel("profondeur (cm)")
h1, lb1 = ax2.get_legend_handles_labels(); h2, lb2 = ax2c.get_legend_handles_labels()
ax2.legend(h1 + h2, lb1 + lb2, loc="upper right", frameon=False, fontsize=9)
ax2.set_title("poids > Archimède  →  il coule ;  poids < Archimède  →  il remonte",
              fontsize=10, color="0.3")
fig.subplots_adjust(left=0.02, right=0.93, top=0.91, bottom=0.08)

TEXTES = {
    "repos":   "Au repos : Archimède > poids,\nle ludion flotte.",
    "appui":   ("On appuie : l'air est comprimé, l'eau entre\ndans le ludion, son poids augmente."
                if SYSTEME == "ampoule+eau" else
                "On appuie : la pression augmente,\nl'air est comprimé, l'eau entre dans le ludion."),
    "fond":    "Poids > Archimède :\nle ludion coule et reste au fond.",
    "relache": ("On relâche : l'air se détend et chasse l'eau,\nle poids diminue, il remonte."
                if SYSTEME == "ampoule+eau" else
                "On relâche : l'air se détend,\nArchimède > poids, il remonte."),
    "fin":     "De retour en surface.",
}

# ---------------- Animation ----------------
def maj(f):
    dp, z, va, B, ph = DP[f], Z[f], VA[f], BB[f], PH_list[f]
    if ph == "appui" and B < W_N:
        ph = "fond" if z >= D_MAX - 1e-6 else "appui"

    # Bouteille déformée + niveau d'eau qui monte un peu
    ys, xd = contour_bouteille(dp)
    paroi_g.set_data(-xd, ys); paroi_d.set_data(xd, ys)
    surf = Y_SURF + 0.02 * dp / DP_MAX
    m = ys <= surf
    eau.set_xy(np.vstack([np.column_stack([-xd[m], ys[m]]),
                          np.column_stack([xd[m], ys[m]])[::-1]]))
    niveau.set_data([-xd[m][-1], xd[m][-1]], [surf, surf])

    # Ludion
    y_haut = surf - 0.01 - z * ECH
    x0 = -LUD_L / 2
    y0 = y_haut - LUD_H
    verre.set_x(x0); verre.set_y(y0)
    frac = 0.62 + EXAGERE * (va - V_REF * 0.98) / (V_REF * 1.0)   # part d'air (exagérée)
    frac = np.clip(frac, 0.15, 0.95)
    h_air = frac * (LUD_H - 0.02)
    air.set_xy((x0, y_haut - h_air)); air.set_height(h_air)
    eau_int.set_xy((x0, y0)); eau_int.set_height(LUD_H - h_air)
    lest.set_xy((x0 - 0.006, y0))

    # Forces (flèches)
    xc = 0.0
    A_f, P_f = ARCH[f], POIDS[f]
    if SYSTEME == "ampoule+eau":
        lA, lP = 0.11, max(0.03, 0.11 + K_F * (P_f - A_f))
    else:
        lP, lA = 0.11, max(0.03, 0.11 + K_F * (A_f - P_f))
    fP.xy = (xc, y0 - lP); fP.set_position((xc, y0))
    fA.xy = (xc, y_haut + lA); fA.set_position((xc, y_haut))
    tP.set_position((xc + 0.03, y0 - lP))
    tA.set_position((xc + 0.03, y_haut + lA - 0.03))

    # Mains
    s = dp / DP_MAX
    for i, sg in enumerate((-1, 1)):
        x_out = sg * (XB + 0.2)
        x_in = sg * (XB + 0.2 - 0.14 * s) if s > 0.01 else x_out
        mains[i].xy = (x_in, 0.5); mains[i].set_position((x_out, 0.5))
    t_main.set_text("on appuie" if s > 0.05 else "")
    t_main.set_position((0, 0.32))

    etat.set_text(TEXTES[ph] + f"\nΔp = {dp/1e3:4.1f} kPa   V_air = {va*1e6:5.3f} cm³"
                  f"\neau dans l'ampoule = {(V_INT - va)*1e6:5.3f} cm³")

    l_dp.set_data(T[:f + 1], DP[:f + 1] / 1e3)
    l_va.set_data(T[:f + 1], VA[:f + 1] * 1e6)
    l_B.set_data(T[:f + 1], ARCH[:f + 1] * 1e3)
    l_P.set_data(T[:f + 1], POIDS[:f + 1] * 1e3)
    l_z.set_data(T[:f + 1], Z[:f + 1] * 100)
    return []

anim = FuncAnimation(fig, maj, frames=N_FRAMES, interval=1000 / FPS, blit=False)

if __name__ == "__main__":
    if "--save" in sys.argv:
        w = FFMpegWriter(fps=FPS) if writers.is_available("ffmpeg") else PillowWriter(fps=FPS)
        anim.save("ludion.gif", writer=w, dpi=80)
        print("Animation enregistrée dans ludion.gif")
    plt.show()
