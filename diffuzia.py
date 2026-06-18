import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# === НАСТРОЙКИ ===
rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 9
rcParams['axes.labelsize'] = 10
rcParams['axes.titlesize'] = 11
rcParams['legend.fontsize'] = 8
rcParams['figure.dpi'] = 120

# === ФИЗИЧЕСКИЕ КОНСТАНТЫ ===
P_0_mmHg = 742
P_0_Pa = P_0_mmHg * 133.322
T_room = 298
T_cold = 269
R = 8.314
mu_air = 28.97e-3
mu_He = 4.00e-3

shell_rho1 = 0.910
shell_rho2 = 0.934

rho_diff_room = (P_0_Pa / (R * T_room)) * (mu_air - mu_He) / 1000
rho_diff_cold = (P_0_Pa / (R * T_cold)) * (mu_air - mu_He) / 1000

# === ПОГРЕШНОСТИ ===
sigma_m = 0.05
sigma_S_rel = 0.03
sigma_rho_ob_rel = 0.0126

# Максимальное допустимое относительное изменение площади (10%)
MAX_S_DEVIATION = 0.10

# ============================================================
# ДАННЫЕ
# ============================================================
group1 = {
    'Медный': {'k': 2.86e-5, 'sigma_k': 2.5e-6, 'm_ob': 2.0, 'm_inf': 27.7, 'm_0': 13.7,
               'S0': 4225, 'k_S': -3.72, 'color': '#CD7F32', 'rho_ob': shell_rho1},
    'Серебро': {'k': 3.31e-5, 'sigma_k': 2.6e-6, 'm_ob': 2.3, 'm_inf': 28.6, 'm_0': 17.7,
                'S0': 3354, 'k_S': -3.31, 'color': '#708090', 'rho_ob': shell_rho1},
    'Факи Чемпион': {'k': 2.93e-5, 'sigma_k': 2.6e-6, 'm_ob': 2.1, 'm_inf': 28.2, 'm_0': 17.0,
                     'S0': 3730, 'k_S': -3.94, 'color': '#0055A4', 'rho_ob': shell_rho2},
    'Апельсин': {'k': 5.69e-5, 'sigma_k': 5.5e-6, 'm_ob': 2.2, 'm_inf': 23.9, 'm_0': 17.2,
                 'S0': 2623, 'k_S': -4.36, 'color': '#FF9F43', 'rho_ob': shell_rho2}
}

group2 = {
    'Малый': {'k': 1.24e-6, 'sigma_k': 7.7e-8, 'm_ob': 4.48, 'm_inf': 12.96, 'm_0': 7.5,
              'S0': 1782, 'k_S': -6.36, 'color': 'red', 'rho_ob': shell_rho2},
    'Средний': {'k': 5.45e-7, 'sigma_k': 8e-9, 'm_ob': 4.38, 'm_inf': 14.22, 'm_0': 3.8,
                'S0': 2557, 'k_S': -8.46, 'color': '#FF9400', 'rho_ob': shell_rho2},
    'Большой': {'k': 3.07e-7, 'sigma_k': 7e-9, 'm_ob': 4.05, 'm_inf': 26.05, 'm_0': 12.6,
                'S0': 2716, 'k_S': -10.70, 'color': 'green', 'rho_ob': shell_rho2},
    'Холодный': {'k': 3.56e-7, 'sigma_k': 1.2e-8, 'm_ob': 4.3, 'm_inf': 16.0, 'm_0': 8.8,
                 'S0': 1655, 'k_S': -7.95, 'color': 'blue', 'rho_ob': shell_rho2}
}


# ============================================================
# ФУНКЦИИ
# ============================================================
def calc_D(t_hours, k, m_ob, m_inf, m_0, S0, k_S, rho_diff, rho_ob_val):
    t_sec = t_hours * 3600
    S_t = S0 + k_S * t_hours
    S_t = np.maximum(S_t, 100)
    delta_m_t = (m_inf - m_0) * np.exp(-k * t_sec)
    D_t = (k * m_ob * delta_m_t) / (S_t ** 2 * rho_diff * rho_ob_val)
    return D_t, S_t


def calc_sigma_D_rel(sigma_k, k, m_ob, delta_m, sigma_rho_diff_rel):
    sigma_delta_m = np.sqrt(sigma_m ** 2 + sigma_m ** 2)
    term_k = (sigma_k / k) ** 2
    term_m_ob = (sigma_m / m_ob) ** 2
    term_delta_m = (sigma_delta_m / delta_m) ** 2
    term_S = (2 * sigma_S_rel) ** 2
    term_rho_ob = sigma_rho_ob_rel ** 2
    term_rho_diff = sigma_rho_diff_rel ** 2
    return np.sqrt(term_k + term_m_ob + term_delta_m + term_S + term_rho_ob + term_rho_diff)


# ============================================================
# ГРАФИК 1: Группа 1
# ============================================================
fig1, ax1 = plt.subplots(figsize=(8, 5))

print("=" * 100)
print("ГРУППА 1: D(t) с критерием |ΔS|/S₀ ≤ 10%")
print("=" * 100)
print(f"{'Шарик':<15} | {'σ_D/D':<10} | {'D_mean':<12} | {'t_valid (ч)':<12} | {'ΔS/S₀':<10}")
print("-" * 100)

for name, p in group1.items():
    delta_m_0 = p['m_inf'] - p['m_0']
    sigma_D_rel = calc_sigma_D_rel(p['sigma_k'], p['k'], p['m_ob'], delta_m_0, 0.02)

    t_hours = np.linspace(0, 15, 1000)
    D_vals, S_vals = calc_D(t_hours, p['k'], p['m_ob'], p['m_inf'], p['m_0'],
                            p['S0'], p['k_S'], rho_diff_room, p['rho_ob'])

    # Критерий: относительное изменение площади ≤ 10%
    S_deviation = np.abs(S_vals - p['S0']) / p['S0']
    valid_mask = S_deviation <= MAX_S_DEVIATION

    if np.any(valid_mask):
        idx_valid = np.where(valid_mask)[0][-1]
    else:
        idx_valid = 0

    t_valid = t_hours[idx_valid]
    D_valid = D_vals[:idx_valid + 1]
    D_mean = np.mean(D_valid)
    S_dev_at_valid = S_deviation[idx_valid]

    print(f"{name:<15} | {sigma_D_rel * 100:<10.2f}% | {D_mean * 1e7:<12.2f} | "
          f"{t_valid:<12.2f} | {S_dev_at_valid * 100:<10.1f}%")

    ax1.plot(t_hours[:idx_valid + 1], D_valid * 1e7, '-', color=p['color'],
             linewidth=2, label=f"{name}")
    ax1.axhline(y=D_mean * 1e7, color=p['color'], linestyle=':', alpha=0.7, linewidth=1)

ax1.set_xlabel('Время, ч')
ax1.set_ylabel(r'$D$, $10^{-7}$ см$^2$/с')
ax1.set_title(f'Группа 1: область применимости (|ΔS|/S₀ ≤ {MAX_S_DEVIATION * 100:.0f}%)')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend()
plt.tight_layout()

# ============================================================
# ГРАФИК 2: Группа 2
# ============================================================
fig2, ax2 = plt.subplots(figsize=(8, 5))

print("\n" + "=" * 100)
print("ГРУППА 2: D(t) с критерием |ΔS|/S₀ ≤ 10%")
print("=" * 100)
print(f"{'Шарик':<15} | {'σ_D/D':<10} | {'D_mean':<12} | {'t_valid (ч)':<12} | {'ΔS/S₀':<10}")
print("-" * 100)

for name, p in group2.items():
    delta_m_0 = p['m_inf'] - p['m_0']
    sigma_rho_diff = np.sqrt(0.027 ** 2 + 0.034 ** 2) if name == 'Холодный' else np.sqrt(0.026 ** 2 + 0.032 ** 2)
    sigma_D_rel = calc_sigma_D_rel(p['sigma_k'], p['k'], p['m_ob'], delta_m_0, sigma_rho_diff)

    rho_val = rho_diff_cold if name == 'Холодный' else rho_diff_room
    t_hours = np.linspace(0, 500, 1000)
    D_vals, S_vals = calc_D(t_hours, p['k'], p['m_ob'], p['m_inf'], p['m_0'],
                            p['S0'], p['k_S'], rho_val, p['rho_ob'])

    S_deviation = np.abs(S_vals - p['S0']) / p['S0']
    valid_mask = S_deviation <= MAX_S_DEVIATION

    if np.any(valid_mask):
        idx_valid = np.where(valid_mask)[0][-1]
    else:
        idx_valid = 0

    t_valid = t_hours[idx_valid]
    D_valid = D_vals[:idx_valid + 1]
    D_mean = np.mean(D_valid)
    S_dev_at_valid = S_deviation[idx_valid]

    print(f"{name:<15} | {sigma_D_rel * 100:<10.2f}% | {D_mean * 1e7:<12.2f} | "
          f"{t_valid:<12.2f} | {S_dev_at_valid * 100:<10.1f}%")

    ax2.plot(t_hours[:idx_valid + 1], D_valid * 1e7, '-', color=p['color'],
             linewidth=2, label=f"{name}")
    ax2.axhline(y=D_mean * 1e7, color=p['color'], linestyle=':', alpha=0.7, linewidth=1)

ax2.set_xlabel('Время, ч')
ax2.set_ylabel(r'$D$, $10^{-7}$ см$^2$/с')
ax2.set_title(f'Группа 2: область применимости (|ΔS|/S₀ ≤ {MAX_S_DEVIATION * 100:.0f}%)')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend()
plt.tight_layout()

plt.show()