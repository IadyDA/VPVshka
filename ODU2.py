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

shell_rho = 0.934  # г/см³ для всех шариков второй группы

rho_diff_room = (P_0_Pa / (R * T_room)) * (mu_air - mu_He) / 1000
rho_diff_cold = (P_0_Pa / (R * T_cold)) * (mu_air - mu_He) / 1000

# ============================================================
# ЭКСПЕРИМЕНТАЛЬНЫЕ ДАННЫЕ (вторая выборка)
# ============================================================

# Малый шарик
time_maly_ch = np.array([0, 1.5, 2, 4, 5.5, 7, 8, 9, 10.5, 11.5, 14.5, 23.5, 27.5, 37.5])
mass_maly_g = np.array([7.5, 7.5, 7.6, 7.6, 7.6, 7.7, 7.8, 7.8, 7.8, 7.9, 7.9, 8.1, 8.2, 8.3])

# Средний шарик
time_sred_ch = np.array(
    [0, 1, 2, 10.5, 12, 12.5, 14.5, 16, 18.5, 19.5, 21, 22, 25, 34, 38, 48, 58, 62.5, 65.5, 70, 72, 81, 89, 94.5, 105.5,
     113, 119.5, 128.5, 144, 152])
mass_sred_g = np.array(
    [3.8, 3.8, 3.9, 4.1, 4.2, 4.2, 4.3, 4.3, 4.3, 4.4, 4.4, 4.4, 4.5, 4.6, 4.7, 4.9, 5.0, 5.1, 5.1, 5.2, 5.3, 5.6, 5.6,
     5.7, 5.8, 6.0, 6.1, 6.2, 6.4, 6.6])

# Большой шарик
time_bolsh_ch = np.array(
    [0, 1, 1.5, 3.5, 5, 6.5, 7.5, 9, 10, 11, 14, 23, 27, 37, 47, 51.5, 54.5, 59, 61, 70, 78, 83.5, 94.5, 102, 108.5,
     117.5, 133, 141])
mass_bolsh_g = np.array(
    [12.6, 12.8, 12.8, 12.9, 12.9, 13.0, 13.0, 13.0, 13.1, 13.0, 13.1, 13.2, 13.2, 13.3, 13.5, 13.6, 13.6, 13.8, 13.8,
     13.8, 14.0, 14.0, 14.1, 14.2, 14.3, 14.4, 14.6, 14.8])

# Холодный шарик
time_holod_ch = np.array(
    [0, 0.5, 2, 3, 5.5, 6.5, 9.5, 18.5, 22.5, 32.5, 42.5, 47, 50, 54.5, 56.5, 65.5, 73.5, 79, 89.5, 97, 103.5, 112.5,
     128, 135.5])
mass_holod_g = np.array(
    [8.8, 8.9, 9.0, 9.0, 9.0, 9.1, 9.0, 9.1, 9.1, 9.3, 9.3, 9.3, 9.3, 9.4, 9.5, 9.5, 9.5, 9.6, 9.6, 9.8, 9.8, 9.9, 10.0,
     10.1])

# ============================================================
# ПАРАМЕТРЫ ШАРИКОВ И S(t)
# ============================================================
group2 = {
    'Малый': {
        'time_ch': time_maly_ch, 'mass_g': mass_maly_g,
        'm_ob': 4.48, 'rho_ob': shell_rho, 'color': 'red',
        'S0': 1825.89, 'kS': -6.355,
        'rho_diff': rho_diff_room
    },
    'Средний': {
        'time_ch': time_sred_ch, 'mass_g': mass_sred_g,
        'm_ob': 4.38, 'rho_ob': shell_rho, 'color': '#FF9400',
        'S0': 2577.72, 'kS': -8.458,
        'rho_diff': rho_diff_room
    },
    'Большой': {
        'time_ch': time_bolsh_ch, 'mass_g': mass_bolsh_g,
        'm_ob': 4.05, 'rho_ob': shell_rho, 'color': 'green',
        'S0': 2750.0, 'kS': -10.702,
        'rho_diff': rho_diff_room
    },
    'Холодный': {
        'time_ch': time_holod_ch, 'mass_g': mass_holod_g,
        'm_ob': 4.3, 'rho_ob': shell_rho, 'color': 'blue',
        'S0': 1721.48, 'kS': -7.951,
        'rho_diff': rho_diff_cold  # ← другая температура!
    }
}


# ============================================================
# ФУНКЦИЯ: прямой расчёт D(t)
# ============================================================
def calc_D_direct(time_ch, mass_g, m_ob, rho_ob, S0, kS, rho_diff):
    """
    D(t) = m'(t) * m_об / (S(t)² * ρ_об * Δρ)
    """
    # Численная производная dm/dt (г/ч)
    dm_dt = np.gradient(mass_g, time_ch)

    # Перевод в г/с
    dm_dt_sec = dm_dt / 3600

    # Площадь S(t)
    S_t = S0 + kS * time_ch
    S_t = np.maximum(S_t, 100)

    # Расчёт D(t)
    D_t = (dm_dt_sec * m_ob) / (S_t ** 2 * rho_ob * rho_diff)

    return D_t, dm_dt, S_t


# ============================================================
# ГРАФИК 2×2
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

print("=" * 80)
print("ПРЯМОЙ РАСЧЁТ D(t) ИЗ ЭКСПЕРИМЕНТАЛЬНЫХ ДАННЫХ (ВТОРАЯ ВЫБОРКА)")
print("=" * 80)
print(f"{'Шарик':<15} | {'<D> (10⁻⁷)':<12} | {'σ_D (10⁻⁷)':<12} | {'N точек':<8}")
print("-" * 80)

for idx, (name, p) in enumerate(group2.items()):
    D_t, dm_dt, S_t = calc_D_direct(
        p['time_ch'], p['mass_g'], p['m_ob'], p['rho_ob'],
        p['S0'], p['kS'], p['rho_diff']
    )

    # Фильтруем физически осмысленные значения
    valid = (D_t > 0) & (D_t < 1e-5) & (S_t > 100)
    D_valid = D_t[valid]
    t_valid = p['time_ch'][valid]

    D_mean = np.mean(D_valid)
    D_std = np.std(D_valid)

    print(f"{name:<15} | {D_mean * 1e7:<12.2f} | {D_std * 1e7:<12.2f} | {len(D_valid):<8}")

    ax = axes[idx]
    ax.errorbar(
        t_valid, D_valid * 1e7,
        yerr=D_std * 1e7,
        fmt='o',
        color=p['color'],
        markersize=3.5,
        elinewidth=0.6,
        capsize=1.5,
        alpha=0.7
    )
    ax.axhline(y=D_mean * 1e7, color=p['color'], linestyle='--',
               linewidth=1, alpha=0.7, label=f'<D> = {D_mean * 1e7:.3f}')

    ax.set_xlabel('Время, ч')
    ax.set_ylabel(r'$D$, $10^{-7}$ см²/с')
    ax.set_title(name)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=7)

plt.suptitle('Прямой расчёт D(t)',
             fontsize=12, y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# ============================================================
# СРАВНЕНИЕ С МОДЕЛЬНЫМ ПОДХОДОМ
# ============================================================
print("\n" + "=" * 80)
print("СРАВНЕНИЕ: ПРЯМОЙ РАСЧЁТ vs ЭКСПОНЕНЦИАЛЬНАЯ МОДЕЛЬ")
print("=" * 80)
print(f"{'Шарик':<15} | {'D прямой (10⁻⁷)':<18} | {'D модель (10⁻⁷)':<18}")
print("-" * 80)

# Модельные значения D из предыдущих расчётов
D_model = {
    'Малый': 0.94,
    'Средний': 0.41,
    'Большой': 0.23,
    'Холодный': 0.35
}

for name, p in group2.items():
    D_t, dm_dt, S_t = calc_D_direct(
        p['time_ch'], p['mass_g'], p['m_ob'], p['rho_ob'],
        p['S0'], p['kS'], p['rho_diff']
    )
    valid = (D_t > 0) & (D_t < 1e-5) & (S_t > 100)
    D_direct = np.mean(D_t[valid])
    print(f"{name:<15} | {D_direct * 1e7:<18.2f} | {D_model[name]:<18.2f}")

print("=" * 80)