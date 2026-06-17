import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

# ============================================================
# НОВЫЕ ДАННЫЕ (время в ЧАСАХ, масса в граммах)
# ============================================================

time_hours = np.array([
    0.0, 0.133, 0.85, 0.9, 1.033, 1.133, 1.183, 1.283, 1.333, 1.4,
    1.45, 1.533, 1.583, 1.667, 1.733, 1.817, 1.9, 2.033, 2.083, 2.15,
    2.25, 2.333, 2.433, 2.5, 2.55, 2.633, 2.7, 2.767, 2.867, 2.933,
    3.0, 3.067, 3.183, 3.267, 3.367, 3.433, 3.5, 3.583, 3.65, 3.783,
    3.85, 4.05, 4.15, 4.583, 4.75, 5.183, 5.233, 5.583, 5.917, 6.2,
    6.717, 15.567, 19.767
])

# Медь
time_maly_ch = np.array([
    0.0, 0.133, 0.85, 0.9, 1.033, 1.133, 1.183, 1.283, 1.333, 1.4,
    1.45, 1.533, 1.583, 1.667, 1.733, 1.817, 1.9, 2.033, 2.083, 2.15,
    2.25, 2.333, 2.433, 2.5, 2.55, 2.633, 2.7, 2.767, 2.867, 2.933,
    3.0, 3.067, 3.183, 3.267, 3.367, 3.433, 3.5, 3.583, 3.65, 3.783,
    3.85, 4.05, 4.15, 4.583, 4.75, 5.183, 5.233, 5.583, 5.917, 6.2,
    6.717
])
mass_maly_g = np.array([
    12.7, 12.9, 13.7, 13.9, 14.0, 14.2, 14.3, 14.4, 14.5, 14.6,
    14.7, 14.8, 14.9, 15.0, 15.1, 15.2, 15.2, 15.3, 15.5, 15.6,
    15.7, 15.8, 15.9, 16.0, 16.1, 16.2, 16.3, 16.4, 16.5, 16.6,
    16.7, 16.8, 16.8, 16.9, 17.0, 17.0, 17.1, 17.2, 17.3, 17.4,
    17.5, 17.6, 17.8, 18.3, 18.3, 18.8, 18.8, 19.1, 19.3, 19.5,
    19.9
])

# Серебро
time_sred_ch = np.array([
    0.0, 0.133, 0.85, 0.9, 1.033, 1.133, 1.183, 1.283, 1.333, 1.4,
    1.45, 1.533, 1.583, 1.667, 1.733, 1.817, 1.9, 2.033, 2.083, 2.15,
    2.25, 2.333, 2.433, 2.5, 2.55, 2.633, 2.7, 2.767, 2.867, 2.933,
    3.0, 3.067, 3.183, 3.267, 3.367, 3.433, 3.5, 3.583, 3.65, 3.783,
    3.85, 4.05, 4.15, 4.583, 4.75, 5.183, 5.233, 5.583, 5.917, 6.2,
    6.717
])
mass_sred_g = np.array([
    16.6, 16.9, 17.7, 17.7, 18.0, 18.0, 18.1, 18.2, 18.3, 18.4,
    18.4, 18.5, 18.7, 18.8, 18.8, 18.9, 19.0, 19.1, 19.1, 19.3,
    19.4, 19.5, 19.6, 19.6, 19.7, 19.7, 19.9, 19.9, 20.1, 20.1,
    20.2, 20.3, 20.4, 20.5, 20.5, 20.6, 20.7, 20.7, 20.8, 20.9,
    20.9, 21.1, 21.2, 21.6, 21.7, 22.0, 22.0, 22.3, 22.6, 22.6,
    23.0
])
# ФАКИ ЧЕМПИОН
time_bolsh_ch = np.array([
    0.0, 0.133, 0.85, 0.9, 1.033, 1.133, 1.183, 1.283, 1.333, 1.4,
    1.45, 1.533, 1.583, 1.667, 1.733, 1.817, 1.9, 2.033, 2.083, 2.15,
    2.25, 2.333, 2.433, 2.5, 2.55, 2.633, 2.7, 2.767, 2.867, 2.933,
    3.0, 3.067, 3.183, 3.267, 3.367, 3.433, 3.5, 3.583, 3.65, 3.783,
    3.85, 4.05, 4.15, 4.583, 4.75, 5.183, 5.233, 5.583, 5.917, 6.2,
    6.717
])
mass_bolsh_g = np.array([
    16.1, 16.3, 17.0, 17.1, 17.1, 17.4, 17.5, 17.5, 17.5, 17.6,
    17.7, 17.7, 17.8, 17.9, 18.0, 18.1, 18.2, 18.3, 18.3, 18.4,
    18.5, 18.6, 18.7, 18.7, 18.8, 18.9, 19.0, 19.1, 19.1, 19.2,
    19.2, 19.3, 19.5, 19.6, 19.7, 19.7, 19.7, 19.8, 19.9, 19.9,
    20.1, 20.2, 20.2, 20.8, 20.8, 21.1, 21.2, 21.4, 21.5, 21.8,
    22.0
])
# Апельсин
time_holod_ch = np.array([
    0.0, 0.133, 0.85, 0.9, 1.033, 1.133, 1.183, 1.283, 1.333, 1.4,
    1.45, 1.533, 1.583, 1.667, 1.733, 1.817, 1.9, 2.033, 2.083, 2.15,
    2.25, 2.333, 2.433, 2.5, 2.55, 2.633, 2.7, 2.767, 2.867, 2.933,
    3.0, 3.067, 3.183, 3.267, 3.367, 3.433, 3.5, 3.583, 3.65, 3.783,
    3.85, 4.05, 4.15, 4.583, 4.75, 5.183, 5.233, 5.583, 5.917
])
mass_holod_g = np.array([
    16.1, 16.4, 17.2, 17.3, 17.5, 17.5, 17.6, 17.7, 17.7, 17.8,
    17.9, 18.0, 18.1, 18.2, 18.3, 18.3, 18.4, 18.5, 18.5, 18.6,
    18.7, 18.9, 18.9, 19.0, 19.1, 19.2, 19.3, 19.3, 19.4, 19.5,
    19.6, 19.7, 19.7, 19.8, 19.8, 19.9, 20.0, 20.0, 20.1, 20.2,
    20.3, 20.4, 20.4, 20.9, 20.9, 21.0, 21.1, 21.3, 21.3
])
# Предельные массы (г)  масса груза + масса оболочки
m_final_g = {
    'Малый': 27.7,
    'Средний': 28.6,
    'Большой': 28.2,
    'Холодный': 23.9
}

# Перевод: часы → секунды, граммы → миллиграммы
time_maly_s = time_maly_ch * 3600
time_sred_s = time_sred_ch * 3600
time_bolsh_s = time_bolsh_ch * 3600
time_holod_s = time_holod_ch * 3600

mass_maly_mg = mass_maly_g * 1000
mass_sred_mg = mass_sred_g * 1000
mass_bolsh_mg = mass_bolsh_g * 1000
mass_holod_mg = mass_holod_g * 1000

# names = ['Малый', 'Средний', 'Большой', 'Холодный']
# data_series = [mass_maly_mg, mass_sred_mg, mass_bolsh_mg, mass_holod_mg]
# time_series = [time_maly_s, time_sred_s, time_bolsh_s, time_holod_s]
# colors = ['#CD7F32', '#708090', '#0055A4', '#FF9F43']

k_maly = np.log( (m_final_g['Малый'] - mass_maly_g[0]) / (m_final_g['Малый'] - mass_maly_g[1:]) ) / time_maly_s[1:]
k_sred = np.log( (m_final_g['Средний'] - mass_sred_g[0]) / (m_final_g['Средний'] - mass_sred_g[1:]) ) / time_sred_s[1:]
k_bolsh = np.log( (m_final_g['Большой'] - mass_bolsh_g[0]) / (m_final_g['Большой'] - mass_bolsh_g[1:]) ) / time_bolsh_s[1:]
k_holod = np.log( (m_final_g['Холодный'] - mass_holod_g[0]) / (m_final_g['Холодный'] - mass_holod_g[1:]) ) / time_holod_s[1:]

# ============================================================
# ГРАФИК логарифмический
# ============================================================
#
# x = np.log((m_final_g['Малый'] - mass_maly_g[0])) - k_maly * time_maly_s[1:]
# y = np.log((m_final_g['Малый'] - mass_maly_g[1:]))
# plt.plot(x, y)

# ============================================================
# ГРАФИКИ k(t)
# ============================================================
# fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# axes = axes.flatten()
#
# # Данные для каждого шарика
# data_k = [
#     (time_maly_ch[1:], k_maly, 'Медь', '#CD7F32'),
#     (time_sred_ch[1:], k_sred, 'Серебро', '#708090'),
#     (time_bolsh_ch[1:], k_bolsh, 'ФАКИ ЧЕМПИОН', '#0055A4'),
#     (time_holod_ch[1:], k_holod, 'Апельсин', '#FF9F43')
# ]
# for i, (time_ch, k_values, name, color) in enumerate(data_k):
#     ax = axes[i]
#
#     # Scatter plot: k vs t
#     ax.scatter(time_ch, k_values * 1e6, c=color, alpha=0.7, s=60,
#                edgecolors='black', linewidth=0.5, zorder=5, label='Эксперимент')
#
#     # Среднее значение k
#     k_mean = np.mean(k_values)
#     ax.axhline(y=k_mean * 1e6, color='black', linestyle='--', linewidth=2,
#                alpha=0.7, label=f'<k> = {k_mean:.3e} с⁻¹')
#
#     # Доверительный интервал (±1σ)
#     k_std = np.std(k_values, ddof=1)
#     ax.axhspan((k_mean - k_std) * 1e6, (k_mean + k_std) * 1e6,
#                color=color, alpha=0.15, label=f'±1σ = {k_std:.3e} с⁻¹')
#
#     ax.set_xlabel('Время, ч', fontsize=12)
#     ax.set_ylabel('k × 10⁶, с⁻¹', fontsize=12)
#     ax.set_title(f'{name} шарик — k(t)', fontsize=13, fontweight='bold')
#     ax.legend(fontsize=9, loc='best')
#     ax.grid(True, alpha=0.3)
#
# plt.tight_layout()
# plt.show()
#
# # ============================================================
# # СВОДНАЯ ТАБЛИЦА
# # ============================================================
# print("\n" + "=" * 70)
# print("СТАТИСТИКА k(t) ДЛЯ КАЖДОГО ШАРИКА")
# print("=" * 70)
# print(f"{'Шарик':<12} {'<k>, с⁻¹':<15} {'σ_k, с⁻¹':<15} {'σ_k/<k>, %':<12} {'N':<5}")
# print("-" * 70)
#
# for time_ch, k_values, name, color in data_k:
#     k_mean = np.mean(k_values)
#     k_std = np.std(k_values, ddof=1)
#     rel_error = (k_std / k_mean) * 100
#     print(f"{name:<12} {k_mean:<15.3e} {k_std:<15.3e} {rel_error:<12.1f} {len(k_values):<5}")
#
# print("=" * 70)

# ============================================================
# y = ln(m_беск - m(t))
# ============================================================
# Палитры: светлая для внутренности точек, темная для обводки и линий
colors_dark = ['#CD7F32', '#708090', '#0055A4', '#FF9F43']
colors_light = ['#C98F5C', '#899CAF', '#3E6DA3', '#FFB36D']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

datasets = [
    (time_maly_ch, mass_maly_g, m_final_g['Малый'], 'Медь', colors_light[0], colors_dark[0]),
    (time_sred_ch, mass_sred_g, m_final_g['Средний'], 'Серебро', colors_light[1], colors_dark[1]),
    (time_bolsh_ch, mass_bolsh_g, m_final_g['Большой'], 'Факи чемпион', colors_light[2], colors_dark[2]),
    (time_holod_ch, mass_holod_g, m_final_g['Холодный'], 'Апельсин', colors_light[3], colors_dark[3])
]

for i, (time, mass, m_inf, name, c_light, c_dark) in enumerate(datasets):
    ax = axes[i]
    y = np.log(m_inf - mass)
    time_s = time * 3600

    # Линейная регрессия
    p, cov = np.polyfit(time_s, y, 1, cov=True)
    k_fit = -p[0]
    b_fit = p[1]

    k_err = np.sqrt(cov[0, 0]) # погрешность k

    # Точки: facecolor — светлый, edgecolor — темный
    ax.scatter(time, y, facecolor=c_light, edgecolor=c_dark, linewidth=1, s=55, zorder=5, label='Эксперимент')

    # Линия фита цвета темной обводки
    ax.plot(time, -k_fit * time_s + b_fit, color=c_dark, linestyle='--', linewidth=1.5, zorder=4,
            label=f'k = ({k_fit:.2e} ± {k_err:.2e}) с⁻¹')

    ax.set_title(f'{name}', fontweight='bold')
    ax.set_xlabel('Время t, ч')
    ax.set_ylabel('ln(m_inf - m(t))')
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=9, loc='best')

plt.tight_layout()
plt.show()

print("\n" + "=" * 50)
print("ИТОГОВЫЕ ЗНАЧЕНИЯ k И ПОГРЕШНОСТЕЙ")
print("=" * 50)
print(f"{'Шарик':<12} {'k ± погрешность, с⁻¹':<30}")
print("-" * 50)

# Повторный лаконичный расчет для вывода текста
for time, mass, m_inf, name, _, _ in datasets:
    y = np.log(m_inf - mass)
    time_s = time * 3600
    p, cov = np.polyfit(time_s, y, deg=1, cov=True)
    k_val = -p[0]
    k_err_val = np.sqrt(cov[0, 0])
    print(f"{name:<12} ({k_val:.2e} ± {k_err_val:.2e}) с⁻¹")

print("=" * 50)