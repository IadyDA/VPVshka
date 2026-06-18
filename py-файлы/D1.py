import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.ticker as ticker

# === НАСТРОЙКИ ГРАФИКА (компактные) ===
rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 9
rcParams['axes.labelsize'] = 10
rcParams['axes.titlesize'] = 11
rcParams['legend.fontsize'] = 8
rcParams['figure.dpi'] = 120
rcParams['mathtext.fontset'] = 'dejavusans'

# === ФИЗИЧЕСКИЕ КОНСТАНТЫ ===
P_0_mmHg = 742
P_0_Pa = P_0_mmHg * 133.322
T = 298
R = 8.314
mu_air = 28.97e-3
mu_He = 4.00e-3
shell_rho1 = 0.910 # г/м3
shell_rho2 = 0.934

rho_diff = (P_0_Pa / (R * T)) * (mu_air - mu_He) / 1000 # rho - rho_He = delta rho

# === Мои расчёты ===
k = {
    'Медный': {'k': 4.229e-05, 'sigma_k': 3.1e-07},
    'Серебро': {'k': 7.001e-05, 'sigma_k': 6.7e-07},
    'Факи чемпион': {'k': 5.408e-05, 'sigma_k': 6.0e-07},
    'Апельсин': {'k': 5.69e-05, 'sigma_k': 5.5e-07},
}

# массы оболочек
shell_masses = {
    'Медный': 2.0,
    'Серебро': 2.3,
    'Факи чемпион': 2.1,
    'Апельсин': 2.2,
}
d_shell_masses_g = 0.005 # погрешность

full_masses = {
    'Медный': 27.7,
    'Серебро': 28.6,
    'Факи чемпион': 28.2,
    'Апельсин': 23.9,
}

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

diff_maly = full_masses['Медный'] - mass_maly_g
diff_sred = full_masses['Серебро'] - mass_sred_g
diff_bolsh = full_masses['Факи чемпион'] - mass_bolsh_g
diff_holod = full_masses['Апельсин'] - mass_holod_g

balls = {
    'Медный': (time_maly_ch, diff_maly),
    'Серебро': (time_sred_ch, diff_sred),
    'Факи чемпион': (time_bolsh_ch, diff_bolsh),
    'Апельсин': (time_holod_ch, diff_holod)
}

S = {
    'Медный': (time_maly_ch, 4172.1 - 222.983 * time_maly_ch),
    'Серебро': (time_sred_ch, 3250.8 - 198.594 * time_sred_ch),
    'Факи чемпион': (time_bolsh_ch, 3562.1 - 236.144 * time_bolsh_ch),
    'Апельсин': (time_holod_ch, 2497.7 - 261.805 * time_holod_ch)
}

colors = {
    'Медный': '#CD7F32', 'Серебро': '#708090',
    'Факи чемпион': '#0055A4', 'Апельсин': '#FF9F43'
}

plt.figure(figsize=(6, 4))

for name in ['Медный', 'Серебро', 'Факи чемпион', 'Апельсин']:
    S_array = S[name][1]  # см2
    k_val = k[name]['k']  # 1/с
    sk = k[name]['sigma_k']

    time_ch = balls[name][0]  # ч
    delta_m = balls[name][1]  # г
    shell_mass = shell_masses[name]  # г

    if name == 'Медный' or name == 'Серебро':
        # Расчет D
        D_array = (k_val * shell_mass * delta_m) / (S_array ** 2 * rho_diff * shell_rho1)

        # Расчет погрешности (исправлен перевод d_shell_masses_g в кг)
        sigma_D = D_array * ( (sk / k_val)**2 + (d_shell_masses_g / shell_mass)**2 + ((0.005**2 + 0.005**2)**0.5 / delta_m)**2 +
                             (2 * 0.03)**2 + (0.0126)**2 + (0.02)**2 ) ** 0.5

    else:
        # Расчет D
        D_array = (k_val * shell_mass * delta_m) / (S_array ** 2 * rho_diff * shell_rho2)

        # Расчет погрешности (исправлен перевод d_shell_masses_g в кг)
        sigma_D = D_array * ((sk / k_val) ** 2 + (d_shell_masses_g / shell_mass) ** 2 + (
                    (0.005 ** 2 + 0.005 ** 2) ** 0.5 / delta_m) ** 2 +
                             (2 * 0.03) ** 2 + (0.0126) ** 2 + (0.02) ** 2) ** 0.5

    # Вывод краткой статистики в консоль
    print(f"Шарик '{name}': среднее D = {np.mean(D_array):.2e} см²/с")

    # Строим точки СРАЗУ с усами погрешности по оси Y
    plt.errorbar(
        time_ch, D_array,
        yerr=sigma_D,
        fmt='o',
        label=name,
        color=colors[name],
        markersize=3.5,
        elinewidth=0.8,  # толщина линии погрешности
        capsize=2,  # размер засечек на концах усов
        capthick=0.8  # толщина засечек
    )

# Оформление графика согласно вашему компактному стилю
plt.xlabel('Время эксперимента, ч')
plt.ylabel(r'Коэффициент диффузии $D$, $10^{-7} \text{см}^2/\text{с}$')
plt.title('Зависимость коэффициента диффузии $D$ от времени')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='best')

# # === Убираем экспоненту и форматируем числа ===
# ax = plt.gca()
# # Умножаем реальные значения (например, 1.5e-8) на 10^8, чтобы на оси выводилось просто "1.5"
# ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x * 1e7:.1f}'))


plt.tight_layout()
plt.show()

# ===================

# === ДАННЫЕ ШАРИКОВ ===
# shell_masses = {
#     'Медный': 2.0, 'Серебро': 2.3, 'Факи Чемпион': 2.1, 'Апельсин': 2.2
# }
#
# S_params = {
#     'Медный': {'S_0': 4087.1, 'k_S': -3.716, 'dS_0': 84.4, 'dk_S': 0.335},
#     'Серебро': {'S_0': 3231.2, 'k_S': -3.310, 'dS_0': 65.6, 'dk_S': 0.253},
#     'Факи Чемпион': {'S_0': 3584.1, 'k_S': -3.936, 'dS_0': 72.4, 'dk_S': 0.275},
#     'Апельсин': {'S_0': 2461.8, 'k_S': -4.363, 'dS_0': 47.0, 'dk_S': 0.152}
# }
#
# exp_params = {
#     'Медный': {'m_inf': 27.7, 'delta_m_mg': 13873.22, 'k_s': 0.000029, 'tau': 580.1},
#     'Серебро': {'m_inf': 28.6, 'delta_m_mg': 10904.74, 'k_s': 0.000033, 'tau': 502.3},
#     'Факи Чемпион': {'m_inf': 28.2, 'delta_m_mg': 11209.55, 'k_s': 0.000029, 'tau': 569.2},
#     'Апельсин': {'m_inf': 23.9, 'delta_m_mg': 6763.84, 'k_s': 0.000056, 'tau': 295.6}
# }
#
# colors = {
#     'Медный': '#CD7F32', 'Серебро': '#708090',
#     'Факи Чемпион': '#0055A4', 'Апельсин': '#FF9F43'
# }


# def calculate_D(t_sec, name):
#     m_shell = shell_masses[name]
#     S_p = S_params[name]
#     E_p = exp_params[name]
#
#     t_min = t_sec / 60
#     S = S_p['S_0'] + S_p['k_S'] * (t_min - 37)
#     if S <= 100:  # защита от слишком малой площади
#         return None, None, None, None
#
#     delta = m_shell / (rho_rubber * S)
#     delta_m_g = E_p['delta_m_mg'] / 1000
#     k_s = E_p['k_s']
#     beta = delta_m_g * k_s * np.exp(-k_s * t_sec)
#
#     D = (beta * delta) / (S * rho_diff)
#     return D, S, delta, beta
#
#
# # === РАСЧЁТ ===
# print("=" * 70)
# print("📊 РАСЧЁТ D(t) до характеристического времени τ")
# print("=" * 70)
#
# results = {}
#
# for name in shell_masses.keys():
#     tau_min = exp_params[name]['tau']  # характеристическое время в минутах
#     t_minutes = np.linspace(0, tau_min, 300)  # 300 точек до τ
#     t_seconds = t_minutes * 60
#
#     D_vals, S_vals = [], []
#     for t_sec in t_seconds:
#         D, S, delta, beta = calculate_D(t_sec, name)
#         if D is not None and 0 < D < 1e-5:
#             D_vals.append(D)
#             S_vals.append(S)
#
#     if D_vals:
#         D_vals = np.array(D_vals)
#         results[name] = {
#             't': t_minutes[:len(D_vals)],
#             'D': D_vals,
#             'D_mean': np.mean(D_vals),
#             'D_std': np.std(D_vals),
#             'tau': tau_min
#         }
#         print(f"🎈 {name:15s} τ={tau_min:5.1f} мин | "
#               f"⟨D⟩={np.mean(D_vals) * 1e8:6.2f}±{np.std(D_vals) * 1e8:4.2f}×10⁻⁸ см²/с")
#
# print("=" * 70)
#
# # === ГРАФИКИ (компактные 2×2) ===
# fig, axes = plt.subplots(2, 2, figsize=(11, 7))  # уменьшено с (14,10)
# axes = axes.flatten()
#
# for idx, name in enumerate(shell_masses.keys()):
#     if name in results:
#         ax = axes[idx]
#         data = results[name]
#
#         ax.plot(data['t'], data['D'] * 1e8, '-', color=colors[name], linewidth=1.5)
#         ax.fill_between(data['t'],
#                         np.maximum(0, (data['D'] - data['D_std']) * 1e8),
#                         (data['D'] + data['D_std']) * 1e8,
#                         color=colors[name], alpha=0.25)
#         ax.axhline(data['D_mean'] * 1e8, color=colors[name], linestyle='--',
#                    linewidth=0.8, alpha=0.7)
#
#         ax.set_xlabel('t, мин', fontsize=9)
#         ax.set_ylabel('D, 10⁻⁸ см²/с', fontsize=9)
#         ax.set_title(f"{name}\n(τ={data['tau']:.0f} мин)", fontsize=10, pad=8)
#         ax.grid(True, linestyle=':', alpha=0.5)
#         ax.set_xlim(0, data['tau'])
#         ax.tick_params(axis='both', labelsize=8)
#
# plt.suptitle('Коэффициент диффузии гелия D(t)', fontsize=12, y=0.995)
# plt.tight_layout(rect=[0, 0, 1, 0.97])
# plt.savefig('D_vs_time_compact.png', dpi=200, bbox_inches='tight')
# plt.show()
#
# # === СРАВНЕНИЕ (компактная столбчатая диаграмма) ===
# fig, ax = plt.subplots(figsize=(7, 4))  # уменьшено с (10,6)
#
# names = list(results.keys())
# D_means = [results[n]['D_mean'] * 1e8 for n in names]
# D_stds = [results[n]['D_std'] * 1e8 for n in names]
#
# bars = ax.bar(names, D_means, yerr=D_stds, capsize=5,
#               color=[colors[n] for n in names], alpha=0.85, width=0.6)
#
# ax.set_ylabel('⟨D⟩, 10⁻⁸ см²/с', fontsize=10)
# ax.set_title('Сравнение коэффициентов диффузии', fontsize=11, pad=12)
# ax.grid(axis='y', linestyle=':', alpha=0.4)
# ax.tick_params(axis='both', labelsize=9)
# ax.set_axisbelow(True)
#
# for bar, val in zip(bars, D_means):
#     ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.2,
#             f'{val:.1f}', ha='center', va='bottom', fontsize=9)
#
# plt.tight_layout()
# plt.savefig('D_comparison_compact.png', dpi=200, bbox_inches='tight')
# plt.show()
#
# # === ИТОГОВАЯ ТАБЛИЦА ===
# print("\n📋 РЕЗУЛЬТАТЫ (усреднённые до τ):")
# print(f"{'Шарик':<18} {'⟨D⟩, 10⁻⁸ см²/с':<18} {'σ, 10⁻⁸ см²/с':<18} {'τ, мин':<10}")
# print("-" * 64)
# for name in names:
#     d = results[name]
#     print(f"{name:<18} {d['D_mean'] * 1e8:<18.2f} {d['D_std'] * 1e8:<18.2f} {d['tau']:<10.1f}")
# print("-" * 64)