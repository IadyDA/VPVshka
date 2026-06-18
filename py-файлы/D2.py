import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# === НАСТРОЙКИ ГРАФИКА (компактные) ===
rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 9
rcParams['axes.labelsize'] = 10
rcParams['axes.titlesize'] = 11
rcParams['legend.fontsize'] = 8
rcParams['figure.dpi'] = 120

# === ФИЗИЧЕСКИЕ КОНСТАНТЫ ===
P_0_mmHg = 742
P_0_Pa = P_0_mmHg * 133.322
T = 298
T4 = 269
R = 8.314
mu_air = 28.97e-3
mu_He = 4.00e-3
rho_rubber = 0.942 # в г/см3
shell_rho = 0.942 # г/м3

rho_diff = (P_0_Pa / (R * T)) * (mu_air - mu_He) / 1000 # rho - rho_He = delta rho
rho_diff4 = (P_0_Pa / (R * T4)) * (mu_air - mu_He) / 1000

# === Мои расчёты ===
k = {
    'Малый': {'k': 1.24e-06, 'sigma_k': 7.65e-08},
    'Средний': {'k': 5.45e-07, 'sigma_k': 7.79e-09},
    'Большой': {'k': 3.07e-07, 'sigma_k': 6.93e-09},
    'Холодный': {'k': 3.56e-07, 'sigma_k': 1.15e-08}
}
# S = {
#     'Малый': (np.array([0, 179, 1472, 3290]), np.array([1824.3, 1746.7, 1592.7, 1447.1])),
#     'Средний': (np.array([0, 192, 1473, 3290]), np.array([2560.8, 2509.1, 2376.2, 2085.1])),
#     'Больший': (np.array([0, 185, 1476, 3198]), np.array([2750.0, 2650.8, 2451.2, 2147.5])),
#     'Холодный': (np.array([0, 190, 1480, 2614]), np.array([1669.8, 1621.1, 1449.3, 1313.9]))
# }

# S = {
#     'Малый': (np.array([630, 809, 2102, 3920]), np.array([1824.3, 1746.7, 1592.7, 1447.1])),
#     'Средний': (np.array([660, 852, 2133, 3950]), np.array([2560.8, 2509.1, 2376.2, 2085.1])),
#     'Большой': (np.array([0, 185, 1476, 3198]), np.array([2750.0, 2650.8, 2451.2, 2147.5])),
#     'Холодный': (np.array([930, 1120, 2410, 3544]), np.array([1669.8, 1621.1, 1449.3, 1313.9]))
# }

# массы оболочек
shell_masses = {
    'Малый': 4.3, 'Средний': 4.38, 'Большой': 4.48, 'Холодный': 4.05
}
d_shell_masses_g = 0.005 # погрешность

full_masses = {
    'Малый': 12.96,
    'Средний': 14.22,
    'Большой': 26.05,
    'Холодный': 16.0
}

# print('='*70)
# print('Толщина оболочки')
# print('='*70)
# for key in shell_masses:
#     m = shell_masses[key]
#     time_min, S_array = S[key]
#     time_s = time_min * 60
#     delta_array = m / (rho_rubber * S_array)
#     print(f"'{key}':")
#     for t, s, d in zip(time_min, S_array, delta_array):
#         print(f"  t = {t:4d} мин: S = {s:6.1f} см², delta = {d*10000:.2f} мкм ({d:.6f} см)")
# print('='*70)
#
# # Структура данных: { 'Тип': ( Массив_Времени_В_Минутах, Массив_Delta_В_Сантиметрах ) }
# delta = {
#     'Малый': (
#         np.array([630.0, 809.0, 2102.0, 3920.0]),
#         np.array([0.002607, 0.002723, 0.002986, 0.003286])
#     ),
#     'Средний': (
#         np.array([660.0, 852.0, 2133.0, 3950.0]),
#         np.array([0.001816, 0.001853, 0.001957, 0.002230])
#     ),
#     'Большой': (
#         np.array([0.0, 185.0, 1476.0, 3198.0]),
#         np.array([0.001563, 0.001622, 0.001754, 0.002002])
#     ),
#     'Холодный': (
#         np.array([930.0, 1120.0, 2410.0, 3544.0]),
#         np.array([0.002734, 0.002816, 0.003150, 0.003474])
#     )
# }
# а оно нам теперь не надо!


# Малый шарик
time_maly_ch = np.array([0, 1.5, 2, 4, 5.5, 7, 8, 9, 10.5, 11.5, 14.5, 23.5, 27.5, 37.5])
mass_maly_g = np.array([7.5, 7.5, 7.6, 7.6, 7.6, 7.7, 7.8, 7.8, 7.8, 7.9, 7.9, 8.1, 8.2, 8.3])

# Средний шарик
time_sred_ch = np.array([0, 1, 2, 10.5, 12, 12.5, 14.5, 16, 18.5, 19.5, 21, 22, 25, 34, 38, 48, 58, 62.5, 65.5, 70, 72, 81, 89, 94.5, 105.5, 113, 119.5, 128.5, 144, 152])
mass_sred_g = np.array([3.8, 3.8, 3.9, 4.1, 4.2, 4.2, 4.3, 4.3, 4.3, 4.4, 4.4, 4.4, 4.5, 4.6, 4.7, 4.9, 5.0, 5.1, 5.1, 5.2, 5.3, 5.6, 5.6, 5.7, 5.8, 6.0, 6.1, 6.2, 6.4, 6.6])

# Большой шарик
time_bolsh_ch = np.array([0, 1, 1.5, 3.5, 5, 6.5, 7.5, 9, 10, 11, 14, 23, 27, 37, 47, 51.5, 54.5, 59, 61, 70, 78, 83.5, 94.5, 102, 108.5, 117.5, 133, 141])
mass_bolsh_g = np.array([12.6, 12.8, 12.8, 12.9, 12.9, 13.0, 13.0, 13.0, 13.1, 13.0, 13.1, 13.2, 13.2, 13.3, 13.5, 13.6, 13.6, 13.8, 13.8, 13.8, 14.0, 14.0, 14.1, 14.2, 14.3, 14.4, 14.6, 14.8])

# Холодный шарик
time_holod_ch = np.array([0, 0.5, 2, 3, 5.5, 6.5, 9.5, 18.5, 22.5, 32.5, 42.5, 47, 50, 54.5, 56.5, 65.5, 73.5, 79, 89.5, 97, 103.5, 112.5, 128, 135.5])
mass_holod_g = np.array([8.8, 8.9, 9.0, 9.0, 9.0, 9.1, 9.0, 9.1, 9.1, 9.3, 9.3, 9.3, 9.3, 9.4, 9.5, 9.5, 9.5, 9.6, 9.6, 9.8, 9.8, 9.9, 10.0, 10.1])

diff_maly = full_masses['Малый'] - mass_maly_g
diff_sred = full_masses['Средний'] - mass_sred_g
diff_bolsh = full_masses['Большой'] - mass_bolsh_g
diff_holod = full_masses['Холодный'] - mass_holod_g

balls = {
    'Малый': (time_maly_ch*60, diff_maly),
    'Средний': (time_sred_ch*60, diff_sred),
    'Большой': (time_bolsh_ch*60, diff_bolsh),
    'Холодный': (time_holod_ch*60, diff_holod)
}

# считаем S(t)
# выглядит страшно, ну да ладно
S = {
    'Малый': (time_maly_ch, 1824.3 + 6.355164344753832*0.25 -6.355164344753832 * time_maly_ch),
    'Средний': (time_sred_ch, 2560.8 + 8.45824327040272*2 -8.45824327040272 * time_sred_ch),
    'Большой': (time_bolsh_ch, 2750 + 10.702166301591062*0 -10.702166301591062 * time_bolsh_ch),
    'Холодный': (time_holod_ch, 1669.8 + 7.950782257660011*6.5 -7.950782257660011 * time_holod_ch)
}


print('delta_m')
print('='*70)
for name, (time, diff) in balls.items():
    print(f"\n--- {name} шарик ---")
    print(f"Массив разностей: {np.round(diff, 3)}")
print('='*70)

# Расчёт D
# === РАСЧЁТ И ПОСТРОЕНИЕ ГРАФИКА D(t) ===

# Настройки цветов для графиков
colors = {
    'Малый': 'red',
    'Средний': '#FF9400',
    'Большой': 'green',
    'Холодный': 'blue'
}

plt.figure(figsize=(6, 4))

for name in ['Малый', 'Средний', 'Большой', 'Холодный']:
    S_array = S[name][1]  # см2
    k_val = k[name]['k']  # 1/с
    sk = k[name]['sigma_k']

    time_ch = balls[name][0] / 60  # ч
    delta_m = balls[name][1]  # г
    shell_mass = shell_masses[name]  # г

    # Выбор плотности в зависимости от температуры
    current_rho_diff = rho_diff4 if name == 'Холодный' else rho_diff

    # Расчет D
    D_array = (k_val * shell_mass * delta_m) / (S_array ** 2 * current_rho_diff * shell_rho)

    # Расчет погрешности (исправлен перевод d_shell_masses_g в кг)
    sigma_D = D_array * ( (sk / k_val)**2 + (d_shell_masses_g / shell_mass)**2 + ((0.005**2 + 0.005**2)**0.5 / delta_m)**2 +
                         (2 * 0.03)**2 + (0.0126)**2 + (0.02)**2 ) ** 0.5

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

# for name in ['Малый', 'Средний', 'Большой', 'Холодный']:
#     S_array = S[name][1] # см2
#     # delta_array = delta[name][1] * 0.0001 # см
#
#     k_val = k[name]['k'] # 1/с
#     sk = k[name]['sigma_k']
#
#     time_ch = balls[name][0] / 60 # ч
#     delta_m = balls[name][1] * 0.001 # кг
#     shell_mass = shell_masses[name] * 0.001 # кг
#
#     if name!='Холодный':
#         D_array = (k_val * shell_mass * delta_m) / (S_array**2 * rho_diff * shell_rho)
#         sigma_D = D_array * ( (sk/k_val)**2 + (d_shell_masses_g * 1000/shell_mass)**2 + (0.01/delta_m)**2 +
#                               (2*0.03)**2 + (0.0126)**2 + (0.02)**2 )**0.5
#     else:
#         D_array = (k_val * shell_mass * delta_m) / (S_array**2 * rho_diff4 * shell_rho)
#         sigma_D = D_array * ((sk / k_val) ** 2 + (d_shell_masses_g / 1000 / shell_mass) ** 2 + (0.01 / delta_m) ** 2 +
#                              (2 * 0.03) ** 2 + (0.0126) ** 2 + (0.02) ** 2) ** 0.5
#
#     # Вывод краткой статистики в консоль
#     print(f"Шарик '{name}': среднее D = {np.mean(D_array):.2e} см²/с")
#
#     # Строим график (время переводим в часы для оси X)
#     # plt.plot(time_ch, D_array, 'o-', label=name, color=colors[name], markersize=4, linewidth=1.2)
#     plt.plot(time_ch, D_array, 'o', label=name, color=colors[name], markersize=4)
#     # plt.errorbar(time_ch, D_array, yerr=sigma_D, fmt='o', label=name,
#     #              color=colors[name], markersize=4, elinewidth=1, capsize=3)

# Оформление графика согласно вашему компактному стилю
plt.xlabel('Время эксперимента, ч')
plt.ylabel(r'Коэффициент диффузии $D$, $\text{см}^2/\text{с}$')
plt.title('Зависимость коэффициента диффузии $D$ от времени')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='best')

plt.tight_layout()
plt.show()

# # ===================
#
# # === ДАННЫЕ ШАРИКОВ ===
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
#
#
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