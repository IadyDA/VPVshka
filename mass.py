import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

# ============================================================
# НОВЫЕ ДАННЫЕ (время в ЧАСАХ, масса в граммах)
# ============================================================

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

# Предельные массы (г)  масса груза + масса оболочки
m_final_g = {
    'Малый': 12.96,
    'Средний': 14.22,
    'Большой': 26.05,
    'Холодный': 16.0
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

names = ['Малый', 'Средний', 'Большой', 'Холодный']
data_series = [mass_maly_mg, mass_sred_mg, mass_bolsh_mg, mass_holod_mg]
time_series = [time_maly_s, time_sred_s, time_bolsh_s, time_holod_s]
colors = ['red', 'orange', 'green', 'blue']

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
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# Данные для каждого шарика
data_k = [
    (time_maly_ch[1:], k_maly, 'Малый', 'red'),
    (time_sred_ch[1:], k_sred, 'Средний', 'orange'),
    (time_bolsh_ch[1:], k_bolsh, 'Большой', 'green'),
    (time_holod_ch[1:], k_holod, 'Холодный', 'blue')
]

for i, (time_ch, k_values, name, color) in enumerate(data_k):
    ax = axes[i]

    # Scatter plot: k vs t
    ax.scatter(time_ch, k_values * 1e6, c=color, alpha=0.7, s=60,
               edgecolors='black', linewidth=0.5, zorder=5, label='Эксперимент')

    # Среднее значение k
    k_mean = np.mean(k_values)
    ax.axhline(y=k_mean * 1e6, color='black', linestyle='--', linewidth=2,
               alpha=0.7, label=f'<k> = {k_mean:.3e} с⁻¹')

    # Доверительный интервал (±1σ)
    k_std = np.std(k_values, ddof=1)
    ax.axhspan((k_mean - k_std) * 1e6, (k_mean + k_std) * 1e6,
               color=color, alpha=0.15, label=f'±1σ = {k_std:.3e} с⁻¹')

    ax.set_xlabel('Время, ч', fontsize=12)
    ax.set_ylabel('k × 10⁶, с⁻¹', fontsize=12)
    ax.set_title(f'{name} шарик — k(t)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# СВОДНАЯ ТАБЛИЦА
# ============================================================
print("\n" + "=" * 70)
print("СТАТИСТИКА k(t) ДЛЯ КАЖДОГО ШАРИКА")
print("=" * 70)
print(f"{'Шарик':<12} {'<k>, с⁻¹':<15} {'σ_k, с⁻¹':<15} {'σ_k/<k>, %':<12} {'N':<5}")
print("-" * 70)

for time_ch, k_values, name, color in data_k:
    k_mean = np.mean(k_values)
    k_std = np.std(k_values, ddof=1)
    rel_error = (k_std / k_mean) * 100
    print(f"{name:<12} {k_mean:<15.3e} {k_std:<15.3e} {rel_error:<12.1f} {len(k_values):<5}")

print("=" * 70)

# ============================================================
# y = ln(m_беск - m(t))
# ============================================================
# Палитры: светлая для внутренности точек, темная для обводки и линий
colors_light = ['#ff6666', '#ffb366', '#66cc66', '#66b2ff']
colors_dark = ['#b30000', '#cc6600', '#006600', '#004c99']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

datasets = [
    (time_maly_ch, mass_maly_g, m_final_g['Малый'], 'Малый', colors_light[0], colors_dark[0]),
    (time_sred_ch, mass_sred_g, m_final_g['Средний'], 'Средний', colors_light[1], colors_dark[1]),
    (time_bolsh_ch, mass_bolsh_g, m_final_g['Большой'], 'Большой', colors_light[2], colors_dark[2]),
    (time_holod_ch, mass_holod_g, m_final_g['Холодный'], 'Холодный', colors_light[3], colors_dark[3])
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

# # ============================================================
# # МОДЕЛЬ И АНАЛИЗ
# # ============================================================
#
# def exponential_fixed_final(t, delta_m, k, m_final):
#     """m(t) = m_final - delta_m * exp(-k*t)"""
#     return m_final - delta_m * np.exp(-k * t)
#
# def calculate_r2(y_true, y_pred):
#     ss_res = np.sum((y_true - y_pred) ** 2)
#     ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
#     return 1 - ss_res / ss_tot
#
# def analyze_exponential(time_s, mass_mg, name, m_final_mg):
#     print(f"\n{'='*70}")
#     print(f"ЭКСПОНЕНЦИАЛЬНЫЙ АНАЛИЗ: {name}")
#     print(f"{'='*70}")
#
#     m0_exp = mass_mg[0]
#     delta_m_initial = m_final_mg - m0_exp
#
#     try:
#         popt, pcov = curve_fit(
#             lambda t, delta_m, k: exponential_fixed_final(t, delta_m, k, m_final_mg),
#             time_s, mass_mg,
#             p0=[delta_m_initial, 0.0001],
#             maxfev=10000
#         )
#
#         delta_m, k = popt
#         m0_fit = m_final_mg - delta_m
#         y_pred = exponential_fixed_final(time_s, delta_m, k, m_final_mg)
#         r2 = calculate_r2(mass_mg, y_pred)
#         perr = np.sqrt(np.diag(pcov))
#
#         print(f"\nПараметры модели:")
#         print(f"  m(t) = m_final - Δm·exp(-k·t)")
#         print(f"\n  m_final (предельная) = {m_final_mg:.1f} мг = {m_final_mg/1000:.2f} г (зафиксировано)")
#         print(f"  Δm                   = {delta_m:.2f} ± {perr[0]:.2f} мг")
#         print(f"  k                    = {k:.8f} ± {perr[1]:.8f} с⁻¹")
#         print(f"\n  m₀ (из модели)       = {m0_fit:.2f} мг = {m0_fit/1000:.2f} г")
#         print(f"  m₀ (эксперимент)     = {m0_exp:.2f} мг = {m0_exp/1000:.2f} г")
#         print(f"  Расхождение          = {abs(m0_fit - m0_exp):.2f} мг ({abs(m0_fit - m0_exp)/m0_exp*100:.1f}%)")
#         print(f"\n  R² = {r2:.6f}")
#
#         print(f"\nПроизводная (скорость изменения массы):")
#         print(f"  dm/dt = Δm·k·exp(-k·t)")
#         print(f"  dm/dt = {delta_m*k:.4f}·exp(-{k:.8f}·t) мг/с")
#         print(f"  Начальная скорость: dm/dt(0) = {delta_m*k:.4f} мг/с")
#
#         tau = 1 / k
#         print(f"\nХарактерное время:")
#         print(f"  τ = 1/k = {tau:.0f} с = {tau/3600:.2f} ч = {tau/60:.1f} мин")
#
#         return {
#             'delta_m': delta_m,
#             'k': k,
#             'm_final': m_final_mg,
#             'r2': r2,
#             'tau': tau,
#             'params': popt
#         }
#
#     except Exception as e:
#         print(f"Ошибка: {e}")
#         return None
#
# # Анализ всех серий
# results = {}
# for mass, time_s, name, color in zip(data_series, time_series, names, colors):
#     m_final_mg = m_final_g[name] * 1000
#     results[name] = analyze_exponential(time_s, mass, name, m_final_mg)
#
# # ============================================================
# # ГРАФИК 1: Экспоненциальное насыщение
# # ============================================================
# fig, axes = plt.subplots(2, 2, figsize=(16, 12))
# axes = axes.flatten()
#
# for i, (mass, time_s, name, color) in enumerate(zip(data_series, time_series, names, colors)):
#     ax = axes[i]
#
#     if results[name] is not None:
#         ax.scatter(time_s/3600, mass, c=color, alpha=0.7, label='Эксперимент', s=50, zorder=5,
#                    edgecolors='black', linewidth=0.5)
#
#         t_fit_ch = np.linspace(time_s.min()/3600, time_s.max()/3600, 500)
#         t_fit_s = t_fit_ch * 3600
#         y_fit = exponential_fixed_final(t_fit_s, *results[name]['params'], results[name]['m_final'])
#         ax.plot(t_fit_ch, y_fit, '-', c=color, linewidth=2.5, label=f'Экспонента: R²={results[name]["r2"]:.4f}')
#
#         # ax.axhline(y=results[name]['m_final'], color=color, linestyle='--', linewidth=2, alpha=0.7,
#         #            label=f'm_∞ = {results[name]["m_final"]/1000:.2f} г')
#
#         # tau_ch = results[name]['tau'] / 3600
#         # ax.axvline(x=tau_ch, color='red', linestyle=':', linewidth=1.5, alpha=0.7,
#         #            label=f'τ = {tau_ch:.2f} ч')
#
#         ax.set_xlabel('Время, ч', fontsize=12)
#         ax.set_ylabel('Масса, мг', fontsize=12)
#         ax.set_title(f'{name} шарик', fontsize=13, fontweight='bold')
#         ax.legend(fontsize=9)
#         ax.grid(True, alpha=0.3)
#
# plt.tight_layout()
# plt.show()
#
# # ============================================================
# # ГРАФИК 2: Производная
# # ============================================================
# fig, axes = plt.subplots(2, 2, figsize=(16, 12))
# axes = axes.flatten()
#
# for i, (mass, time_s, name, color) in enumerate(zip(data_series, time_series, names, colors)):
#     ax = axes[i]
#
#     if results[name] is not None:
#         delta_m = results[name]['delta_m']
#         k = results[name]['k']
#
#         mass_smooth = gaussian_filter1d(mass, sigma=1.5)
#         derivative_exp = np.gradient(mass_smooth, time_s)
#         ax.scatter(time_s/3600, derivative_exp, c=color, alpha=0.6, label='Эксперимент', s=40, zorder=5,
#                    edgecolors='black', linewidth=0.5)
#
#         t_fit_ch = np.linspace(time_s.min()/3600, time_s.max()/3600, 500)
#         t_fit_s = t_fit_ch * 3600
#         derivative_analytic = delta_m * k * np.exp(-k * t_fit_s)
#         ax.plot(t_fit_ch, derivative_analytic, '-', c=color, linewidth=2.5,
#                 label=f'dm/dt = {delta_m*k:.4f}·exp(-{k:.8f}·t)')
#
#         ax.set_xlabel('Время, ч', fontsize=12)
#         ax.set_ylabel('dm/dt, мг/с', fontsize=12)
#         ax.set_title(f'{name} шарик — Производная массы', fontsize=12, fontweight='bold')
#         ax.legend(fontsize=10)
#         ax.grid(True, alpha=0.3)
#
# plt.tight_layout()
# plt.show()
#
# # ============================================================
# # ГРАФИК 3: Полулогарифмический масштаб
# # ============================================================
# fig, axes = plt.subplots(2, 2, figsize=(16, 12))
# axes = axes.flatten()
#
# for i, (mass, time_s, name, color) in enumerate(zip(data_series, time_series, names, colors)):
#     ax = axes[i]
#
#     if results[name] is not None:
#         delta_m = results[name]['delta_m']
#         k = results[name]['k']
#         m_final = results[name]['m_final']
#
#         delta_m_exp = m_final - mass
#         # Фильтруем отрицательные значения (если есть)
#         valid = delta_m_exp > 0
#
#         ax.scatter(time_s[valid]/3600, np.log(delta_m_exp[valid]), c=color, alpha=0.7, s=50,
#                    label='Эксперимент', zorder=5, edgecolors='black', linewidth=0.5)
#
#         t_fit_ch = np.linspace(time_s.min()/3600, time_s.max()/3600, 500)
#         t_fit_s = t_fit_ch * 3600
#         ln_delta_fit = np.log(delta_m) - k * t_fit_s
#         ax.plot(t_fit_ch, ln_delta_fit, '-', c=color, linewidth=2.5,
#                 label=f'ln(Δm) - k·t (R²={results[name]["r2"]:.4f})')
#
#         ax.set_xlabel('Время, ч', fontsize=12)
#         ax.set_ylabel('ln(m∞ - m)', fontsize=12)
#         ax.set_title(f'{name} шарик — Полулогарифмический масштаб', fontsize=12, fontweight='bold')
#         ax.legend(fontsize=10)
#         ax.grid(True, alpha=0.3)
#
# plt.tight_layout()
# plt.show()
#
# # ============================================================
# # СВОДНАЯ ТАБЛИЦА
# # ============================================================
# print("\n" + "="*90)
# print("СВОДНАЯ ТАБЛИЦА ПАРАМЕТРОВ ДИФФУЗИИ ГЕЛИЯ")
# print("="*90)
# print(f"{'Шарик':<12} {'m_∞, г':<10} {'Δm, мг':<12} {'k, с⁻¹':<15} {'τ, ч':<10} {'τ, мин':<10} {'R²':<10} {'v₀, мг/с':<12}")
# print("-"*90)
#
# for name in names:
#     if results[name] is not None:
#         m_final = results[name]['m_final'] / 1000
#         delta_m = results[name]['delta_m']
#         k = results[name]['k']
#         tau = results[name]['tau']
#         r2 = results[name]['r2']
#         v0 = delta_m * k
#         print(f"{name:<12} {m_final:<10.2f} {delta_m:<12.2f} {k:<15.6f} {tau/3600:<10.2f} {tau/60:<10.1f} {r2:<10.6f} {v0:<12.4f}")
#
# print("\n" + "="*90)
# print("ФИЗИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ:")
# print("="*90)
# print("Процесс: СДУВАНИЕ шариков с гелием")
# print("Модель: m(t) = m_∞ - Δm·exp(-k·t)")
# print("\nПараметры:")
# print("  m_∞ - масса оболочки + груза (когда весь гелий выйдет)")
# print("  Δm - изменение массы (связано с начальной подъемной силой)")
# print("  k - константа скорости диффузии гелия через оболочку")
# print("  τ = 1/k - характерное время (за которое выходит 63% гелия)")
# print("\nЧем БОЛЬШЕ k, тем БЫСТРЕЕ сдувается шарик!")
# print("="*90)
#
# # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
# # РАСЧЁТ И ОТОБРАЖЕНИЕ ПОГРЕШНОСТЕЙ
# # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
#
# instrumental_error_g = 0.05  # г
# instrumental_error_mg = instrumental_error_g * 1000  # 50 мг
#
#
# def calculate_approximation_errors(time_s, mass_mg, params, m_final_mg):
#     """Расчёт погрешностей аппроксимации"""
#     delta_m, k = params
#
#     # Предсказанные значения
#     mass_pred = exponential_fixed_final(time_s, delta_m, k, m_final_mg)
#
#     # Остатки
#     residuals = mass_mg - mass_pred
#
#     # RMSE
#     rmse = np.sqrt(np.mean(residuals ** 2))
#
#     # Стандартная ошибка параметров
#     perr = np.sqrt(np.diag(np.linalg.inv(
#         np.array([[np.sum(np.exp(-2 * k * time_s)), np.sum(delta_m * time_s * np.exp(-2 * k * time_s))],
#                   [np.sum(delta_m * time_s * np.exp(-2 * k * time_s)),
#                    np.sum((delta_m * time_s) ** 2 * np.exp(-2 * k * time_s))]]
#                  )) * rmse ** 2))
#
#     return rmse, residuals
#
# # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
# # ГРАФИК 1 С ПОГРЕШНОСТЯМИ: Экспоненциальное насыщение
# # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
# fig, axes = plt.subplots(2, 2, figsize=(16, 12))
# axes = axes.flatten()
#
# for i, (mass, time_s, name, color) in enumerate(zip(data_series, time_series, names, colors)):
#     ax = axes[i]
#
#     if results[name] is not None:
#         m_final_mg = m_final_g[name] * 1000
#
#         # 1. Экспериментальные точки с ПОГРЕШНОСТЯМИ
#         ax.errorbar(time_s / 3600, mass,
#                     yerr=instrumental_error_mg,  # Приборная погрешность
#                     fmt='o',
#                     c=color,
#                     alpha=0.7,
#                     label='Эксперимент',
#                     capsize=4,
#                     capthick=1.5,
#                     ecolor=color,
#                     elinewidth=1.5,
#                     markersize=6,
#                     zorder=5)
#
#         # 2. Аппроксимирующая кривая
#         t_fit_ch = np.linspace(time_s.min() / 3600, time_s.max() / 3600, 500)
#         t_fit_s = t_fit_ch * 3600
#         y_fit = exponential_fixed_final(t_fit_s, *results[name]['params'], results[name]['m_final'])
#         ax.plot(t_fit_ch, y_fit, '-', c=color, linewidth=2.5,
#                 label=f'Аппроксимация\nR²={results[name]["r2"]:.4f}')
#
#         # 3. Доверительный интервал аппроксимации (±RMSE)
#         rmse, _ = calculate_approximation_errors(time_s, mass, results[name]['params'], m_final_mg)
#         ax.fill_between(t_fit_ch, y_fit - rmse, y_fit + rmse,
#                         alpha=0.3, color=color,
#                         label=f'±σ (RMSE={rmse:.1f} мг)')
#
#         # 4. Предельная масса
#         ax.axhline(y=m_final_mg, color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
#                    label=f'm_∞ = {m_final_mg / 1000:.2f} г')
#
#         ax.set_xlabel('Время, ч', fontsize=12)
#         ax.set_ylabel('Масса, мг', fontsize=12)
#         ax.set_title(f'{name} шарик\n(погрешность весов: ±{instrumental_error_g} г)',
#                      fontsize=12, fontweight='bold')
#         ax.legend(fontsize=9, loc='lower right')
#         ax.grid(True, alpha=0.3)
#
# plt.tight_layout()
# plt.show()
#
# # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
# # ГРАФИК ПОГРЕШНОСТЕЙ: Остатки(разности)
# # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
# fig, axes = plt.subplots(2, 2, figsize=(16, 12))
# axes = axes.flatten()
#
# for i, (mass, time_s, name, color) in enumerate(zip(data_series, time_series, names, colors)):
#     ax = axes[i]
#
#     if results[name] is not None:
#         m_final_mg = m_final_g[name] * 1000
#
#         # Расчёт остатков
#         mass_pred = exponential_fixed_final(time_s, *results[name]['params'], m_final_mg)
#         residuals = mass - mass_pred
#
#         rmse = np.sqrt(np.mean(residuals ** 2))
#         max_error = np.max(np.abs(residuals))
#
#         # График остатков
#         ax.scatter(time_s / 3600, residuals, c=color, alpha=0.7, s=50,
#                    edgecolors='black', linewidth=0.5, zorder=5)
#         ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
#         ax.axhline(y=rmse, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
#                    label=f'+RMSE = +{rmse:.1f} мг')
#         ax.axhline(y=-rmse, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
#                    label=f'-RMSE = -{rmse:.1f} мг')
#         ax.axhline(y=instrumental_error_mg, color='orange', linestyle=':', linewidth=1.5, alpha=0.7,
#                    label=f'Приборная = +{instrumental_error_mg:.1f} мг')
#         ax.axhline(y=-instrumental_error_mg, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
#
#         ax.set_xlabel('Время, ч', fontsize=12)
#         ax.set_ylabel('Остаток, мг', fontsize=12)
#         ax.set_title(f'{name} шарик\nRMSE={rmse:.2f} мг, Max={max_error:.2f} мг',
#                      fontsize=12, fontweight='bold')
#         ax.legend(fontsize=8)
#         ax.grid(True, alpha=0.3)
#
# plt.tight_layout()
# plt.show()
#
# # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
# # ТАБЛИЦА ПОГРЕШНОСТЕЙ
# # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
# print("\n" + "=" * 90)
# print("ТАБЛИЦА ПОГРЕШНОСТЕЙ")
# print("=" * 90)
# print(f"{'Шарик':<12} {'Приборная, мг':<15} {'RMSE, мг':<12} {'Max, мг':<12} {'Суммарная, мг':<15}")
# print("-" * 90)
#
# for i, (mass, time_s, name, color) in enumerate(zip(data_series, time_series, names, colors)):
#     if results[name] is not None:
#         m_final_mg = m_final_g[name] * 1000
#         rmse, residuals = calculate_approximation_errors(time_s, mass, results[name]['params'], m_final_mg)
#         max_error = np.max(np.abs(residuals))
#
#         # Суммарная погрешность (квадратичное сложение)
#         total_error = np.sqrt(instrumental_error_mg ** 2 + rmse ** 2)
#
#         print(f"{name:<12} {instrumental_error_mg:<15.1f} {rmse:<12.2f} {max_error:<12.2f} {total_error:<15.2f}")
#
# print("=" * 90)
# print(f"\nПриборная погрешность весов: ±{instrumental_error_g} г = ±{instrumental_error_mg:.1f} мг")
# print("Суммарная погрешность = √(приборная² + RMSE²)")
# print("=" * 90)