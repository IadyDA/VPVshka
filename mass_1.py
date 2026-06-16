import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy import stats

# Данные (время в минутах, масса в граммах)
time_min = np.array(
    [0, 3, 11, 17, 20, 26, 29, 33, 36, 41, 44, 49, 53, 58, 63, 71, 74, 78, 84, 89, 95, 99, 102, 107, 111, 115, 121, 125,
     129, 133, 140, 145, 151, 155, 159, 164, 168, 176, 180, 192, 198])

medny_g = np.array(
    [13.7, 13.9, 14.0, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 14.9, 15.0, 15.1, 15.2, 15.2, 15.3, 15.5, 15.6, 15.7,
     15.8, 15.9, 16.0, 16.1, 16.2, 16.3, 16.4, 16.5, 16.6, 16.7, 16.8, 16.8, 16.9, 17.0, 17.0, 17.1, 17.2, 17.3, 17.4,
     17.5, 17.6, 17.8])

serebro_g = np.array(
    [17.7, 17.7, 18.0, 18.0, 18.1, 18.2, 18.3, 18.4, 18.4, 18.5, 18.7, 18.8, 18.8, 18.9, 19.0, 19.1, 19.1, 19.3, 19.4,
     19.5, 19.6, 19.6, 19.7, 19.7, 19.9, 19.9, 20.1, 20.1, 20.2, 20.3, 20.4, 20.5, 20.5, 20.6, 20.7, 20.7, 20.8, 20.9,
     20.9, 21.1, 21.2])

faki_g = np.array(
    [17.0, 17.1, 17.1, 17.4, 17.5, 17.5, 17.5, 17.6, 17.7, 17.7, 17.8, 17.9, 18.0, 18.1, 18.2, 18.3, 18.3, 18.4, 18.5,
     18.6, 18.7, 18.7, 18.8, 18.9, 19.0, 19.1, 19.1, 19.2, 19.2, 19.3, 19.5, 19.6, 19.7, 19.7, 19.7, 19.8, 19.9, 19.9,
     20.1, 20.2, 20.2])

apelsin_g = np.array(
    [17.2, 17.3, 17.5, 17.5, 17.6, 17.7, 17.7, 17.8, 17.9, 18.0, 18.1, 18.2, 18.3, 18.3, 18.4, 18.5, 18.5, 18.6, 18.7,
     18.9, 18.9, 19.0, 19.1, 19.2, 19.3, 19.3, 19.4, 19.5, 19.6, 19.7, 19.7, 19.8, 19.8, 19.9, 20.0, 20.0, 20.1, 20.2,
     20.3, 20.4, 20.4])

# Предельные массы (оболочка + груз) в граммах
m_final_g = {
    'Медный': 23.9,
    'Серебро': 23.9,
    'Факи чемпион': 23.9,
    'Апельсин': 23.9
}

# Перевод в мг и секунды (для расчёта k)
time_s = time_min * 60
medny_mg = medny_g * 1000
serebro_mg = serebro_g * 1000
faki_mg = faki_g * 1000
apelsin_mg = apelsin_g * 1000

names = ['Медный', 'Серебро', 'Факи чемпион', 'Апельсин']
data_series = [medny_mg, serebro_mg, faki_mg, apelsin_mg]
colors = ['#CD7F32', '#708090', '#0055A4', '#FF9F43']


def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


# Анализ через ЛИНЕЙНУЮ АППРОКСИМАЦИЮ ln(m_∞ - m) = ln(Δm) - kt
def analyze_linearized(time_s, mass_mg, name, m_final_mg):
    print(f"\n{'=' * 70}")
    print(f"ЛИНЕАРИЗОВАННЫЙ АНАЛИЗ: {name}")
    print(f"{'=' * 70}")

    delta_m_exp = m_final_mg - mass_mg
    valid = delta_m_exp > 0
    ln_delta = np.log(delta_m_exp[valid])
    time_valid = time_s[valid]

    slope, intercept, r_value, p_value, std_err = stats.linregress(time_valid, ln_delta)

    k = -slope
    k_error = std_err
    delta_m_0 = np.exp(intercept)
    r2_linear = r_value ** 2

    mass_pred = m_final_mg - np.exp(intercept + slope * time_valid)
    r2_mass = calculate_r2(mass_mg[valid], mass_pred)

    print(f"\nПараметры из линейной аппроксимации:")
    print(f"  ln(m_∞ - m) = ln(Δm₀) - k·t")
    print(f"  k = {k:.6f} ± {k_error:.6f} с⁻¹")
    print(f"  Δm₀ = {delta_m_0:.2f} мг")
    print(f"  R² (линейная) = {r2_linear:.6f}")
    print(f"  R² (масса)    = {r2_mass:.6f}")

    tau = 1 / k
    print(f"  τ = 1/k = {tau:.0f} с = {tau / 60:.1f} мин")

    return {
        'k': k,
        'k_error': k_error,
        'delta_m_0': delta_m_0,
        'intercept': intercept,
        'slope': slope,
        'r2_linear': r2_linear,
        'r2_mass': r2_mass,
        'tau': tau,
        'm_final': m_final_mg
    }


# Анализ всех серий
results = {}
for mass, name in zip(data_series, names):
    m_final_mg = m_final_g[name] * 1000
    results[name] = analyze_linearized(time_s, mass, name, m_final_mg)

# ============================================================
# ОКНО 1: Экспоненциальное насыщение (4 графика 2x2)
# ============================================================
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
axes1 = axes1.flatten()

for i, (mass, name, color) in enumerate(zip(data_series, names, colors)):
    ax = axes1[i]
    k = results[name]['k']
    delta_m_0 = results[name]['delta_m_0']
    m_final = results[name]['m_final']

    ax.scatter(time_min, mass, c=color, alpha=0.7, label='Эксперимент', s=50, zorder=5, edgecolors='black',
               linewidth=0.5)

    t_fit_min = np.linspace(time_min.min(), time_min.max(), 500)
    t_fit_s = t_fit_min * 60
    y_fit = m_final - delta_m_0 * np.exp(-k * t_fit_s)
    ax.plot(t_fit_min, y_fit, '-', c=color, linewidth=2.5, label=f'Аппроксимация\nR²={results[name]["r2_mass"]:.4f}')

    ax.set_xlabel('Время, мин', fontsize=11)
    ax.set_ylabel('Масса, мг', fontsize=11)
    ax.set_title(f'{name}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Экспоненциальное насыщение массы шариков', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================
# ОКНО 2: Производная (4 графика 2x2)
# ============================================================
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
axes2 = axes2.flatten()

for i, (mass, name, color) in enumerate(zip(data_series, names, colors)):
    ax = axes2[i]
    k = results[name]['k']
    delta_m_0 = results[name]['delta_m_0']

    mass_smooth = gaussian_filter1d(mass, sigma=1.5)
    derivative_exp = np.gradient(mass_smooth, time_s)
    ax.scatter(time_min, derivative_exp, c=color, alpha=0.6, label='Эксперимент', s=40, zorder=5, edgecolors='black',
               linewidth=0.5)

    t_fit_min = np.linspace(time_min.min(), time_min.max(), 500)
    t_fit_s = t_fit_min * 60
    derivative_analytic = delta_m_0 * k * np.exp(-k * t_fit_s)
    ax.plot(t_fit_min, derivative_analytic, '-', c=color, linewidth=2.5,
            label=f'dm/dt = {delta_m_0 * k:.4f}·exp(-{k:.6f}·t)')

    # Горизонтальная пунктирная линия - средняя скорость (показывает, что производная НЕ постоянна)
    avg_derivative = np.mean(derivative_exp)
    ax.axhline(y=avg_derivative, color=color, linestyle='--', linewidth=1.5, alpha=0.5,
               label=f'Средняя: {avg_derivative:.4f}')

    ax.set_xlabel('Время, мин', fontsize=11)
    ax.set_ylabel('dm/dt, мг/с', fontsize=11)
    ax.set_title(f'{name}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Производная массы', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================
# ОКНО 3: Линеаризация (4 графика 2x2)
# ============================================================
fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
axes3 = axes3.flatten()

for i, (mass, name, color) in enumerate(zip(data_series, names, colors)):
    ax = axes3[i]
    m_final = results[name]['m_final']

    delta_m_exp = m_final - mass
    valid = delta_m_exp > 0
    ax.scatter(time_min[valid], np.log(delta_m_exp[valid]), c=color, alpha=0.7, s=50, label='Эксперимент', zorder=5,
               edgecolors='black', linewidth=0.5)

    t_fit_min = np.linspace(time_min.min(), time_min.max(), 500)
    t_fit_s = t_fit_min * 60
    ln_delta_fit = results[name]['intercept'] + results[name]['slope'] * t_fit_s
    ax.plot(t_fit_min, ln_delta_fit, '-', c=color, linewidth=2.5,
            label=f'ln(Δm) = {results[name]["intercept"]:.2f} {results[name]["slope"]:+.6f}·t\nR²={results[name]["r2_linear"]:.4f}')

    ax.set_xlabel('Время, мин', fontsize=11)
    ax.set_ylabel('ln(m_∞ - m)', fontsize=11)
    ax.set_title(f'{name} (определение k)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Линеаризация: ln(m_∞ - m) от времени', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================
# ОКНО 4: Сводный график насыщения (все шарики)
# ============================================================
fig4, ax4 = plt.subplots(figsize=(12, 8))

for mass, name, color in zip(data_series, names, colors):
    k = results[name]['k']
    delta_m_0 = results[name]['delta_m_0']
    m_final = results[name]['m_final']

    ax4.scatter(time_min, mass, c=color, alpha=0.6, label=f'{name} (эксп.)', s=40, zorder=5, edgecolors='black',
                linewidth=0.5)

    t_fit_min = np.linspace(time_min.min(), time_min.max(), 500)
    t_fit_s = t_fit_min * 60
    y_fit = m_final - delta_m_0 * np.exp(-k * t_fit_s)
    ax4.plot(t_fit_min, y_fit, '-', c=color, linewidth=2.5, label=f'{name} (аппрок.)', alpha=0.8)

ax4.set_xlabel('Время, мин', fontsize=12)
ax4.set_ylabel('Масса, мг', fontsize=12)
ax4.set_title('Все шарики: Экспоненциальное насыщение', fontsize=14, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# ОКНО 5: Сводный график производных (все шарики)
# ============================================================
fig5, ax5 = plt.subplots(figsize=(12, 8))

for name, color in zip(names, colors):
    k = results[name]['k']
    delta_m_0 = results[name]['delta_m_0']

    t_fit_min = np.linspace(time_min.min(), time_min.max(), 500)
    t_fit_s = t_fit_min * 60
    derivative = delta_m_0 * k * np.exp(-k * t_fit_s)
    ax5.plot(t_fit_min, derivative, '-', c=color, linewidth=2.5, label=f'{name}: {delta_m_0 * k:.4f}·exp(-{k:.6f}·t)',
             alpha=0.8)

ax5.set_xlabel('Время, мин', fontsize=12)
ax5.set_ylabel('dm/dt, мг/с', fontsize=12)
ax5.set_title('Все шарики: Производная массы', fontsize=14, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# ОКНО 6: Сводный график линеаризации (все шарики)
# ============================================================
fig6, ax6 = plt.subplots(figsize=(12, 8))

for mass, name, color in zip(data_series, names, colors):
    m_final = results[name]['m_final']
    delta_m_exp = m_final - mass
    valid = delta_m_exp > 0

    ax6.scatter(time_min[valid], np.log(delta_m_exp[valid]), c=color, alpha=0.6, label=f'{name} (эксп.)', s=40,
                zorder=5, edgecolors='black', linewidth=0.5)

    t_fit_min = np.linspace(time_min.min(), time_min.max(), 500)
    t_fit_s = t_fit_min * 60
    ln_fit = results[name]['intercept'] + results[name]['slope'] * t_fit_s
    ax6.plot(t_fit_min, ln_fit, '-', c=color, linewidth=2.5, label=f'{name} (аппрок.)', alpha=0.8)

ax6.set_xlabel('Время, мин', fontsize=12)
ax6.set_ylabel('ln(m_∞ - m)', fontsize=12)
ax6.set_title('Все шарики: Линеаризация', fontsize=14, fontweight='bold')
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# СВОДНАЯ ТАБЛИЦА
# ============================================================
print("\n" + "=" * 100)
print("СВОДНАЯ ТАБЛИЦА ПАРАМЕТРОВ ДИФФУЗИИ ГЕЛИЯ")
print("=" * 100)
print(f"{'Шарик':<15} {'k, с⁻¹':<20} {'Δk, с⁻¹':<15} {'τ, мин':<12} {'R² (лин)':<12} {'R² (масса)':<12}")
print("-" * 100)

for name in names:
    k = results[name]['k']
    k_err = results[name]['k_error']
    tau = results[name]['tau']
    r2_lin = results[name]['r2_linear']
    r2_mass = results[name]['r2_mass']
    print(f"{name:<15} {k:<20.6f} {k_err:<15.6f} {tau / 60:<12.1f} {r2_lin:<12.6f} {r2_mass:<12.6f}")

print("\n" + "=" * 100)
print("МЕТОД ОПРЕДЕЛЕНИЯ k:")
print("=" * 100)
print("1. Вычисляем ln(m_∞ - m(t)) для каждого измерения")
print("2. Строим линейную регрессию: ln(m_∞ - m) = ln(Δm₀) - k·t")
print("3. Коэффициент k = -slope (отрицательный угловой коэффициент)")
print("4. Погрешность k = стандартная ошибка наклона из linregress")
print("=" * 100)