import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy import stats

# Данные (время в часах, масса в граммах)
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

# Предельные массы (г)
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
time_hours = [time_maly_ch, time_sred_ch, time_bolsh_ch, time_holod_ch]
colors = ['red', 'orange', 'green', 'blue']


# ============================================================
# ФУНКЦИИ КРАСИВОГО ФОРМАТИРОВАНИЯ
# ============================================================
def format_scientific(value, digits=2):
    """Форматирует число в научный вид: 2.5·10⁻⁵ (с учётом знака)"""
    if value == 0:
        return "0"
    sign = '-' if value < 0 else ''
    exp = int(np.floor(np.log10(abs(value))))
    mantissa = abs(value) / (10 ** exp)

    superscripts = {
        '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³',
        '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹'
    }

    exp_str = str(exp)
    exp_superscript = ''.join(superscripts[c] for c in exp_str)

    return f"{sign}{mantissa:.{digits}f}·10{exp_superscript}"


def format_k_scientific(k):
    """Форматирует положительное число k в научный вид"""
    return format_scientific(k, 2)


def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


# Анализ через ЛИНЕЙНУЮ АППРОКСИМАЦИЮ
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
    print(f"  k = {format_k_scientific(k)} с⁻¹")
    print(f"  k = {k:.6f} ± {k_error:.6f} с⁻¹")
    print(f"  Δm₀ = {delta_m_0:.2f} мг")
    print(f"  R² (линейная) = {r2_linear:.6f}")
    print(f"  R² (масса)    = {r2_mass:.6f}")

    tau = 1 / k
    print(f"  τ = 1/k = {tau:.0f} с = {tau / 3600:.2f} ч = {tau / 60:.1f} мин")

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
for mass, time_s, name in zip(data_series, time_series, names):
    m_final_mg = m_final_g[name] * 1000
    results[name] = analyze_linearized(time_s, mass, name, m_final_mg)

# ============================================================
# ОКНО 1: Экспоненциальное насыщение (4 графика 2x2) - С ФОРМУЛОЙ
# ============================================================
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
axes1 = axes1.flatten()

for i, (mass, time_ch, name, color) in enumerate(zip(data_series, time_hours, names, colors)):
    ax = axes1[i]
    k = results[name]['k']
    delta_m_0 = results[name]['delta_m_0']
    m_final = results[name]['m_final']

    ax.scatter(time_ch, mass / 1000, c=color, alpha=0.7, label='Эксперимент', s=50, zorder=5, edgecolors='black',
               linewidth=0.5)

    t_fit_ch = np.linspace(time_ch.min(), time_ch.max(), 500)
    t_fit_s = t_fit_ch * 3600
    y_fit = m_final - delta_m_0 * np.exp(-k * t_fit_s)
    ax.plot(t_fit_ch, y_fit / 1000, '-', c=color, linewidth=2.5,
            label=f'm(t) = {m_final / 1000:.1f} - {delta_m_0 / 1000:.1f}·exp(-{format_k_scientific(k)}·t)\nR²={results[name]["r2_mass"]:.4f}')

    ax.set_xlabel('Время, ч', fontsize=11)
    ax.set_ylabel('Масса, г', fontsize=11)
    ax.set_title(f'{name} шарик', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Экспоненциальное насыщение массы шариков (вторая группа)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================
# ОКНО 2: Производная (4 графика 2x2)
# ============================================================
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
axes2 = axes2.flatten()

for i, (mass, time_s, time_ch, name, color) in enumerate(zip(data_series, time_series, time_hours, names, colors)):
    ax = axes2[i]
    k = results[name]['k']
    delta_m_0 = results[name]['delta_m_0']

    mass_smooth = gaussian_filter1d(mass, sigma=1.5)
    derivative_exp = np.gradient(mass_smooth, time_s)
    ax.scatter(time_ch, derivative_exp, c=color, alpha=0.6, label='Эксперимент', s=40, zorder=5, edgecolors='black',
               linewidth=0.5)

    t_fit_ch = np.linspace(time_ch.min(), time_ch.max(), 500)
    t_fit_s = t_fit_ch * 3600
    derivative_analytic = delta_m_0 * k * np.exp(-k * t_fit_s)
    ax.plot(t_fit_ch, derivative_analytic, '-', c=color, linewidth=2.5,
            label=f'dm/dt = {delta_m_0 * k:.3f}·exp(-{format_k_scientific(k)}·t)')

    avg_derivative = np.mean(derivative_exp)
    ax.axhline(y=avg_derivative, color=color, linestyle='--', linewidth=1.5, alpha=0.5,
               label=f'Средняя: {avg_derivative:.3f}')

    ax.set_xlabel('Время, ч', fontsize=11)
    ax.set_ylabel('dm/dt, мг/с', fontsize=11)
    ax.set_title(f'{name} шарик', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Производная массы', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================
# ОКНО 3: Линеаризация (4 графика 2x2) - С НАУЧНЫМ ФОРМАТОМ slope
# ============================================================
fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
axes3 = axes3.flatten()

for i, (mass, time_s, time_ch, name, color) in enumerate(zip(data_series, time_series, time_hours, names, colors)):
    ax = axes3[i]
    m_final = results[name]['m_final']
    intercept = results[name]['intercept']
    slope = results[name]['slope']

    delta_m_exp = m_final - mass
    valid = delta_m_exp > 0
    ax.scatter(time_ch[valid], np.log(delta_m_exp[valid]), c=color, alpha=0.7, s=50, label='Эксперимент', zorder=5,
               edgecolors='black', linewidth=0.5)

    t_fit_ch = np.linspace(time_ch.min(), time_ch.max(), 500)
    t_fit_s = t_fit_ch * 3600
    ln_delta_fit = intercept + slope * t_fit_s
    ax.plot(t_fit_ch, ln_delta_fit, '-', c=color, linewidth=2.5,
            label=f'ln(Δm) = {intercept:.2f} {format_scientific(slope)}·t\nR²={results[name]["r2_linear"]:.4f}')

    ax.set_xlabel('Время, ч', fontsize=11)
    ax.set_ylabel('ln(m_∞ - m)', fontsize=11)
    ax.set_title(f'{name} шарик', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Линеаризация: ln(m_∞ - m) от времени', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================
# ОКНО 4: Сводный график насыщения (все шарики) - С ФОРМУЛОЙ
# ============================================================
fig4, ax4 = plt.subplots(figsize=(12, 8))

for mass, time_ch, name, color in zip(data_series, time_hours, names, colors):
    k = results[name]['k']
    delta_m_0 = results[name]['delta_m_0']
    m_final = results[name]['m_final']

    ax4.scatter(time_ch, mass / 1000, c=color, alpha=0.6, s=40, zorder=5, edgecolors='black',
                linewidth=0.5)

    t_fit_ch = np.linspace(time_ch.min(), time_ch.max(), 500)
    t_fit_s = t_fit_ch * 3600
    y_fit = m_final - delta_m_0 * np.exp(-k * t_fit_s)
    ax4.plot(t_fit_ch, y_fit / 1000, '-', c=color, linewidth=2.5,
             label=f'{name}: m(t) = {m_final / 1000:.1f} - {delta_m_0 / 1000:.1f}·exp(-{format_k_scientific(k)}·t)',
             alpha=0.8)

ax4.set_xlabel('Время, ч', fontsize=12)
ax4.set_ylabel('Масса, г', fontsize=12)
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

    idx = names.index(name)
    time_ch = time_hours[idx]

    t_fit_ch = np.linspace(time_ch.min(), time_ch.max(), 500)
    t_fit_s = t_fit_ch * 3600
    derivative = delta_m_0 * k * np.exp(-k * t_fit_s)
    ax5.plot(t_fit_ch, derivative, '-', c=color, linewidth=2.5,
             label=f'{name}: {delta_m_0 * k:.3f}·exp(-{format_k_scientific(k)}·t)', alpha=0.8)

ax5.set_xlabel('Время, ч', fontsize=12)
ax5.set_ylabel('dm/dt, мг/с', fontsize=12)
ax5.set_title('Все шарики: производная массы', fontsize=14, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# ОКНО 6: Сводный график линеаризации - С НАУЧНЫМ ФОРМАТОМ
# ============================================================
fig6, ax6 = plt.subplots(figsize=(12, 8))

for mass, time_s, time_ch, name, color in zip(data_series, time_series, time_hours, names, colors):
    m_final = results[name]['m_final']
    intercept = results[name]['intercept']
    slope = results[name]['slope']

    delta_m_exp = m_final - mass
    valid = delta_m_exp > 0

    ax6.scatter(time_ch[valid], np.log(delta_m_exp[valid]), c=color, alpha=0.6, s=40, zorder=5,
                edgecolors='black', linewidth=0.5)

    t_fit_ch = np.linspace(time_ch.min(), time_ch.max(), 500)
    t_fit_s = t_fit_ch * 3600
    ln_fit = intercept + slope * t_fit_s
    ax6.plot(t_fit_ch, ln_fit, '-', c=color, linewidth=2.5,
             label=f'{name}: {intercept:.2f} {format_scientific(slope)}·t', alpha=0.8)

ax6.set_xlabel('Время, ч', fontsize=12)
ax6.set_ylabel('ln(m_∞ - m)', fontsize=12)
ax6.set_title('Все шарики: Линеаризация', fontsize=14, fontweight='bold')
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# СВОДНАЯ ТАБЛИЦА
# ============================================================
print("\n" + "=" * 110)
print("СВОДНАЯ ТАБЛИЦА ПАРАМЕТРОВ ДИФФУЗИИ ГЕЛИЯ (ВТОРАЯ ГРУППА)")
print("=" * 110)
print(f"{'Шарик':<12} {'k, с⁻¹':<25} {'Δk, с⁻¹':<15} {'τ, ч':<12} {'τ, мин':<10} {'R² (лин)':<12} {'R² (масса)':<12}")
print("-" * 110)

for name in names:
    k = results[name]['k']
    k_err = results[name]['k_error']
    tau = results[name]['tau']
    r2_lin = results[name]['r2_linear']
    r2_mass = results[name]['r2_mass']
    print(
        f"{name:<12} {format_k_scientific(k):<25} {k_err:<15.9f} {tau / 3600:<12.2f} {tau / 60:<10.1f} {r2_lin:<12.6f} {r2_mass:<12.6f}")

print("\n" + "=" * 110)
print("МЕТОД ОПРЕДЕЛЕНИЯ k:")
print("=" * 110)
print("1. Вычисляем ln(m_∞ - m(t)) для каждого измерения")
print("2. Строим линейную регрессию: ln(m_∞ - m) = ln(Δm₀) - k·t")
print("3. Коэффициент k = -slope (отрицательный угловой коэффициент)")
print("4. Погрешность k = стандартная ошибка наклона из linregress")
print("\nВРЕМЯ: измеряется в часах, для расчёта k переводится в секунды")
print("=" * 110)