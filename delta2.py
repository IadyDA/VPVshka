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
R = 8.314
mu_air = 28.97e-3
mu_He = 4.00e-3

# === ПЛОТНОСТИ МАТЕРИАЛОВ (кг/м³ → г/см³) ===
# 1 кг/м³ = 0.001 г/см³
rho_rubber = {
    'Медный': 0.910,  # 910 кг/м³
    'Серебро': 0.905,  # 910 кг/м³
    'Факи Чемпион': 0.938,  # 934 кг/м³
    'Апельсин': 0.934  # 934 кг/м³
}

# Разность плотностей воздуха и гелия (г/см³)
rho_diff = (P_0_Pa / (R * T)) * (mu_air - mu_He) / 1000

# === ДАННЫЕ ШАРИКОВ ===
shell_masses = {'Медный': 2.0, 'Серебро': 2.3, 'Факи Чемпион': 2.1, 'Апельсин': 2.2}

S_params = {
    'Медный': {'S_0': 4087.1, 'k_S': -3.716},
    'Серебро': {'S_0': 3231.2, 'k_S': -3.310},
    'Факи Чемпион': {'S_0': 3584.1, 'k_S': -3.936},
    'Апельсин': {'S_0': 2461.8, 'k_S': -4.363}
}

exp_params = {
    'Медный': {'delta_m_mg': 13873.22, 'k_s': 0.000029, 'tau': 580.1},
    'Серебро': {'delta_m_mg': 10904.74, 'k_s': 0.000033, 'tau': 502.3},
    'Факи Чемпион': {'delta_m_mg': 11209.55, 'k_s': 0.000029, 'tau': 569.2},
    'Апельсин': {'delta_m_mg': 6763.84, 'k_s': 0.000056, 'tau': 295.6}
}

# Цвета и стили линий для различения
styles = {
    'Медный': {'color': '#CD7F32', 'linestyle': '-', 'marker': 'o'},
    'Серебро': {'color': '#708090', 'linestyle': '--', 'marker': 's'},
    'Факи Чемпион': {'color': '#0055A4', 'linestyle': '-.', 'marker': '^'},
    'Апельсин': {'color': '#FF9F43', 'linestyle': ':', 'marker': 'd'}
}


def calculate_D(t_sec, name):
    m_shell = shell_masses[name]
    S_p = S_params[name]
    E_p = exp_params[name]
    rho = rho_rubber[name]  # 🔹 Индивидуальная плотность для данного шарика

    t_min = t_sec / 60
    S = S_p['S_0'] + S_p['k_S'] * (t_min - 37)
    if S <= 100:
        return None, None

    # 🔹 Толщина оболочки с учётом индивидуальной плотности: δ = m / (ρ·S)
    delta = m_shell / (rho * S)

    delta_m_g = E_p['delta_m_mg'] / 1000
    k_s = E_p['k_s']
    beta = delta_m_g * k_s * np.exp(-k_s * t_sec)

    D = (beta * delta) / (S * rho_diff)
    return D, S


# === РАСЧЁТ ===
print("📊 Расчёт D(t) для всех шариков (с учётом индивидуальных плотностей)...")
results = {}

for name in shell_masses.keys():
    tau_min = exp_params[name]['tau']
    t_minutes = np.linspace(0, tau_min, 250)
    t_seconds = t_minutes * 60
    D_vals = []
    for t_sec in t_seconds:
        D, S = calculate_D(t_sec, name)
        if D is not None and 0 < D < 1e-5:
            D_vals.append(D)
    if D_vals:
        D_vals = np.array(D_vals)
        results[name] = {
            't': t_minutes[:len(D_vals)],
            'D': D_vals,
            'D_mean': np.mean(D_vals),
            'D_std': np.std(D_vals),
            'tau': tau_min
        }
        print(
            f"✓ {name:15s} τ={tau_min:5.1f} мин | ρ={rho_rubber[name]:.3f} г/см³ | ⟨D⟩={np.mean(D_vals) * 1e8:5.2f}×10⁻⁸ см²/с")

# === ОДИН ОБЩИЙ ГРАФИК ===
fig, ax = plt.subplots(figsize=(10, 6))

for name in results.keys():
    data = results[name]
    st = styles[name]

    # Основная кривая
    ax.plot(data['t'], data['D'] * 1e8,
            color=st['color'], linestyle=st['linestyle'],
            marker=st['marker'], markevery=25, markersize=4,
            linewidth=1.5, label=name, alpha=0.9)

    # Область неопределённости (полупрозрачная)
    ax.fill_between(data['t'],
                    np.maximum(0, (data['D'] - data['D_std']) * 1e8),
                    (data['D'] + data['D_std']) * 1e8,
                    color=st['color'], alpha=0.15)

    # Горизонтальная линия среднего (тонкая)
    ax.axhline(data['D_mean'] * 1e8, color=st['color'], linestyle=':',
               linewidth=0.7, alpha=0.5)

# Оформление графика
ax.set_xlabel('Время, мин', fontsize=10)
ax.set_ylabel('Коэффициент диффузии D, 10⁻⁸ см²/с', fontsize=10)
ax.set_title('Зависимость D(t) для разных шариков\n(учтены индивидуальные плотности резины)', fontsize=12, pad=15)
ax.grid(True, linestyle=':', alpha=0.5)
ax.legend(frameon=True, fancybox=True, loc='upper right', fontsize=8)
ax.set_xlim(0, max(r['tau'] for r in results.values()))
ax.set_axisbelow(True)

# Добавляем легенду со средними значениями
legend_text = '\n'.join([f"{name}: ⟨D⟩={results[name]['D_mean'] * 1e8:.2f}×10⁻⁸ см²/с"
                         for name in results.keys()])
props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, fontsize=8,
        verticalalignment='top', bbox=props, family='monospace')

plt.tight_layout()
plt.savefig('D_all_together_individual_rho.png', dpi=200, bbox_inches='tight')
plt.show()

# === ИТОГОВАЯ ТАБЛИЦА ===
print("\n📋 Сводная таблица коэффициентов диффузии:")
print(f"{'Шарик':<18} {'ρ, г/см³':<10} {'D⟩, 10⁻ см²/с':<20} {'σ, 10⁻⁸ см²/с':<20} {'τ, мин':<10}")
print("-" * 78)
for name in results.keys():
    d = results[name]
    rel = d['D_std'] / d['D_mean'] * 100
    print(
        f"{name:<18} {rho_rubber[name]:<10.3f} {d['D_mean'] * 1e8:<20.2f} {d['D_std'] * 1e8:<20.2f} {d['tau']:<10.1f} ({rel:.1f}%)")
print("-" * 78)
print(f"\n📚 Литературное значение: ~5–10 × 10⁸ см²/с ✓")

# Добавьте этот код после основного расчёта:
print("\n🔍 Проверка: нормировка D·S² (должно быть ~const):")
print(f"{'Шарик':<18} {'D·S², 10⁻² см⁴/с':<20}")
print("-" * 40)
for name in results.keys():
    S_avg = S_params[name]['S_0']  # можно взять среднее по интервалу
    D_mean = results[name]['D_mean']
    print(f"{name:<18} {D_mean * S_avg**2 * 1e2:<20.2f}")