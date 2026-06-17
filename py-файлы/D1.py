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
rho_rubber = 1.05

rho_diff = (P_0_Pa / (R * T)) * (mu_air - mu_He) / 1000 # rho - rho_He = delta rho

# === Мои расчёты ===
k = {
    'Медный': {'k': 4087.1, 'sigma_k': -3.716},
    'Серебро': {'k': 3231.2, 'sigma_k': -3.310},
    'Факи Чемпион': {'k': 3584.1, 'ksigma_k_S': -3.936},
    'Апельсин': {'k': 2461.8, 'k_sigma_kS': -4.363}
}



# ===================

# === ДАННЫЕ ШАРИКОВ ===
shell_masses = {
    'Медный': 2.0, 'Серебро': 2.3, 'Факи Чемпион': 2.1, 'Апельсин': 2.2
}

S_params = {
    'Медный': {'S_0': 4087.1, 'k_S': -3.716, 'dS_0': 84.4, 'dk_S': 0.335},
    'Серебро': {'S_0': 3231.2, 'k_S': -3.310, 'dS_0': 65.6, 'dk_S': 0.253},
    'Факи Чемпион': {'S_0': 3584.1, 'k_S': -3.936, 'dS_0': 72.4, 'dk_S': 0.275},
    'Апельсин': {'S_0': 2461.8, 'k_S': -4.363, 'dS_0': 47.0, 'dk_S': 0.152}
}

exp_params = {
    'Медный': {'m_inf': 27.7, 'delta_m_mg': 13873.22, 'k_s': 0.000029, 'tau': 580.1},
    'Серебро': {'m_inf': 28.6, 'delta_m_mg': 10904.74, 'k_s': 0.000033, 'tau': 502.3},
    'Факи Чемпион': {'m_inf': 28.2, 'delta_m_mg': 11209.55, 'k_s': 0.000029, 'tau': 569.2},
    'Апельсин': {'m_inf': 23.9, 'delta_m_mg': 6763.84, 'k_s': 0.000056, 'tau': 295.6}
}

colors = {
    'Медный': '#CD7F32', 'Серебро': '#708090',
    'Факи Чемпион': '#0055A4', 'Апельсин': '#FF9F43'
}


def calculate_D(t_sec, name):
    m_shell = shell_masses[name]
    S_p = S_params[name]
    E_p = exp_params[name]

    t_min = t_sec / 60
    S = S_p['S_0'] + S_p['k_S'] * (t_min - 37)
    if S <= 100:  # защита от слишком малой площади
        return None, None, None, None

    delta = m_shell / (rho_rubber * S)
    delta_m_g = E_p['delta_m_mg'] / 1000
    k_s = E_p['k_s']
    beta = delta_m_g * k_s * np.exp(-k_s * t_sec)

    D = (beta * delta) / (S * rho_diff)
    return D, S, delta, beta


# === РАСЧЁТ ===
print("=" * 70)
print("📊 РАСЧЁТ D(t) до характеристического времени τ")
print("=" * 70)

results = {}

for name in shell_masses.keys():
    tau_min = exp_params[name]['tau']  # характеристическое время в минутах
    t_minutes = np.linspace(0, tau_min, 300)  # 300 точек до τ
    t_seconds = t_minutes * 60

    D_vals, S_vals = [], []
    for t_sec in t_seconds:
        D, S, delta, beta = calculate_D(t_sec, name)
        if D is not None and 0 < D < 1e-5:
            D_vals.append(D)
            S_vals.append(S)

    if D_vals:
        D_vals = np.array(D_vals)
        results[name] = {
            't': t_minutes[:len(D_vals)],
            'D': D_vals,
            'D_mean': np.mean(D_vals),
            'D_std': np.std(D_vals),
            'tau': tau_min
        }
        print(f"🎈 {name:15s} τ={tau_min:5.1f} мин | "
              f"⟨D⟩={np.mean(D_vals) * 1e8:6.2f}±{np.std(D_vals) * 1e8:4.2f}×10⁻⁸ см²/с")

print("=" * 70)

# === ГРАФИКИ (компактные 2×2) ===
fig, axes = plt.subplots(2, 2, figsize=(11, 7))  # уменьшено с (14,10)
axes = axes.flatten()

for idx, name in enumerate(shell_masses.keys()):
    if name in results:
        ax = axes[idx]
        data = results[name]

        ax.plot(data['t'], data['D'] * 1e8, '-', color=colors[name], linewidth=1.5)
        ax.fill_between(data['t'],
                        np.maximum(0, (data['D'] - data['D_std']) * 1e8),
                        (data['D'] + data['D_std']) * 1e8,
                        color=colors[name], alpha=0.25)
        ax.axhline(data['D_mean'] * 1e8, color=colors[name], linestyle='--',
                   linewidth=0.8, alpha=0.7)

        ax.set_xlabel('t, мин', fontsize=9)
        ax.set_ylabel('D, 10⁻⁸ см²/с', fontsize=9)
        ax.set_title(f"{name}\n(τ={data['tau']:.0f} мин)", fontsize=10, pad=8)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.set_xlim(0, data['tau'])
        ax.tick_params(axis='both', labelsize=8)

plt.suptitle('Коэффициент диффузии гелия D(t)', fontsize=12, y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('D_vs_time_compact.png', dpi=200, bbox_inches='tight')
plt.show()

# === СРАВНЕНИЕ (компактная столбчатая диаграмма) ===
fig, ax = plt.subplots(figsize=(7, 4))  # уменьшено с (10,6)

names = list(results.keys())
D_means = [results[n]['D_mean'] * 1e8 for n in names]
D_stds = [results[n]['D_std'] * 1e8 for n in names]

bars = ax.bar(names, D_means, yerr=D_stds, capsize=5,
              color=[colors[n] for n in names], alpha=0.85, width=0.6)

ax.set_ylabel('⟨D⟩, 10⁻⁸ см²/с', fontsize=10)
ax.set_title('Сравнение коэффициентов диффузии', fontsize=11, pad=12)
ax.grid(axis='y', linestyle=':', alpha=0.4)
ax.tick_params(axis='both', labelsize=9)
ax.set_axisbelow(True)

for bar, val in zip(bars, D_means):
    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.2,
            f'{val:.1f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('D_comparison_compact.png', dpi=200, bbox_inches='tight')
plt.show()

# === ИТОГОВАЯ ТАБЛИЦА ===
print("\n📋 РЕЗУЛЬТАТЫ (усреднённые до τ):")
print(f"{'Шарик':<18} {'⟨D⟩, 10⁻⁸ см²/с':<18} {'σ, 10⁻⁸ см²/с':<18} {'τ, мин':<10}")
print("-" * 64)
for name in names:
    d = results[name]
    print(f"{name:<18} {d['D_mean'] * 1e8:<18.2f} {d['D_std'] * 1e8:<18.2f} {d['tau']:<10.1f}")
print("-" * 64)