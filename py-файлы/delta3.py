import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# === НАСТРОЙКИ ГРАФИКА ===
rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 9
rcParams['axes.labelsize'] = 10
rcParams['axes.titlesize'] = 11
rcParams['legend.fontsize'] = 7
rcParams['figure.dpi'] = 120

# === ФИЗИЧЕСКИЕ КОНСТАНТЫ ===
P_0_Pa = 742 * 133.322  # 742 мм рт.ст. → Па
T = 298  # К
R = 8.314  # Дж/(моль·К)
mu_air = 28.97e-3  # кг/моль
mu_He = 4.00e-3  # кг/моль
rho_rubber = 1.05  # г/см³
rho_diff = (P_0_Pa / (R * T)) * (mu_air - mu_He) / 1000  # г/см³

# === ДАННЫЕ ШАРИКОВ ===
shell_masses = {'Медный': 2.0, 'Серебро': 2.3, 'Факи Чемпион': 2.1, 'Апельсин': 2.2}

# Параметры площади: S(t) = S_0 + k_S·(t_мин - 37), см²
S_params = {
    'Медный': {'S_0': 4087.1, 'k_S': -3.716},
    'Серебро': {'S_0': 3231.2, 'k_S': -3.310},
    'Факи Чемпион': {'S_0': 3584.1, 'k_S': -3.936},
    'Апельсин': {'S_0': 2461.8, 'k_S': -4.363}
}

# Параметры моделей массы (t в МИНУТАХ, m в мг)
# Линейная: m = β_lin·t + m0
# Квадратичная: m = a·t² + b·t + c → β(t) = 2a·t + b
mass_models = {
    'Медный': {
        'linear': {'beta_mg_min': 0.3378},
        'quadratic': {'a': -8e-6, 'b': 0.4305, 'c': 13765.81},
        'tau': 580.1, 'color': '#CD7F32'
    },
    'Серебро': {
        'linear': {'beta_mg_min': 0.2995},
        'quadratic': {'a': -6e-6, 'b': 0.3731, 'c': 17671.05},
        'tau': 502.3, 'color': '#708090'
    },
    'Факи Чемпион': {
        'linear': {'beta_mg_min': 0.2782},
        'quadratic': {'a': -4e-6, 'b': 0.3203, 'c': 17001.70},
        'tau': 569.2, 'color': '#0055A4'
    },
    'Апельсин': {
        'linear': {'beta_mg_min': 0.2799},
        'quadratic': {'a': -6e-6, 'b': 0.3442, 'c': 17189.12},
        'tau': 295.6, 'color': '#FF9F43'
    }
}


def calculate_D(t_min, name, model_type):
    """Вычисляет D(t) для заданной модели массы"""
    m_shell = shell_masses[name]
    S_p = S_params[name]
    M_p = mass_models[name]

    # Площадь поверхности
    S = S_p['S_0'] + S_p['k_S'] * (t_min - 37)
    if S < 500:
        return None

    # Толщина оболочки
    delta = m_shell / (rho_rubber * S)  # см

    # Производная dm/dt = β [г/с]
    if model_type == 'linear':
        beta_mg_min = M_p['linear']['beta_mg_min']
        beta = beta_mg_min / 60 / 1000  # мг/мин → г/с
    else:  # quadratic
        a = M_p['quadratic']['a']  # мг/мин²
        b = M_p['quadratic']['b']  # мг/мин
        beta_mg_min = 2 * a * t_min + b  # мг/мин
        beta = beta_mg_min / 60 / 1000  # мг/мин → г/с

    # Коэффициент диффузии
    D = (beta * delta) / (S * rho_diff)  # см²/с
    return D


# === РАСЧЁТ ===
print("📊 Сравнение моделей: линейная vs квадратичная")
print("=" * 70)

results = {}
t_minutes = np.linspace(0, 600, 400)  # общий диапазон для сравнения

for name in shell_masses.keys():
    tau = mass_models[name]['tau']
    t_plot = np.linspace(0, tau, 300)

    D_lin, D_quad = [], []
    for t in t_plot:
        d_lin = calculate_D(t, name, 'linear')
        d_quad = calculate_D(t, name, 'quadratic')
        if d_lin and 0 < d_lin < 1e-5:
            D_lin.append(d_lin)
        if d_quad and 0 < d_quad < 1e-5:
            D_quad.append(d_quad)

    if D_lin and D_quad:
        results[name] = {
            't': t_plot[:min(len(D_lin), len(D_quad))],
            'D_lin': np.array(D_lin),
            'D_quad': np.array(D_quad),
            'tau': tau,
            'color': mass_models[name]['color'],
            'D_lin_mean': np.mean(D_lin),
            'D_quad_mean': np.mean(D_quad)
        }
        diff = abs(np.mean(D_lin) - np.mean(D_quad)) / np.mean(D_lin) * 100
        print(f"✓ {name:15s} | Δ⟨D⟩/⟨D⟩_лин = {diff:5.1f}%")

print("=" * 70)

# === ОДИН СРАВНИТЕЛЬНЫЙ ГРАФИК ===
fig, ax = plt.subplots(figsize=(11, 6))

for name, data in results.items():
    c = data['color']
    # Линейная модель — сплошная линия
    ax.plot(data['t'], data['D_lin'] * 1e8, '-', color=c, linewidth=2,
            label=f'{name} (лин.)', alpha=0.9)
    # Квадратичная модель — пунктир того же цвета
    ax.plot(data['t'], data['D_quad'] * 1e8, '--', color=c, linewidth=1.5,
            label=f'{name} (квадр.)', alpha=0.7)

ax.set_xlabel('Время, мин', fontsize=10)
ax.set_ylabel('Коэффициент диффузии D, 10⁻⁸ см²/с', fontsize=10)
ax.set_title('Влияние модели m(t) на расчёт D(t)', fontsize=12, pad=15)
ax.grid(True, linestyle=':', alpha=0.5)
ax.legend(ncol=2, fontsize=7, frameon=True, loc='upper right')
ax.set_xlim(0, max(d['tau'] for d in results.values()))
ax.set_axisbelow(True)

# Текстовая справка
info_text = 'Сплошная: линейная $m(t)$\nПунктир: квадратичная $m(t)$\n\n' + \
            '\n'.join([f"{n}: τ={d['tau']:.0f} мин" for n, d in results.items()])
props = dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray')
ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=8,
        verticalalignment='top', bbox=props, family='monospace')

plt.tight_layout()
plt.savefig('D_linear_vs_quadratic.png', dpi=200, bbox_inches='tight')
plt.show()

# === ТАБЛИЦА СРАВНЕНИЯ ===
print("\n📋 Сравнение средних значений D:")
print(f"{'Шарик':<18} {'⟨D⟩_лин, 10⁻⁸':<16} {'⟨D⟩_квадр, 10⁻⁸':<18} {'Разница, %':<12}")
print("-" * 64)
for name, data in results.items():
    d_lin = data['D_lin_mean'] * 1e8
    d_quad = data['D_quad_mean'] * 1e8
    diff = abs(d_lin - d_quad) / d_lin * 100
    print(f"{name:<18} {d_lin:<16.2f} {d_quad:<18.2f} {diff:<12.1f}")
print("-" * 64)