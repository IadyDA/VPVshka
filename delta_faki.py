import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# === НАСТРОЙКИ ГРАФИКА ===
rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 9
rcParams['axes.labelsize'] = 10
rcParams['axes.titlesize'] = 11
rcParams['legend.fontsize'] = 8
rcParams['figure.dpi'] = 120

# === ФИЗИЧЕСКИЕ КОНСТАНТЫ ===
P_0_mmHg = 742
P_0_Pa = P_0_mmHg * 133.322
R = 8.314
mu_air = 28.97e-3
mu_He = 4.00e-3

# === ТЕМПЕРАТУРЫ (К) ===
# Холодный шарик: -4°C = 269.15 K, остальные: 298 K
temperatures = {
    'Малый': 298.0,
    'Средний': 298.0,
    'Большой': 298.0,
    'Холодный': 269.15  # -4°C ± 0.3°C
}

# === ПЛОТНОСТИ ===
rho_rubber = {
    'Малый': 0.942,
    'Средний': 0.942,
    'Большой': 0.942,
    'Холодный': 0.942
}

# Разность плотностей воздуха и гелия для каждой температуры (г/см³)
rho_diff = {}
for name, T in temperatures.items():
    rho_diff[name] = (P_0_Pa / (R * T)) * (mu_air - mu_He) / 1000

# === МАССЫ ОБОЛОЧЕК (г) ===
# ⚠️ Если у вас есть точные значения — замените!
shell_masses = {
    'Малый': 4.48,
    'Средний': 4.38,
    'Большой': 4.05,
    'Холодный': 4.3
}

# === ПАРАМЕТРЫ ПЛОЩАДИ S(t) = S_0 + k_S·t (t в минутах) ===
S_params = {
    'Малый': {'S_0': 1783.74, 'k_S': -0.1061},
    'Средний': {'S_0': 2468.01, 'k_S': -0.14},
    'Большой': {'S_0': 2717.43, 'k_S': -0.1791},
    'Холодный': {'S_0': 1580.35, 'k_S': -0.1334}
}

# === ЭКСПОНЕНЦИАЛЬНЫЕ ПАРАМЕТРЫ m(t) ===
# Δm в мг, k в с⁻¹, τ в минутах
exp_params = {
    'Малый': {'delta_m_mg': 5425.09, 'k_s': 1.26e-6, 'tau': 13197.4},
    'Средний': {'delta_m_mg': 10291.34, 'k_s': 5.5e-7, 'tau': 30433.0},
    'Большой': {'delta_m_mg': 13211.51, 'k_s': 3.1e-7, 'tau': 54194.7},
    'Холодный': {'delta_m_mg': 7069.59, 'k_s': 3.5e-7, 'tau': 46949.1}
}

# Цвета и стили
styles = {
    'Малый': {'color': '#E74C3C', 'linestyle': '-', 'marker': 'o'},
    'Средний': {'color': 'orange', 'linestyle': '--', 'marker': 's'},
    'Большой': {'color': '#2ECC71', 'linestyle': '-.', 'marker': '^'},
    'Холодный': {'color': 'blue', 'linestyle': ':', 'marker': 'd'}
}

# Ограничение времени для графика (τ очень большие — до 54000 мин)
T_MAX_PLOT = 3000


def calculate_D(t_sec, name):
    m_shell = shell_masses[name]
    S_p = S_params[name]
    E_p = exp_params[name]
    rho = rho_rubber[name]
    rd = rho_diff[name]  # 🔹 индивидуальная разность плотностей

    t_min = t_sec / 60
    # 🔹 Убран сдвиг -37: S(t) = S_0 + k_S·t
    S = S_p['S_0'] + S_p['k_S'] * t_min
    if S <= 100:
        return None, None

    delta = m_shell / (rho * S)

    delta_m_g = E_p['delta_m_mg'] / 1000
    k_s = E_p['k_s']
    beta = delta_m_g * k_s * np.exp(-k_s * t_sec)

    D = (beta * delta) / (S * rd)
    return D, S


# === РАСЧЁТ ===
print("📊 Расчёт D(t) для второй выборки шариков...")
print(f"{'Шарик':<12} {'T, K':<8} {'ρ_diff, г/см³':<16} {'τ, мин':<10} {'⟨D⟩, 10⁻⁸ см²/с':<18}")
print("-" * 70)

results = {}

for name in shell_masses.keys():
    # Ограничиваем время до T_MAX_PLOT или τ (что меньше)
    t_max = min(T_MAX_PLOT, exp_params[name]['tau'])
    t_minutes = np.linspace(0, t_max, 600)  # 🔹 увеличено с 300 до 600 точек
    t_seconds = t_minutes * 60

    D_vals = []
    for t_sec in t_seconds:
        D, S = calculate_D(t_sec, name)
        if D is not None and 0 < D < 1e-4:
            D_vals.append(D)

    if D_vals:
        D_vals = np.array(D_vals)
        results[name] = {
            't': t_minutes[:len(D_vals)],
            'D': D_vals,
            'D_mean': np.mean(D_vals),
            'D_std': np.std(D_vals),
            'tau': exp_params[name]['tau']
        }
        print(f"{name:<12} {temperatures[name]:<8.1f} {rho_diff[name]:<16.5f} "
              f"{exp_params[name]['tau']:<10.0f} {np.mean(D_vals) * 1e8:<18.3f}")

print("-" * 70)

# === ОДИН ОБЩИЙ ГРАФИК ===
fig, ax = plt.subplots(figsize=(10, 6))

for name in results.keys():
    data = results[name]
    st = styles[name]

    # Основная кривая
    ax.plot(data['t'], data['D'] * 1e8,
            color=st['color'], linestyle=st['linestyle'],
            marker=st['marker'], markevery=60, markersize=5,  # 🔹 markevery увеличен
            linewidth=1.8, label=name, alpha=0.9)

    # Область неопределённости
    ax.fill_between(data['t'],
                    np.maximum(0, (data['D'] - data['D_std']) * 1e8),
                    (data['D'] + data['D_std']) * 1e8,
                    color=st['color'], alpha=0.15)

    # Горизонтальная линия среднего
    ax.axhline(data['D_mean'] * 1e8, color=st['color'], linestyle=':',
               linewidth=0.7, alpha=0.5)

# Оформление
ax.set_xlabel('Время, мин', fontsize=10)
ax.set_ylabel('Коэффициент диффузии D, 10⁻⁸ см²/с', fontsize=10)
ax.set_title(f'Зависимость D(t) для второй выборки шариков\n'
             f'(интервал {T_MAX_PLOT} мин ≈ {T_MAX_PLOT/60:.0f} ч)',
            fontsize=12, pad=15)
ax.grid(True, linestyle=':', alpha=0.5)
ax.legend(frameon=True, fancybox=True, loc='upper right', fontsize=8)
ax.set_xlim(0, T_MAX_PLOT)
ax.set_axisbelow(True)

plt.show()

# === ИТОГОВАЯ ТАБЛИЦА ===
print("\n📋 Сводная таблица:")
print(f"{'Шарик':<12} {'T, K':<8} {'⟨D⟩, 10⁻⁸ см²/с':<18} {'σ, 10⁻⁸ см²/с':<18} {'τ, мин':<10}")
print("-" * 66)
for name in results.keys():
    d = results[name]
    rel = d['D_std'] / d['D_mean'] * 100
    print(f"{name:<12} {temperatures[name]:<8.1f} {d['D_mean'] * 1e8:<18.3f} "
          f"{d['D_std'] * 1e8:<18.3f} {d['tau']:<10.0f} (σ/⟨D⟩={rel:.1f}%)")
print("-" * 66)

# === ПРОВЕРКА D·S² ===
print("\n🔍 Проверка нормировки D·S²:")
print(f"{'Шарик':<12} {'S_ср, см²':<12} {'D·S², 10⁻² см⁴/с':<20}")
print("-" * 44)
for name in results.keys():
    S_0 = S_params[name]['S_0']
    k_S = S_params[name]['k_S']
    t_max = min(T_MAX_PLOT, exp_params[name]['tau'])
    S_avg = S_0 + k_S * t_max / 2
    D_mean = results[name]['D_mean']
    print(f"{name:<12} {S_avg:<12.0f} {D_mean * S_avg ** 2 * 1e2:<20.2f}")