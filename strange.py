import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit
import pandas as pd

# === НАСТРОЙКИ ===
rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 9
rcParams['axes.labelsize'] = 10
rcParams['axes.titlesize'] = 11
rcParams['legend.fontsize'] = 8
rcParams['figure.dpi'] = 120

# === ФИЗИЧЕСКИЕ КОНСТАНТЫ ===
P_0_mmHg = 742
P_0_Pa = P_0_mmHg * 133.322
T_room = 298
T_cold = 269
R = 8.314
mu_air = 28.97e-3
mu_He = 4.00e-3

shell_rho1 = 0.910
shell_rho2 = 0.934

rho_diff_room = (P_0_Pa / (R * T_room)) * (mu_air - mu_He) / 1000
rho_diff_cold = (P_0_Pa / (R * T_cold)) * (mu_air - mu_He) / 1000

# ============================================================
# ДАННЫЕ: ГРУППА 1
# ============================================================
time_g1 = np.array([
    0.0, 0.133, 0.85, 0.9, 1.033, 1.133, 1.183, 1.283, 1.333, 1.4,
    1.45, 1.533, 1.583, 1.667, 1.733, 1.817, 1.9, 2.033, 2.083, 2.15,
    2.25, 2.333, 2.433, 2.5, 2.55, 2.633, 2.7, 2.767, 2.867, 2.933,
    3.0, 3.067, 3.183, 3.267, 3.367, 3.433, 3.5, 3.583, 3.65, 3.783,
    3.85, 4.05, 4.15, 4.583, 4.75, 5.183, 5.233, 5.583, 5.917, 6.2, 6.717
])

mass_med = np.array([12.7, 12.9, 13.7, 13.9, 14.0, 14.2, 14.3, 14.4, 14.5, 14.6,
                     14.7, 14.8, 14.9, 15.0, 15.1, 15.2, 15.2, 15.3, 15.5, 15.6,
                     15.7, 15.8, 15.9, 16.0, 16.1, 16.2, 16.3, 16.4, 16.5, 16.6,
                     16.7, 16.8, 16.8, 16.9, 17.0, 17.0, 17.1, 17.2, 17.3, 17.4,
                     17.5, 17.6, 17.8, 18.3, 18.3, 18.8, 18.8, 19.1, 19.3, 19.5, 19.9])

mass_ser = np.array([16.6, 16.9, 17.7, 17.7, 18.0, 18.0, 18.1, 18.2, 18.3, 18.4,
                     18.4, 18.5, 18.7, 18.8, 18.8, 18.9, 19.0, 19.1, 19.1, 19.3,
                     19.4, 19.5, 19.6, 19.6, 19.7, 19.7, 19.9, 19.9, 20.1, 20.1,
                     20.2, 20.3, 20.4, 20.5, 20.5, 20.6, 20.7, 20.7, 20.8, 20.9,
                     20.9, 21.1, 21.2, 21.6, 21.7, 22.0, 22.0, 22.3, 22.6, 22.6, 23.0])

mass_faki = np.array([16.1, 16.3, 17.0, 17.1, 17.1, 17.4, 17.5, 17.5, 17.5, 17.6,
                      17.7, 17.7, 17.8, 17.9, 18.0, 18.1, 18.2, 18.3, 18.3, 18.4,
                      18.5, 18.6, 18.7, 18.7, 18.8, 18.9, 19.0, 19.1, 19.1, 19.2,
                      19.2, 19.3, 19.5, 19.6, 19.7, 19.7, 19.7, 19.8, 19.9, 19.9,
                      20.1, 20.2, 20.2, 20.8, 20.8, 21.1, 21.2, 21.4, 21.5, 21.8, 22.0])

time_apel = np.array([0.0, 0.133, 0.85, 0.9, 1.033, 1.133, 1.183, 1.283, 1.333, 1.4,
                      1.45, 1.533, 1.583, 1.667, 1.733, 1.817, 1.9, 2.033, 2.083, 2.15,
                      2.25, 2.333, 2.433, 2.5, 2.55, 2.633, 2.7, 2.767, 2.867, 2.933,
                      3.0, 3.067, 3.183, 3.267, 3.367, 3.433, 3.5, 3.583, 3.65, 3.783,
                      3.85, 4.05, 4.15, 4.583, 4.75, 5.183, 5.233, 5.583, 5.917])

mass_apel = np.array([16.1, 16.4, 17.2, 17.3, 17.5, 17.5, 17.6, 17.7, 17.7, 17.8,
                      17.9, 18.0, 18.1, 18.2, 18.3, 18.3, 18.4, 18.5, 18.5, 18.6,
                      18.7, 18.9, 18.9, 19.0, 19.1, 19.2, 19.3, 19.3, 19.4, 19.5,
                      19.6, 19.7, 19.7, 19.8, 19.8, 19.9, 20.0, 20.0, 20.1, 20.2,
                      20.3, 20.4, 20.4, 20.9, 20.9, 21.0, 21.1, 21.3, 21.3])

group1 = {
    'Медный': {'time': time_g1, 'mass': mass_med, 'm_ob': 2.0, 'rho_ob': shell_rho1,
               'S0': 4172.1, 'kS': -222.983, 'color': '#CD7F32', 'rho_diff': rho_diff_room},
    'Серебро': {'time': time_g1, 'mass': mass_ser, 'm_ob': 2.3, 'rho_ob': shell_rho1,
                'S0': 3250.8, 'kS': -198.594, 'color': '#708090', 'rho_diff': rho_diff_room},
    'Факи Чемпион': {'time': time_g1, 'mass': mass_faki, 'm_ob': 2.1, 'rho_ob': shell_rho2,
                     'S0': 3562.1, 'kS': -236.144, 'color': '#0055A4', 'rho_diff': rho_diff_room},
    'Апельсин': {'time': time_apel, 'mass': mass_apel, 'm_ob': 2.2, 'rho_ob': shell_rho2,
                 'S0': 2497.7, 'kS': -261.805, 'color': '#FF9F43', 'rho_diff': rho_diff_room}
}

# ============================================================
# ДАННЫЕ: ГРУППА 2
# ============================================================
time_maly = np.array([0, 1.5, 2, 4, 5.5, 7, 8, 9, 10.5, 11.5, 14.5, 23.5, 27.5, 37.5])
mass_maly = np.array([7.5, 7.5, 7.6, 7.6, 7.6, 7.7, 7.8, 7.8, 7.8, 7.9, 7.9, 8.1, 8.2, 8.3])

time_sred = np.array(
    [0, 1, 2, 10.5, 12, 12.5, 14.5, 16, 18.5, 19.5, 21, 22, 25, 34, 38, 48, 58, 62.5, 65.5, 70, 72, 81, 89, 94.5, 105.5,
     113, 119.5, 128.5, 144, 152])
mass_sred = np.array(
    [3.8, 3.8, 3.9, 4.1, 4.2, 4.2, 4.3, 4.3, 4.3, 4.4, 4.4, 4.4, 4.5, 4.6, 4.7, 4.9, 5.0, 5.1, 5.1, 5.2, 5.3, 5.6, 5.6,
     5.7, 5.8, 6.0, 6.1, 6.2, 6.4, 6.6])

time_bolsh = np.array(
    [0, 1, 1.5, 3.5, 5, 6.5, 7.5, 9, 10, 11, 14, 23, 27, 37, 47, 51.5, 54.5, 59, 61, 70, 78, 83.5, 94.5, 102, 108.5,
     117.5, 133, 141])
mass_bolsh = np.array(
    [12.6, 12.8, 12.8, 12.9, 12.9, 13.0, 13.0, 13.0, 13.1, 13.0, 13.1, 13.2, 13.2, 13.3, 13.5, 13.6, 13.6, 13.8, 13.8,
     13.8, 14.0, 14.0, 14.1, 14.2, 14.3, 14.4, 14.6, 14.8])

time_holod = np.array(
    [0, 0.5, 2, 3, 5.5, 6.5, 9.5, 18.5, 22.5, 32.5, 42.5, 47, 50, 54.5, 56.5, 65.5, 73.5, 79, 89.5, 97, 103.5, 112.5,
     128, 135.5])
mass_holod = np.array(
    [8.8, 8.9, 9.0, 9.0, 9.0, 9.1, 9.0, 9.1, 9.1, 9.3, 9.3, 9.3, 9.3, 9.4, 9.5, 9.5, 9.5, 9.6, 9.6, 9.8, 9.8, 9.9, 10.0,
     10.1])

group2 = {
    'Малый': {'time': time_maly, 'mass': mass_maly, 'm_ob': 4.48, 'rho_ob': shell_rho2,
              'S0': 1825.89, 'kS': -6.355, 'color': 'red', 'rho_diff': rho_diff_room},
    'Средний': {'time': time_sred, 'mass': mass_sred, 'm_ob': 4.38, 'rho_ob': shell_rho2,
                'S0': 2577.72, 'kS': -8.458, 'color': '#FF9400', 'rho_diff': rho_diff_room},
    'Большой': {'time': time_bolsh, 'mass': mass_bolsh, 'm_ob': 4.05, 'rho_ob': shell_rho2,
                'S0': 2750.0, 'kS': -10.702, 'color': 'green', 'rho_diff': rho_diff_room},
    'Холодный': {'time': time_holod, 'mass': mass_holod, 'm_ob': 4.3, 'rho_ob': shell_rho2,
                 'S0': 1721.48, 'kS': -7.951, 'color': 'blue', 'rho_diff': rho_diff_cold}
}


# ============================================================
# МОДЕЛИ
# ============================================================
def model_exp(t, k, m_inf, m_0):
    """Экспоненциальная модель. k в 1/с, t в часах"""
    return m_inf - (m_inf - m_0) * np.exp(-k * t * 3600)


def model_cubic(t, alpha, S0, kS, m_0):
    """Кубическая модель. alpha в г/(см·ч), t в часах"""
    return m_0 + (alpha / (3 * kS)) * ((S0 + kS * t) ** 3 - S0 ** 3)


# ============================================================
# ЗНАЧЕНИЯ ИЗ КАРТИНОК (экспоненциальная модель)
# ============================================================
D_exp_from_pics = {
    'Медный': (7.5e-8, 0.5e-8),
    'Серебро': (18.6e-8, 1.2e-8),
    'Факи Чемпион': (11.4e-8, 0.7e-8),
    'Апельсин': (15.3e-8, 1.0e-8),
    'Малый': (0.94e-8, 0.08e-8),
    'Средний': (0.41e-8, 0.03e-8),
    'Большой': (0.27e-8, 0.02e-8),
    'Холодный': (0.35e-8, 0.02e-8)
}


# ============================================================
# ФУНКЦИЯ АНАЛИЗА
# ============================================================
def analyze_ball(name, p):
    t = p['time']
    m = p['mass']
    m_0 = m[0]
    S_t = p['S0'] + p['kS'] * t

    results = {}

    # === МЕТОД 1: ЭКСПОНЕНЦИАЛЬНЫЙ (берём из картинок) ===
    D_exp_mean, D_exp_std = D_exp_from_pics[name]
    results['exp'] = {'D_mean': D_exp_mean, 'D_std': D_exp_std}

    # === МЕТОД 2: ПРЯМОЙ РАСЧЁТ ===
    dm_dt_data = np.gradient(m, t) / 3600  # г/с
    D_data_t = (dm_dt_data * p['m_ob']) / (S_t ** 2 * p['rho_ob'] * p['rho_diff'])

    valid = (D_data_t > 0) & (D_data_t < 1e-5)
    D_data_valid = D_data_t[valid]

    D_data_mean = np.mean(D_data_valid) if len(D_data_valid) > 0 else np.nan
    D_data_std = np.std(D_data_valid, ddof=1) if len(D_data_valid) > 1 else np.nan

    results['direct'] = {'D_mean': D_data_mean, 'D_std': D_data_std}

    # === МЕТОД 3: КУБИЧЕСКАЯ МОДЕЛЬ ===
    try:
        # Начальное приближение
        alpha_guess = 1e-10

        popt_cub, pcov_cub = curve_fit(
            lambda t, a: model_cubic(t, a, p['S0'], p['kS'], m_0),
            t, m, p0=[alpha_guess],
            bounds=([0], [np.inf])
        )
        alpha_cub = popt_cub[0]

        # Погрешность alpha из ковариационной матрицы
        sigma_alpha = np.sqrt(np.diag(pcov_cub))[0] if pcov_cub is not None else 0

        # D = alpha * m_ob / (rho_ob * delta_rho)
        # alpha в г/(см⁴·ч), переводим в г/(см·с)
        alpha_cub_sec = alpha_cub / 3600
        sigma_alpha_sec = sigma_alpha / 3600

        D_cub = alpha_cub_sec * p['m_ob'] / (p['rho_ob'] * p['rho_diff'])

        # Погрешность D из погрешности alpha
        sigma_D_cub = sigma_alpha_sec * p['m_ob'] / (p['rho_ob'] * p['rho_diff'])

        results['cubic'] = {'D_mean': D_cub, 'D_std': sigma_D_cub, 'alpha': alpha_cub}
    except Exception as e:
        results['cubic'] = {'D_mean': np.nan, 'D_std': np.nan, 'alpha': np.nan}

    return results


# ============================================================
# ФУНКЦИЯ ПОСТРОЕНИЯ ГРАФИКОВ
# ============================================================
def plot_group(group, group_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    all_results = {}

    for name, p in group.items():
        results = analyze_ball(name, p)
        all_results[name] = results

        t = p['time']
        m = p['mass']
        m_0 = m[0]
        S_t = p['S0'] + p['kS'] * t

        # === График 1: m(t) с двумя моделями ===
        axes[0].plot(t, m, 'o', color=p['color'], markersize=4, alpha=0.7)

        # Экспоненциальная модель
        if not np.isnan(results['exp']['D_mean']):
            # Для графика m(t) используем подгонку экспоненты
            try:
                k_guess = 1e-5 if name in ['Медный', 'Серебро', 'Факи Чемпион', 'Апельсин'] else 1e-7
                m_inf_guess = m[-1] + 0.5 * (m[-1] - m[0])
                popt_exp, _ = curve_fit(
                    lambda t, k, m_inf: model_exp(t, k, m_inf, m_0),
                    t, m, p0=[k_guess, m_inf_guess],
                    bounds=([0, m_0], [1e-3, m_0 + 20])
                )
                k_exp, m_inf_exp = popt_exp
                t_smooth = np.linspace(t[0], t[-1], 200)
                m_exp_smooth = model_exp(t_smooth, k_exp, m_inf_exp, m_0)
                axes[0].plot(t_smooth, m_exp_smooth, '--', color=p['color'], linewidth=1.2,
                             alpha=0.7, label=f"{name} (эксп)")
            except:
                pass

        # Кубическая модель
        if not np.isnan(results['cubic']['D_mean']):
            alpha_cub = results['cubic']['alpha']
            t_smooth = np.linspace(t[0], t[-1], 200)
            m_cub_smooth = model_cubic(t_smooth, alpha_cub, p['S0'], p['kS'], m_0)
            axes[0].plot(t_smooth, m_cub_smooth, '-', color=p['color'], linewidth=1.5,
                         label=f"{name} (куб): D={results['cubic']['D_mean'] * 1e8:.2f}±{results['cubic']['D_std'] * 1e8:.2f}")

        # === График 2: D(t) из кубической модели ===
        if not np.isnan(results['cubic']['D_mean']):
            alpha_cub = results['cubic']['alpha']
            dm_dt_cub = alpha_cub * (p['S0'] + p['kS'] * t) ** 2 / 3600  # г/с
            D_cub_t = (dm_dt_cub * p['m_ob']) / (S_t ** 2 * p['rho_ob'] * p['rho_diff'])

            axes[1].plot(t, D_cub_t * 1e8, 'o', color=p['color'], markersize=4, alpha=0.7,
                         label=f"{name}: D={results['cubic']['D_mean'] * 1e8:.2f}±{results['cubic']['D_std'] * 1e8:.2f}")
            axes[1].axhline(y=results['cubic']['D_mean'] * 1e8, color=p['color'],
                            linestyle=':', linewidth=1, alpha=0.7)

    # Оформление
    axes[0].set_xlabel('Время, ч')
    axes[0].set_ylabel('Масса, г')
    axes[0].set_title(f'{group_name}: m(t) — сравнение моделей')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best', fontsize=7)

    axes[1].set_xlabel('Время, ч')
    axes[1].set_ylabel(r'$D$, $10^{-8}$ см²/с')
    axes[1].set_title(f'{group_name}: D(t) из кубической модели')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best', fontsize=7)

    plt.tight_layout()
    plt.show()

    return all_results


# ============================================================
# ПОСТРОЕНИЕ ГРАФИКОВ
# ============================================================
results_g1 = plot_group(group1, 'Группа 1')
results_g2 = plot_group(group2, 'Группа 2')

# ============================================================
# ТАБЛИЦА РЕЗУЛЬТАТОВ
# ============================================================
table_data = []

for group_name, results in [('Группа 1', results_g1), ('Группа 2', results_g2)]:
    for name, res in results.items():
        row = {
            'Группа': group_name,
            'Шарик': name,
            'D_эксп (10⁻⁸)': f"{res['exp']['D_mean'] * 1e8:.2f} ± {res['exp']['D_std'] * 1e8:.2f}",
            'D_прямой (10⁻⁸)': f"{res['direct']['D_mean'] * 1e8:.2f} ± {res['direct']['D_std'] * 1e8:.2f}" if not np.isnan(
                res['direct']['D_mean']) else "NaN",
            'D_кубический (10⁻⁸)': f"{res['cubic']['D_mean'] * 1e8:.2f} ± {res['cubic']['D_std'] * 1e8:.2f}" if not np.isnan(
                res['cubic']['D_mean']) else "NaN"
        }
        table_data.append(row)

df = pd.DataFrame(table_data)
df.to_csv('diffusion_results_comparison.csv', index=False, encoding='utf-8-sig')

print("\n" + "=" * 100)
print("СРАВНЕНИЕ ТРЁХ МЕТОДОВ РАСЧЁТА КОЭФФИЦИЕНТА ДИФФУЗИИ")
print("=" * 100)
print(df.to_string(index=False))
print("=" * 100)
print("\nРезультаты сохранены в файл: diffusion_results_comparison.csv")
print("=" * 100)