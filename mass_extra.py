import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

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

# Перевод в мг и секунды
time_s = time_min * 60
medny_mg = medny_g * 1000
serebro_mg = serebro_g * 1000
faki_mg = faki_g * 1000
apelsin_mg = apelsin_g * 1000

names = ['Медный', 'Серебро', 'Факи чемпион', 'Апельсин']
data_series = [medny_mg, serebro_mg, faki_mg, apelsin_mg]
colors = ['brown', 'gray', 'blue', 'orange']


# Модели для аппроксимации
def linear_model(t, a, b):
    return a * t + b


def quadratic_model(t, a, b, c):
    return a * t ** 2 + b * t + c


def cubic_model(t, a, b, c, d):
    return a * t ** 3 + b * t ** 2 + c * t + d


def exponential_model(t, a, b, c):
    return a * np.exp(b * t) + c


def logarithmic_model(t, a, b, c):
    return a * np.log(t + 1) + c  # +1 чтобы избежать log(0)


def power_model(t, a, b, c):
    return a * t ** b + c


def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


# Функция анализа всех моделей
def analyze_all_models(time_s, mass_mg, name):
    print(f"\n{'=' * 70}")
    print(f"Анализ для: {name}")
    print(f"{'=' * 70}")

    models = {}

    # 1. Линейная
    try:
        popt, _ = curve_fit(linear_model, time_s, mass_mg)
        y_pred = linear_model(time_s, *popt)
        r2 = calculate_r2(mass_mg, y_pred)
        models['linear'] = {'params': popt, 'r2': r2, 'func': linear_model}
        print(f"Линейная: m(t) = {popt[0]:.4f}·t + {popt[1]:.2f} мг, R² = {r2:.6f}")
    except:
        pass

    # 2. Квадратичная
    try:
        popt, _ = curve_fit(quadratic_model, time_s, mass_mg)
        y_pred = quadratic_model(time_s, *popt)
        r2 = calculate_r2(mass_mg, y_pred)
        models['quadratic'] = {'params': popt, 'r2': r2, 'func': quadratic_model}
        print(f"Квадратичная: m(t) = {popt[0]:.6f}·t² + {popt[1]:.4f}·t + {popt[2]:.2f} мг, R² = {r2:.6f}")
    except:
        pass

    # 3. Кубическая
    try:
        popt, _ = curve_fit(cubic_model, time_s, mass_mg)
        y_pred = cubic_model(time_s, *popt)
        r2 = calculate_r2(mass_mg, y_pred)
        models['cubic'] = {'params': popt, 'r2': r2, 'func': cubic_model}
        print(
            f"Кубическая: m(t) = {popt[0]:.8f}·t³ + {popt[1]:.6f}·t² + {popt[2]:.4f}·t + {popt[3]:.2f} мг, R² = {r2:.6f}")
    except:
        pass

    # 4. Экспоненциальная
    try:
        p0 = [mass_mg[0], 0.0001, 0]
        popt, _ = curve_fit(exponential_model, time_s, mass_mg, p0=p0, maxfev=10000)
        y_pred = exponential_model(time_s, *popt)
        r2 = calculate_r2(mass_mg, y_pred)
        models['exponential'] = {'params': popt, 'r2': r2, 'func': exponential_model}
        print(f"Экспоненциальная: m(t) = {popt[0]:.2f}·exp({popt[1]:.6f}·t) + {popt[2]:.2f} мг, R² = {r2:.6f}")
    except:
        print("Экспоненциальная: не удалось подобрать")

    # 5. Логарифмическая
    try:
        popt, _ = curve_fit(logarithmic_model, time_s, mass_mg)
        y_pred = logarithmic_model(time_s, *popt)
        r2 = calculate_r2(mass_mg, y_pred)
        models['logarithmic'] = {'params': popt, 'r2': r2, 'func': logarithmic_model}
        print(f"Логарифмическая: m(t) = {popt[0]:.2f}·ln(t+1) + {popt[2]:.2f} мг, R² = {r2:.6f}")
    except:
        print("Логарифмическая: не удалось подобрать")

    # 6. Степенная
    try:
        p0 = [1, 0.5, mass_mg[0]]
        popt, _ = curve_fit(power_model, time_s, mass_mg, p0=p0, maxfev=10000)
        y_pred = power_model(time_s, *popt)
        r2 = calculate_r2(mass_mg, y_pred)
        models['power'] = {'params': popt, 'r2': r2, 'func': power_model}
        print(f"Степенная: m(t) = {popt[0]:.2f}·t^{popt[1]:.4f} + {popt[2]:.2f} мг, R² = {r2:.6f}")
    except:
        print("Степенная: не удалось подобрать")

    # Определение лучшей модели
    if models:
        best_model = max(models, key=lambda k: models[k]['r2'])
        print(f"\n>>> Лучшая модель: {best_model.upper()} (R² = {models[best_model]['r2']:.6f})")

    return models


# Анализ всех серий
all_results = {}
for mass, name in zip(data_series, names):
    all_results[name] = analyze_all_models(time_s, mass, name)

# Визуализация - все модели для каждой серии
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, (mass, name, color) in enumerate(zip(data_series, names, colors)):
    ax = axes[i]

    # Исходные данные
    ax.scatter(time_s, mass, c='black', alpha=0.7, label='Эксперимент', s=40, zorder=5)

    # Построение всех моделей
    models = all_results[name]
    for model_name, model_data in models.items():
        t_fit = np.linspace(time_s.min(), time_s.max(), 500)
        y_fit = model_data['func'](t_fit, *model_data['params'])
        ax.plot(t_fit, y_fit, linewidth=2, label=f"{model_name}: R²={model_data['r2']:.4f}")

    ax.set_xlabel('Время, с', fontsize=12)
    ax.set_ylabel('Масса, мг', fontsize=12)
    ax.set_title(f'{name} - Сравнение моделей', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# График производных для лучшей модели
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, (mass, name, color) in enumerate(zip(data_series, names, colors)):
    ax = axes[i]

    models = all_results[name]
    best_model_name = max(models, key=lambda k: models[k]['r2'])
    best_model = models[best_model_name]

    # Численная производная экспериментальных данных
    mass_smooth = gaussian_filter1d(mass, sigma=1)
    derivative_exp = np.gradient(mass_smooth, time_s)
    ax.scatter(time_s, derivative_exp, c='black', alpha=0.5, label='Эксп. производная', s=20)

    # Аналитическая производная лучшей модели
    t_fit = np.linspace(time_s.min(), time_s.max(), 500)
    y_fit = best_model['func'](t_fit, *best_model['params'])
    derivative_fit = np.gradient(y_fit, t_fit)
    ax.plot(t_fit, derivative_fit, '-', c=color, linewidth=2, label=f"Производная {best_model_name}")

    ax.set_xlabel('Время, с', fontsize=12)
    ax.set_ylabel('dm/dt, мг/с', fontsize=12)
    ax.set_title(f'{name}\nЛучшая: {best_model_name} (R²={best_model["r2"]:.4f})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Сводная таблица
print("\n" + "=" * 80)
print("СВОДНАЯ ТАБЛИЦА ЛУЧШИХ МОДЕЛЕЙ")
print("=" * 80)
print(f"{'Серия':<15} {'Лучшая модель':<15} {'R²':<10} {'Параметры'}")
print("-" * 80)

for name in names:
    models = all_results[name]
    best_model_name = max(models, key=lambda k: models[k]['r2'])
    best_model = models[best_model_name]
    params_str = ', '.join([f"{p:.4f}" for p in best_model['params']])
    print(f"{name:<15} {best_model_name:<15} {best_model['r2']:<10.6f} {params_str}")