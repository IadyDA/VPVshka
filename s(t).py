import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib import rcParams

# === НАСТРОЙКИ ГРАФИКА (для отчёта/презентации) ===
rcParams['font.family'] = 'DejaVu Sans'  # или 'Arial', 'Times New Roman'
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 14
rcParams['legend.fontsize'] = 10
rcParams['figure.dpi'] = 150

# === ДАННЫЕ ===
time = np.array([0, 87, 140, 233, 386])  # минуты

# Площадь поверхности в см² (из ваших расчётов)
areas = {
    'Медный': np.array([4172.1, 3915.12, 3686.5, 3453.4, 2755.7]),
    'Серебро': np.array([3250.8, 3040.5, 2905.6, 2792.7, 2010.1]),
    'Факи Чемпион': np.array([3562.1, 3442.9, 3208.3, 2987.2, 2146.5]),
    'Апельсин': np.array([2496.7, 2322.1, 1947.2, 1763.1, 915.9])
}

# Погрешность 3% для каждого значения
error_percent = 0.03
errors = {name: area * error_percent for name, area in areas.items()}

# Цвета для графиков
colors = {'Медный': '#CD7F32', 'Серебро': '#C0C0C0',
          'Факи Чемпион': '#0055A4', 'Апельсин': '#FF9F43'}


# === ФУНКЦИЯ ЛИНЕЙНОЙ РЕГРЕССИИ С ВЗВЕШИВАНИЕМ ===
def weighted_linear_fit(x, y, yerr):
    """Взвешенный МНК: возвращает k, b, их погрешности и R²"""
    weights = 1 / yerr ** 2
    W = np.sum(weights)
    Wx = np.sum(weights * x)
    Wxx = np.sum(weights * x ** 2)
    Wy = np.sum(weights * y)
    Wxy = np.sum(weights * x * y)

    denom = W * Wxx - Wx ** 2
    k = (W * Wxy - Wx * Wy) / denom
    b = (Wxx * Wy - Wx * Wxy) / denom

    # Погрешности коэффициентов
    dk = np.sqrt(W / denom)
    db = np.sqrt(Wxx / denom)

    # R² (взвешенный)
    y_pred = k * x + b
    ss_res = np.sum(weights * (y - y_pred) ** 2)
    ss_tot = np.sum(weights * (y - np.average(y, weights=weights)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return k, b, dk, db, r2


# === АНАЛИЗ И ВЫВОД ===
print("=" * 70)
print("📊 ЛИНЕЙНЫЙ АНАЛИЗ: Площадь поверхности от времени")
print("=" * 70)
print(f"Погрешность каждого измерения: ±{error_percent * 100:.1f}%\n")

results = {}

for name, area in areas.items():
    k, b, dk, db, r2 = weighted_linear_fit(time, area, errors[name])
    results[name] = {'k': k, 'b': b, 'dk': dk, 'db': db, 'r2': r2}

    # Формула с нулём в t=0
    print(f"🎈 {name}:")
    print(f"   S(t) = ({b:.1f} ± {db:.1f}) + ({k:.3f} ± {dk:.3f})·t   [см²],  t в мин")
    print(f"   R² = {r2:.4f}  {'✅ Линейность подтверждена' if r2 > 0.98 else '⚠️ Возможна нелинейность'}")

    # Формула со сдвигом нуля на 37 минут: S = b + k·t = (b + k·37) + k·(t - 37)
    b_shifted = b + k * 37
    print(f"   При t₀ = 37 мин: S(t) = ({b_shifted:.1f} ± {db:.1f}) + ({k:.3f} ± {dk:.3f})·(t - 37)")
    print()

print("=" * 70)
print("📈 СРАВНЕНИЕ СКОРОСТЕЙ УМЕНЬШЕНИЯ ПЛОЩАДИ (коэффициент k):")
print("=" * 70)
print("📈 СРАВНЕНИЕ СКОРОСТЕЙ УМЕНЬШЕНИЯ ПЛОЩАДИ (коэффициент k):")
# Сортируем по абсолютному значению k (от самого быстрого уменьшения к медленному)
sorted_results = sorted(results.items(), key=lambda x: abs(x[1]['k']))

for name, res in sorted_results:
    print(f"   {name:15s}: k = {res['k']:.3f} ± {res['dk']:.3f} см²/мин  (R²={res['r2']:.3f})")
print("\n⚠️ Отрицательный k означает уменьшение площади со временем (утечка гелия).")
print("=" * 70)
for name, res in sorted_results:
    print(f"   {name:15s}: k = {res['k']:.3f} ± {res['dk']:.3f} см²/мин  (R²={res['r2']:.3f})")
print("\n⚠️ Отрицательный k означает уменьшение площади со временем (утечка гелия).")
print("=" * 70)

# === ГРАФИК ===
fig, ax = plt.subplots(figsize=(10, 6))

for name, area in areas.items():
    k, b, _, _, _ = results[name]['k'], results[name]['b'], None, None, None
    k, b, dk, db, r2 = results[name]['k'], results[name]['b'], results[name]['dk'], results[name]['db'], results[name][
        'r2']

    # Точки с погрешностями
    ax.errorbar(time, area, yerr=errors[name], fmt='o', color=colors[name],
                label=name, capsize=4, markersize=6, linewidth=1.5, alpha=0.9)

    # Линия регрессии (плавная)
    t_fit = np.linspace(time.min(), time.max(), 100)
    s_fit = k * t_fit + b
    ax.plot(t_fit, s_fit, '-', color=colors[name], linewidth=2, alpha=0.8)

# Оформление
ax.set_xlabel('Время, мин', fontsize=12)
ax.set_ylabel('Площадь поверхности, см²', fontsize=12)
ax.set_title('Зависимость площади поверхности шариков от времени', fontsize=14, pad=20)
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend(frameon=True, fancybox=True, shadow=False)
ax.set_axisbelow(True)

plt.tight_layout()
plt.show()

# === ДОПОЛНИТЕЛЬНО: проверка линейности через остатки ===
print("\n🔍 АНАЛИЗ ОСТАТКОВ (проверка линейности):")
for name, area in areas.items():
    k, b, _, _, r2 = results[name]['k'], results[name]['b'], None, None, results[name]['r2']
    residuals = area - (k * time + b)
    max_res = np.max(np.abs(residuals))
    mean_area = np.mean(area)
    print(f"{name:15s}: макс. отклонение от прямой = {max_res:.1f} см² "
          f"({max_res / mean_area * 100:.2f}% от среднего значения)")