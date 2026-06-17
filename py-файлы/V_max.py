import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

# ===== 1. ВАШИ ДАННЫЕ (в метрах) =====
# x — вдоль большой оси (ось вращения), y — радиус сечения
# x_data = np.array([6.993E-3, 3.060E-2, 5.887E-2, 0.101, 0.156,
#                    0.219, 0.282, 0.339, 0.377, 0.392])
# y_data = np.array([2.480E-2, 6.696E-2, 0.102, 0.135, 0.158,
#                    0.165, 0.154, 0.120, 7.100E-2, 0])

# ===== НОВЫЕ ДАННЫЕ (маленький шарик) =====
# x_data = np.array([2.331E-3, 1.388E-2, 3.929E-2, 7.470E-2, 0.114,
#                    0.149, 0.213, 0.254, 0.280, 0.288])
# y_data = np.array([2.560E-2, 5.450E-2, 8.699E-2, 0.112, 0.126,
#                    0.129, 0.116, 8.697E-2, 4.784E-2, 0])

# ===== НОВЫЕ ДАННЫЕ (средний шарик) =====
# x_data = np.array([5.99E-3, 2.026E-2, 3.530E-2, 6.6330E-2, 0.121,
#                    0.182, 0.249, 0.295, 0.317, 0.322])
# y_data = np.array([3.921E-2, 6.999E-2, 8.945E-2, 0.118, 0.144,
#                    0.151, 0.133, 9.656E-2, 5.478E-2, 0])

# ===== ДАННЫЕ (шарик при другой температуре) =====
# x_data = np.array([5.230E-3, 2.345E-2, 5.554E-2, 0.100, 0.150,
#                    0.211, 0.255, 0.279, 0.286, 0.288])
# y_data = np.array([4.311E-2, 8.211E-2, 0.111, 0.132, 0.138,
#                    0.122, 9.127E-2, 5.474E-2, 3.3E-2, 0])
#медный 1
# x_data = np.array([8.319E-3, 2.427E-2, 5.137E-2, 8.050E-2, 0.189,
#                    0.306, 0.407, 0.454, 0.478, 0.488])
# y_data = np.array([1.108E-2, 2.957E-2, 4.781E-2, 5.920E-2, 0.137,
#                    0.160, 0.131, 8.920E-2, 4.207E-2, 0])
#серебро 1
# x_data = np.array([1.047E-2, 3.437E-2, 8.860E-2, 0.158, 0.222,
#                    0.281, 0.329, 0.354, 0.365, 0.371])
# y_data = np.array([3.356E-2, 6.578E-2, 0.117, 0.148, 0.155,
#                    0.139, 0.104, 7.034E-2, 3.954E-2, 3.592E-4])
#синий 1
# x_data = np.array([5.001E-3, 2.776E-2, 6.255E-2, 0.127, 0.214,
#                    0.286, 0.334, 0.364, 0.379, 0.385])
# y_data = np.array([2.437E-2, 5.893E-2, 9.565E-2, 0.139, 0.157,
#                    0.143, 0.115, 7.981E-2, 4.577E-2, 4.546E-4])
#orange 1
# x_data = np.array([8.352E-3, 3.357E-2, 6.139E-2, 0.113, 0.164,
#                    0.215, 0.257, 0.277, 0.288, 0.290])
# y_data = np.array([3.783E-2, 7.439E-2, 0.102, 0.126, 0.131,
#                    0.120, 9.037E-2, 6.356E-2, 3.395E-2, 4.389E-4])
#медный 2
# x_data = np.array([2.200E-3, 1.049E-2, 3.215E-2, 7.513E-2, 0.143,
#                    0.235, 0.290, 0.384, 0.439, 0.460])
# y_data = np.array([5.922E-3, 1.083E-2, 3.452E-2, 6.007E-2, 0.117,
#                    0.155, 0.160, 0.132, 7.458E-2, 4.984E-4])
#серебро 2
# x_data = np.array([9.289E-3, 4.941E-2, 9.140E-2, 0.138, 0.192,
#                    0.258, 0.307, 0.336, 0.345, 0.347])
# y_data = np.array([2.905E-2, 8.129E-2, 0.115, 0.137, 0.145,
#                    0.133, 9.979E-2, 5.817E-2, 2.538E-2, 6.326E-4])
#синий 2
# x_data = np.array([6.337E-3, 3.719E-2, 9.093E-2, 0.148, 0.218,
#                    0.287, 0.335, 0.362, 0.375, 0.381])
# y_data = np.array([2.813E-2, 7.593E-2, 0.125, 0.151, 0.160,
#                    0.145, 0.112, 7.531E-2, 4.454E-2, 5.673E-4])

x_data = np.array([7.913E-3, 3.565E-2, 9.496E-2, 0.159, 0.219,
                   0.263, 0.281, 0.285])
y_data = np.array([3.368E-2, 7.848E-2, 0.119, 0.128, 0.113,
                   7.456E-2, 3.674E-2, 1.672E-3])
# Добавляем точку горлышка (0,0), как вы указали
x = np.insert(x_data, 0, 0.0)
y = np.insert(y_data, 0, 0.0)

# Убедимся, что x строго возрастает (требуется для интерполяции)
sort_idx = np.argsort(x)
x = x[sort_idx]
y = y[sort_idx]

print("📏 Геометрия шарика:")
print(f"   Длина (ось X): {x[-1]*100:.2f} см")
print(f"   Макс. радиус (ось Y): {max(y)*100:.2f} см")
print(f"   Макс. диаметр: {2*max(y)*100:.2f} см")
print()

# ===== 2. ИНТЕРПОЛЯЦИЯ ПРОФИЛЯ =====
# y(x) — радиус как функция координаты вдоль оси
spline = PchipInterpolator(x, y)

def integrand(z):
    r = spline(z)
    return np.pi * np.maximum(r, 0.0)**2  # защита от отрицательных значений сплайна

# ===== 3. ЧИСЛЕННОЕ ИНТЕГРИРОВАНИЕ =====
x_min, x_max = 0.0, x[-1]
volume, error = quad(integrand, x_min, x_max, epsabs=1e-10, epsrel=1e-10)

print("📊 РЕЗУЛЬТАТЫ РАСЧЁТА:")
print(f"   Объём: {volume*1e6:.1f} см³ (мл)")
print(f"   Объём: {volume*1e3:.2f} л")
print(f"   Погрешность квадратур: ±{error*1e6:.2e} см³")
print()

# Сравнение с эллипсоидом тех же габаритов
a = x_max / 2          # большая полуось
b = max(y)             # малая полуось (радиус)
V_ell = (4/3) * np.pi * a * b**2
print(f"📐 Для справки:")
print(f"   Объём эллипсоида (a={a*100:.1f} см, b={b*100:.1f} см): {V_ell*1e3:.2f} л")
print(f"   Коэффициент формы k = V_реал / V_эллипс: {volume/V_ell:.3f}")
print()


# ===== ОДИН ГРАФИК (профиль вращения) =====
x_fine = np.linspace(0, x[-1], 500)
y_fine = spline(x_fine)

plt.figure(figsize=(8, 6))
plt.plot(x_fine*100, y_fine*100, 'b-', linewidth=2, label='PCHIP интерполяция')
plt.plot(x_fine*100, -y_fine*100, 'b-', linewidth=2)
plt.plot(x*100, y*100, 'ro', markersize=6, label='Измеренные точки')
plt.axhline(0, color='k', linewidth=0.5)
plt.axvline(0, color='k', linewidth=0.5)
plt.gca().set_aspect('equal')
plt.xlabel('Ось X, см', fontsize=12)
plt.ylabel('Ось Y, см', fontsize=12)
plt.title('Профиль шарика (вращение вокруг оси X)', fontsize=14, pad=20)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


def calculate_surface_area_robust(spline, x_points, y_points):
    """Более устойчивый расчёт площади поверхности"""
    dspline = spline.derivative()

    def integrand(x):
        r = spline(x)
        dr = dspline(x)
        r_safe = np.maximum(r, 1e-10)
        return 2 * np.pi * r_safe * np.sqrt(1 + dr ** 2)

    # Интегрируем по частям, пропуская самые концы (где r ~ 0)
    # Начинаем с первой ненулевой точки и заканчиваем перед последней
    x_start = x_points[1] if len(x_points) > 2 else x_points[0] + 1e-6
    x_end = x_points[-2] if len(x_points) > 2 else x_points[-1] - 1e-6

    area_main, error_main = quad(integrand, x_start, x_end, limit=200)

    # Добавляем приближённую площадь концевых участков (как конусы)
    # Левый конус (горлышко)
    if x_points[0] == 0 and len(x_points) > 1:
        r1 = y_points[1]
        h1 = x_points[1]
        area_left = np.pi * r1 * np.sqrt(r1 ** 2 + h1 ** 2)
    else:
        area_left = 0

    # Правый конус (кончик)
    if len(x_points) > 2:
        r2 = y_points[-2]
        h2 = x_points[-1] - x_points[-2]
        area_right = np.pi * r2 * np.sqrt(r2 ** 2 + h2 ** 2)
    else:
        area_right = 0

    total_area = area_main + area_left + area_right
    total_error = error_main  # пренебрегаем погрешностью конусов

    return total_area, total_error


# ===== ИСПОЛЬЗОВАНИЕ =====
area, area_err = calculate_surface_area_robust(spline, x, y)

print(f"\n📐 Площадь поверхности:")
print(f"   S = {area * 1e4:.2f} см²  ({area:.4f} м²)")
print(f"   Погрешность интегрирования: ±{area_err * 1e4:.2f} см²")