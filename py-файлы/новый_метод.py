import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.integrate import quad

# === НАСТРОЙКИ ===
IMAGE_PATH = 'big3.jpg'  # Путь к вашему фото
RULER_LENGTH_M = 0.15       # 20 см в метрах

print("📸 Загрузка изображения...")
try:
    img = plt.imread(IMAGE_PATH)
except FileNotFoundError:
    print(f"❌ Файл {IMAGE_PATH} не найден. Положите фото в папку со скриптом или укажите полный путь.")
    exit()

plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.axis('image')  # Сохраняет пропорции пикселей
plt.grid(False)
plt.xticks([])
plt.yticks([])

# === ЭТАП 1: ЛИНЕЙКА ===
print("1️⃣ Кликните по ДВУМ концам линейки (20 см) на картинке.")
print("   Затем нажмите Enter в окне графика.")
ruler_pts = plt.ginput(2)
if len(ruler_pts) != 2:
    print("❌ Нужно ровно 2 точки для линейки. Перезапустите.")
    exit()

p1, p2 = np.array(ruler_pts[0]), np.array(ruler_pts[1])
pixel_dist = np.linalg.norm(p2 - p1)
pixels_per_meter = pixel_dist / RULER_LENGTH_M
print(f"✅ Масштаб задан: {pixels_per_meter:.1f} пикс/м\n")

# === ЭТАП 2: ОСЬ ВРАЩЕНИЯ ===
print("2️ Кликните по ДВУМ концам длинной оси шарика.")
print("   Затем нажмите Enter.")
axis_pts = plt.ginput(2)
ax_start, ax_end = np.array(axis_pts[0]), np.array(axis_pts[1])
ax_vec = ax_end - ax_start
ax_unit = ax_vec / np.linalg.norm(ax_vec)
print("✅ Ось задана.\n")

# === ЭТАП 3: КОНТУР ===
print("3️⃣ Кликайте по контуру шарика (с одной стороны).")
print("   Для завершения нажмите Enter в окне графика.")
profile_pts = plt.ginput(-1)  # -1 = кликайте сколько угодно, Enter для выхода
if len(profile_pts) < 3:
    print("❌ Нужно минимум 3 точки контура.")
    exit()

plt.close()
print(f"✅ Принято {len(profile_pts)} точек контура. Считаю...\n")

# === МАТЕМАТИЧЕСКАЯ ОБРАБОТКА ===
data_x, data_r = [], []
for pt in profile_pts:
    p = np.array(pt)
    rel = p - ax_start
    x_coord = np.dot(rel, ax_unit) / pixels_per_meter
    # Расстояние до оси (2D cross product magnitude)
    r_coord = abs(rel[0]*ax_unit[1] - rel[1]*ax_unit[0]) / pixels_per_meter
    data_x.append(x_coord)
    data_r.append(r_coord)

# Сортировка по возрастанию x
idx = np.argsort(data_x)
x_data = np.array(data_x)[idx]
r_data = np.array(data_r)[idx]

# Интерполяция без осцилляций
spline = PchipInterpolator(x_data, r_data)
dspline = spline.derivative()

def integrand_area(x):
    r = spline(x)
    dr = dspline(x)
    return 2 * np.pi * np.maximum(r, 0) * np.sqrt(1 + dr**2)

S, err = quad(integrand_area, x_data[0], x_data[-1], limit=200)

# === ВЫВОД ===
print("="*40)
print("📊 РЕЗУЛЬТАТЫ РАСЧЁТА ПЛОЩАДИ ПОВЕРХНОСТИ")
print("="*40)
print(f"📏 Длина шарика: {x_data[-1]*100:.1f} см")
print(f"📏 Макс. радиус: {np.max(r_data)*100:.1f} см")
print(f" Площадь поверхности S = {S:.4f} м²")
print(f"🔵 Площадь поверхности S = {S*1e4:.1f} см²")
print(f"⚖️ Погрешность квадратур: ±{err:.2e} м²")
print("="*40)

# График для проверки
x_fine = np.linspace(x_data[0], x_data[-1], 400)
r_fine = spline(x_fine)
plt.figure(figsize=(6,3))
plt.plot(x_fine*100, r_fine*100, 'b-', lw=2)
plt.plot(x_fine*100, -r_fine*100, 'b-', lw=2)
plt.plot(x_data*100, r_data*100, 'ro', ms=5)
plt.axhline(0, color='k', lw=0.5)
plt.gca().set_aspect('equal')
plt.title(f'Восстановленный профиль\nS = {S*1e4:.1f} см²')
plt.xlabel('Длина (см)')
plt.ylabel('Радиус (см)')
plt.grid(alpha=0.3)
plt.show()