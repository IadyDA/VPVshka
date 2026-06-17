# import pandas as pd
# import matplotlib.pyplot as plt
# import datetime
#
# # 1. Вносим данные из таблицы на изображении
# data = [
#     # Категория «маленький»
#     ('маленький', '18.04--10:47', 1746.7),
#     ('маленький', '18.04--13:46', 1824.3),
#     ('маленький', '20.04--17:37', 1447.1),
#
#     # Категория «средний»
#     ('средний', '18.04--10:48', 2560.8),
#     ('средний', '18.04--14:00', 2409.1),
#     ('средний', '20.04--17:38', 2085.1),
#
#     # Категория «большой»
#     ('большой', '18.04--12:20', 3058.3),
#     ('большой', '20.04--17:38', 2147.5),
#
#     # Категория «холодный»
#     ('холодный', '18.04--22:04', 1669.8),
#     ('холодный', '20.04--17:38', 1313.9)
# ]
#
# # 2. Создаем DataFrame
# df = pd.DataFrame(data, columns=['Категория', 'Время_строка', 'Площадь'])
#
#
# # 3. Функция преобразования пользовательского формата времени в datetime
# # Подставляем абстрактный текущий год для корректного парсинга
# def parse_datetime(time_str):
#     return datetime.datetime.strptime(f"{time_str} 2026", "%d.%m--%H:%M %Y")
#
#
# df['Время'] = df['Время_строка'].apply(parse_datetime)
#
# # Сортируем данные хронологически, чтобы линии графика не перекручивались
# df = df.sort_values(by='Время')
#
# # 4. Визуализация графиков
# plt.figure(figsize=(10, 6))
#
# # Цвета и маркеры для каждой категории
# styles = {
#     'маленький': {'color': 'blue', 'marker': 'o'},
#     'средний': {'color': 'green', 'marker': 's'},
#     'большой': {'color': 'red', 'marker': '^'},
#     'холодный': {'color': 'cyan', 'marker': 'd'}
# }
#
# # Строим отдельную линию для каждой категории
# for category, group in df.groupby('Категория'):
#     plt.plot(
#         group['Время'],
#         group['Площадь'],
#         label=category,
#         color=styles[category]['color'],
#         marker=styles[category]['marker'],
#         linewidth=2
#     )
#
# # Настройка осей и внешнего вида
# plt.title('Зависимость площади от времени по категориям', fontsize=14, fontweight='bold')
# plt.xlabel('Дата и время', fontsize=12)
# plt.ylabel('Площадь ($см^2$)', fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend(title='Категории', fontsize=10)
#
# # Форматируем отображение времени на оси X для читаемости
# plt.gcf().autofmt_xdate()
#
# # Показываем график
# plt.tight_layout()
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
#
# # Оригинальные наборы данных
# data = {
#     'Малый': (np.array([0, 179, 1472, 3290]), np.array([1824.3, 1746.7, 1592.7, 1447.1])),
#     'Средний': (np.array([0, 192, 1473, 3290]), np.array([2560.8, 2509.1, 2376.2, 2085.1])),
#     'Больший': (np.array([0, 185, 1476, 3198]), np.array([2750, 2650.8, 2451.2, 2147.5])), #ваще его надо подгонять
#     'Холодный': (np.array([0, 190, 1480, 2614]), np.array([1669.8, 1621.1, 1449.3, 1313.9]))
# }
#
# colors = {'Малый': 'red', 'Средний': '#FF9400', 'Больший': 'green', 'Холодный': 'blue'}
# markers = {'Малый': 'o', 'Средний': 's', 'Больший': 'D', 'Холодный': 'o'}
#
# plt.figure(figsize=(10, 6))
#
# all_t, all_S = [], []
#
# for label, (t, S) in data.items():
#     all_t.extend(t)
#     all_S.extend(S)
#     n = len(t)
#
#     # Расчет МНК
#     m_t, m_S = np.mean(t), np.mean(S)
#     denom = np.mean(t ** 2) - m_t ** 2
#     if denom == 0: continue
#
#     b = (np.mean(t * S) - m_t * m_S) / denom
#     a = m_S - b * m_t
#
#     # Расчет погрешностей
#     val_under_root = (np.mean(S ** 2) - m_S ** 2) / denom - b ** 2
#     sigma_b = (1 / np.sqrt(n)) * np.sqrt(max(0, val_under_root))
#     sigma_a = sigma_b * np.sqrt(denom)
#
#     y_err = np.sqrt(sigma_a ** 2 + (t * sigma_b) ** 2)
#     if n <= 2 or np.all(y_err == 0):
#         y_err = np.ones_like(S) * 100.0  # Усы для визуализации, если точек мало
#
#     # Формируем красивую подпись с уравнением МНК
#     legend_label = f'{label}: $S(t) = {a:.2f} {b:+.4f} \cdot t$'
#
#     # Передаем legend_label вместо исходного label
#     plt.errorbar(t, S, yerr=y_err, fmt=markers[label], color=colors[label],
#                  ecolor=colors[label], elinewidth=1.5, capsize=4, ms=6, label=legend_label, zorder=4)
#
#     # Линии аппроксимации подгоняются под максимальное t каждого графика
#     t_line = np.linspace(0, max(t), 100)
#     plt.plot(t_line, b * t_line + a, color=colors[label], linestyle='-', linewidth=1.8, alpha=0.8, zorder=3)
#
# # Восстановленные оригинальные подписи
# plt.title('Зависимость площади поверхности шариков от времени', fontsize=12, pad=15)
# plt.xlabel('Время, мин', fontsize=10)
# plt.ylabel('Площадь поверхности, $см^2$', fontsize=10)
# plt.grid(True, linestyle='--', alpha=0.5, zorder=1)
# plt.legend(loc='upper right')
#
# # Динамический расчет масштаба по массивам данных с отступами
# plt.xlim(min(all_t) - 100, max(all_t) + 100)
# plt.ylim(min(all_S) - 200, max(all_S) + 200)
#
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# === ИСХОДНЫЕ ДАННЫЕ ===
# Время в минутах переводим в часы (/ 60.0) прямо внутри словаря
data = {
    'Малый': (np.array([0, 179, 1472, 3290]) / 60.0, np.array([1824.3, 1746.7, 1592.7, 1447.1])),
    'Средний': (np.array([0, 192, 1473, 3290]) / 60.0, np.array([2560.8, 2509.1, 2376.2, 2085.1])),
    'Больший': (np.array([0, 185, 1476, 3198]) / 60.0, np.array([2750.0, 2650.8, 2451.2, 2147.5])),
    'Холодный': (np.array([0, 190, 1480, 2614]) / 60.0, np.array([1669.8, 1621.1, 1449.3, 1313.9]))
}

# Погрешность 3% для каждого значения
error_percent = 0.03

colors = {'Малый': 'red', 'Средний': '#FF9400', 'Больший': 'green', 'Холодный': 'blue'}
markers = {'Малый': 'o', 'Средний': 's', 'Больший': 'D', 'Холодный': 'o'}


# === ФУНКЦИЯ ВЗВЕШЕННОГО МНК ===
def weighted_linear_fit(x, y, yerr):
    """Взвешенный МНК: возвращает k (наклон), b (сдвиг)"""
    weights = 1 / yerr ** 2
    W = np.sum(weights)
    Wx = np.sum(weights * x)
    Wxx = np.sum(weights * x ** 2)
    Wy = np.sum(weights * y)
    Wxy = np.sum(weights * x * y)

    denom = W * Wxx - Wx ** 2
    if denom == 0:
        return 0, 0
    k = (W * Wxy - Wx * Wy) / denom
    b = (Wxx * Wy - Wx * Wxy) / denom
    return k, b


# === ПОСТРОЕНИЕ ГРАФИКА ===
plt.figure(figsize=(10, 6))

all_t, all_S = [], []

for label, (t, S) in data.items():
    all_t.extend(t)
    all_S.extend(S)

    # Считаем абсолютную погрешность 3% для каждой точки индивидуально
    y_err = S * error_percent

    # Находим коэффициенты прямой методом взвешенного МНК
    # b_coeff — наклон (см²/ч), a_coeff — свободный член (см²)
    b_coeff, a_coeff = weighted_linear_fit(t, S, y_err)

    # Формируем красивую подпись с уравнением S(t). Знак минус перед b_coeff встанет сам.
    legend_label = f'{label}: $S(t) = {a_coeff:.0f} {b_coeff:.2f} \\cdot t$'

    # Отображаем экспериментальные точки с усами 3% (без label, чтобы не дублировать)
    plt.errorbar(t, S, yerr=y_err, fmt=markers[label], color=colors[label],
                 ecolor=colors[label], elinewidth=1.5, capsize=4, ms=6, zorder=4)

    # Линии аппроксимации (передаем label с уравнением сюда)
    t_line = np.linspace(0, max(t), 100)
    plt.plot(t_line, b_coeff * t_line + a_coeff, color=colors[label],
             linestyle='-', linewidth=1.8, alpha=0.8, label=legend_label, zorder=3)

# Оформление графика
plt.title('Зависимость площади поверхности шариков от времени', fontsize=12, pad=15)
plt.xlabel('Время, ч', fontsize=10)  # Изменили на часы
plt.ylabel('Площадь поверхности, $см^2$', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5, zorder=1)
plt.legend(loc='upper right', fontsize=9)

# Автоматические отступы для осей (пересчитаны под часы)
plt.xlim(min(all_t) - 2.0, max(all_t) + 2.0)
plt.ylim(min(all_S) - 200, max(all_S) + 200)

plt.show()

# === ВЫВОД ДЛЯ ВСТАВКИ В ODS (С TAB-РАЗДЕЛИТЕЛЕМ) ===
print("\n📋 СКОПИРУЙТЕ И ВСТАВЬТЕ В ODS (Разделитель: Табуляция):")
print("-" * 60)
# Заголовки столбцов
print("Группа\tВремя (ч)\tПлощадь S (см²)\tПогрешность ±dS (см²)")

for label, (t, S) in data.items():
    # Рассчитываем 3% погрешность для каждой точки
    y_err = S * error_percent

    for i in range(len(t)):
        # Заменяем точки на запятые для русскоязычной локали ODS (если требуется)
        # Если у вас ODS принимает точки, замените ниже .replace('.', ',') на обычный вывод
        t_str = f"{t[i]:.4f}".replace('.', ',')
        S_str = f"{S[i]:.2f}".replace('.', ',')
        err_str = f"{y_err[i]:.2f}".replace('.', ',')

        print(f"{label}\t{t_str}\t{S_str}\t{err_str}")
print("-" * 60)
