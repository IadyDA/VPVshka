import numpy as np

# === ДАННЫЕ ===
D_room = 0.94e-7   # см²/с, малый шарик при 298 К
T_room = 298       # К

D_cold = 0.35e-7   # см²/с, холодный шарик при 269 К
T_cold = 269       # К

R = 8.314          # Дж/(моль·К)

# === ПОГРЕШНОСТИ ===
sigma_T = 3.0      # К, погрешность температуры
sigma_D_rel = 0.09 # 9%, относительная погрешность D

# === РАСЧЁТ ЭНЕРГИИ АКТИВАЦИИ ===
ratio = D_room / D_cold
ln_ratio = np.log(ratio)

delta_inv_T = 1/T_cold - 1/T_room

E_a = R * ln_ratio / delta_inv_T  # Дж/моль
E_a_kJ = E_a / 1000  # кДж/моль

# === РАСЧЁТ ПОГРЕШНОСТИ Ea ===
# Формула: Ea = R * ln(D1/D2) / (1/T2 - 1/T1)
# Обозначим: A = ln(D1/D2), B = 1/T2 - 1/T1
# Тогда Ea = R * A / B

# Погрешность A = ln(D1/D2):
# σ_A = sqrt((σ_D1/D1)² + (σ_D2/D2)²)
sigma_A = np.sqrt(sigma_D_rel**2 + sigma_D_rel**2)

# Погрешность B = 1/T2 - 1/T1:
# σ_B = sqrt((σ_T1/T1²)² + (σ_T2/T2²)²)
sigma_B = np.sqrt((sigma_T/T_cold**2)**2 + (sigma_T/T_room**2)**2)

# Погрешность Ea:
# (σ_Ea/Ea)² = (σ_A/A)² + (σ_B/B)²
sigma_Ea_rel = np.sqrt((sigma_A/ln_ratio)**2 + (sigma_B/delta_inv_T)**2)
sigma_Ea = E_a * sigma_Ea_rel
sigma_Ea_kJ = sigma_Ea / 1000

print("=" * 70)
print("РАСЧЁТ ЭНЕРГИИ АКТИВАЦИИ ДИФФУЗИИ (уравнение Аррениуса)")
print("=" * 70)
print(f"D(298 К) = {D_room*1e7:.2f} × 10⁻⁷ см²/с  (σ = {sigma_D_rel*100:.0f}%)")
print(f"D(269 К) = {D_cold*1e7:.2f} × 10⁻⁷ см²/с  (σ = {sigma_D_rel*100:.0f}%)")
print(f"Отношение D(298)/D(269) = {ratio:.2f}")
print(f"ln(D1/D2) = {ln_ratio:.3f} ± {sigma_A:.3f}")
print(f"1/T2 - 1/T1 = {delta_inv_T:.5f} ± {sigma_B:.5f} К⁻¹")
print("-" * 70)
print(f"Энергия активации:")
print(f"  Ea = {E_a:.0f} ± {sigma_Ea:.0f} Дж/моль")
print(f"  Ea = {E_a_kJ:.2f} ± {sigma_Ea_kJ:.2f} кДж/моль")
print("=" * 70)

# === ПРЕДЕЛЫ ДОВЕРИТЕЛЬНОГО ИНТЕРВАЛА ===
print(f"\nДоверительный интервал (1σ):")
print(f"  [{E_a_kJ - sigma_Ea_kJ:.2f}; {E_a_kJ + sigma_Ea_kJ:.2f}] кДж/моль")

# === ВКЛАД РАЗЛИЧНЫХ ПОГРЕШНОСТЕЙ ===
print("\n" + "=" * 70)
print("ВКЛАД В ПОЛНУЮ ПОГРЕШНОСТЬ:")
print("-" * 70)
contrib_D = (sigma_A/ln_ratio)**2 / (sigma_Ea_rel**2) * 100
contrib_T = (sigma_B/delta_inv_T)**2 / (sigma_Ea_rel**2) * 100
print(f"  Погрешность измерения D: {contrib_D:.1f}%")
print(f"  Погрешность измерения T: {contrib_T:.1f}%")
print("=" * 70)

# === СРАВНЕНИЕ С ЛИТЕРАТУРНЫМИ ДАННЫМИ ===
print("\n" + "=" * 70)
print("СРАВНЕНИЕ С ТИПИЧНЫМИ ЗНАЧЕНИЯМИ")
print("=" * 70)
print("Для диффузии гелия в полимерах (резине):")
print("  Типичная Ea = 20-60 кДж/моль")
print(f"  Наша Ea = {E_a_kJ:.2f} ± {sigma_Ea_kJ:.2f} кДж/моль")

if (E_a_kJ - sigma_Ea_kJ) <= 60 and (E_a_kJ + sigma_Ea_kJ) >= 20:
    print("  ✓ Значение согласуется с типичным диапазоном!")
else:
    print("  ⚠ Значение выходит за пределы типичного диапазона")

print("=" * 70)

# === ПРОВЕРКА: найдём D0 ===
D0 = D_room / np.exp(-E_a/(R*T_room))
print(f"\nПредэкспоненциальный множитель:")
print(f"  D0 = {D0*1e7:.2f} × 10⁻⁷ см²/с")

# === ПРЕДСКАЗАНИЕ D ПРИ ДРУГИХ ТЕМПЕРАТУРАХ ===
print("\n" + "=" * 70)
print("ПРЕДСКАЗАНИЕ D ПРИ РАЗНЫХ ТЕМПЕРАТУРАХ:")
print("=" * 70)
for T_test in [273, 283, 293, 298, 308, 318]:
    D_pred = D0 * np.exp(-E_a/(R*T_test))
    # Погрешность предсказания (грубая оценка)
    D_pred_rel_err = np.sqrt(sigma_Ea_rel**2 + sigma_D_rel**2)
    print(f"  T = {T_test} К ({T_test-273:+3d}°C): D = {D_pred*1e7:.3f} × 10⁻⁷ см²/с")
print("=" * 70)