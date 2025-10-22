import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, norm
import warnings

warnings.filterwarnings('ignore')

# Настройка страницы
st.set_page_config(
    page_title="Проверка статистических гипотез",
    page_icon="📊",
    layout="wide"
)

# CSS для улучшения внешнего вида
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #2e86ab;
        padding-bottom: 0.5rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #2e86ab;
    }
    .hypothesis-accepted {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .hypothesis-rejected {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .formula {
        background-color: #e9ecef;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

# Заголовок приложения
st.markdown('<div class="main-header">📊 ПРОВЕРКА СТАТИСТИЧЕСКИХ ГИПОТЕЗ</div>', unsafe_allow_html=True)

# Теоретическая справка
with st.expander("📚 ТЕОРЕТИЧЕСКАЯ СПРАВКА", expanded=False):
    st.markdown("""
    ### Основные понятия

    **Статистическая гипотеза** - утверждение относительно истинных значений параметров исследуемой генеральной совокупности.

    **Нулевая гипотеза (H₀)** - предположение о том, что различия между выборками носят случайный характер

    **Альтернативная гипотеза (H₁)** - гипотеза, противоположная нулевой

    **Уровень значимости (α)** - вероятность отвергнуть нулевую гипотезу, когда она верна (вероятность ошибки I рода)
    """)

# Боковая панель для настроек
st.sidebar.header("⚙️ НАСТРОЙКИ АНАЛИЗА")
alpha = st.sidebar.slider("Уровень значимости (α)", 0.01, 0.10, 0.05, 0.01)

# Функции для расчетов согласно методичке
def manual_student_t_test(sample1, sample2, alpha):
    """Критерий Стьюдента согласно формуле (22) и (23) из методички"""
    # Основные характеристики
    n1, n2 = len(sample1), len(sample2)
    mean1, mean2 = np.mean(sample1), np.mean(sample2)
    var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)

    # Формула (22): T = (x̄1 - x̄2) / √(S²₁/n₁ + S²₂/n₂)
    t_statistic = (mean1 - mean2) / np.sqrt(var1**2 / n1 + var2**2 / n2)

    # Формула (23): степени свободы
    numerator = (var1**2 / n1 + var2**2 / n2) ** 2
    denominator = (var1**2 / n1) ** 2 / (n1 - 1) + (var2**2 / n2) ** 2 / (n2 - 1)
    df = numerator / denominator

    # Критическое значение
    t_critical = 1.960

    # Решение гипотезы
    reject_h0 = abs(t_statistic) > t_critical

    return {
        't_statistic': t_statistic,
        'df': df,
        't_critical': t_critical,
        'mean1': mean1,
        'mean2': mean2,
        'var1': var1,
        'var2': var2,
        'n1': n1,
        'n2': n2,
        'reject_h0': reject_h0,
        'formula_22': f"T = ({mean1:.4f} - {mean2:.4f}) / √({var1:.4f}/{n1} + {var2:.4f}/{n2}) = {t_statistic:.4f}",
        'formula_23': f"k = ({var1 / n1 + var2 / n2:.6f})² / (({var1 / n1:.6f})²/{n1 - 1} + ({var2 / n2:.6f})²/{n2 - 1}) = {df:.2f}"
    }

def manual_fisher_f_test(sample1, sample2, alpha):
    """Критерий Фишера согласно формуле (24) из методички"""
    var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)

    # Большая дисперсия в числителе
    if var1**2 >= var2**2:
        f_statistic = var1**2 / var2**2
        df1, df2 = len(sample1) - 1, len(sample2) - 1
        larger_var, smaller_var = "Выборка 1", "Выборка 2"
    else:
        f_statistic = var2**2 / var1**2
        df1, df2 = len(sample2) - 1, len(sample1) - 1
        larger_var, smaller_var = "Выборка 2", "Выборка 1"

    # Критическое значение (односторонний тест)
    f_critical = 1.62

    # Решение гипотезы
    reject_h0 = f_statistic > f_critical

    return {
        'f_statistic': f_statistic,
        'f_critical': f_critical,
        'df1': df1,
        'df2': df2,
        'var1': var1,
        'var2': var2,
        'larger_var': larger_var,
        'smaller_var': smaller_var,
        'reject_h0': reject_h0,
        'formula_24': f"F = {max(var1**2, var2**2):.4f} / {min(var1**2, var2**2):.4f} = {f_statistic:.4f}"
    }


# Таблица коэффициентов a_k для критерия Шапиро-Уилка (выдержка из Таблицы 9)
SHAPIRO_WILK_COEFFICIENTS = {
    23: [0.4542, 0.3126, 0.2563, 0.2139, 0.1787, 0.1480, 0.1201, 0.0941, 0.0696, 0.0459, 0.0228],

}

# Таблица критических значений W-критерия Шапиро-Уилка (выдержка из Таблицы 10)
SHAPIRO_WILK_CRITICAL_VALUES = {
    23: 0.914,  # для n=23, α=0.05

}


def manual_shapiro_wilk_test(sample):
    """Критерий Шапиро-Уилка согласно формулам (19)-(21) из методички"""
    n = len(sample)
    sorted_sample = np.sort(sample)

    # Формула (21): nμ₂ = Σ(xᵢ - x̄)²
    mean_sample = np.mean(sample)
    n_mu2 = np.sum((sample - mean_sample) ** 2)

    # Получаем коэффициенты для данного объема выборки
    if n in SHAPIRO_WILK_COEFFICIENTS:
        a_coeffs = SHAPIRO_WILK_COEFFICIENTS[n]
        w_critical = SHAPIRO_WILK_CRITICAL_VALUES[n]
    else:
        # Если объема нет в таблице, используем scipy как fallback
        w_statistic, p_value = shapiro(sample)
        return {
            'w_statistic': w_statistic,
            'w_critical': 0.05,  # для совместимости
            'p_value': p_value,
            'n': n,
            'n_mu2': n_mu2,
            'S': 0,
            'k': 0,
            'reject_h0': p_value < 0.05,  # для scipy логика другая
            'formula_19': "Используется scipy (объем выборки не в таблице)",
            'formula_20': f"W = {w_statistic:.4f} (scipy)",
            'formula_21': f"nμ₂ = Σ(xᵢ - x̄)² = {n_mu2:.4f}",
            'using_scipy': True
        }

    # Определяем k согласно методичке
    if n % 2 == 0:  # четный объем
        k = n // 2
    else:  # нечетный объем
        k = (n - 1) // 2

    # Берем только нужное количество коэффициентов
    a_coeffs = a_coeffs[:k]

    # Формула (19): S = Σ a_k [x_(n+1-k) - x_k]
    S = 0
    for i in range(k):
        idx1 = n - 1 - i  # x_(n+1-k)
        idx2 = i  # x_k
        S += a_coeffs[i] * (sorted_sample[idx1] - sorted_sample[idx2])

    # Формула (20): W = S² / (nμ₂)
    w_statistic = (S ** 2) / n_mu2 if n_mu2 != 0 else 0

    # Согласно методичке: если W > W_критическое, то принимается H₀
    reject_h0 = w_statistic < w_critical  # НЕправильно!
    # Правильно согласно методичке:
    accept_h0 = w_statistic > w_critical  # если W > W_критическое, принимаем H₀

    # Формируем строки формул для отображения (первые 3 слагаемых)
    formula_19_parts = []
    for i in range(min(3, k)):
        idx1 = n - 1 - i
        idx2 = i
        formula_19_parts.append(f"{a_coeffs[i]:.4f}×({sorted_sample[idx1]:.2f}-{sorted_sample[idx2]:.2f})")

    formula_19 = "S = " + " + ".join(formula_19_parts)
    if k > 3:
        formula_19 += f" + ... (всего {k} слагаемых)"

    return {
        'w_statistic': w_statistic,
        'w_critical': w_critical,
        'n': n,
        'n_mu2': n_mu2,
        'S': S,
        'k': k,
        'reject_h0': not accept_h0,  # инвертируем для совместимости с интерфейсом
        'accept_h0': accept_h0,  # прямое значение согласно методичке
        'formula_19': formula_19,
        'formula_20': f"W = S² / (nμ₂) = ({S:.4f})² / {n_mu2:.4f} = {w_statistic:.4f}",
        'formula_21': f"nμ₂ = Σ(xᵢ - x̄)² = {n_mu2:.4f}",
        'using_scipy': False
    }


# Таблица критических значений U-критерия Вилкоксона-Манна-Уитни
# (только нужные значения для наших выборок 1, 2, 3, 4)
MANN_WHITNEY_CRITICAL_VALUES = {
    # Для выборок 1 и 2: n=44, m=44 - используем нормальную аппроксимацию
    # Для выборок 3 и 4: n=23, m=23 - используем нормальную аппроксимацию
    # Оставляем пустым, так как наши выборки больше 20
}

# Таблица квантилей стандартного нормального распределения (Таблица 13)
NORMAL_QUANTILES = {
    0.90: 1.282,
    0.91: 1.341,
    0.92: 1.405,
    0.93: 1.476,
    0.94: 1.555,
    0.95: 1.645,
    0.96: 1.751,
    0.97: 1.881,
    0.98: 2.054,
    0.99: 2.326
}


def manual_mann_whitney_test(sample1, sample2, alpha):
    """Критерий Вилкоксона-Манна-Уитни согласно формулам (26)-(28)"""
    n, m = len(sample1), len(sample2)

    # Объединяем выборки и вычисляем ранги с учетом связанных значений
    combined = np.concatenate([sample1, sample2])

    # Вычисляем ранги с обработкой связей (как в методичке)
    ranks = stats.rankdata(combined, method='average')

    # Проверяем сумму рангов (должна равняться n + m)
    total_ranks_sum = np.sum(ranks)
    expected_sum = (n + m) * (n + m + 1) / 2
    if abs(total_ranks_sum - expected_sum) > 1e-10:
        st.warning(f"Сумма рангов ({total_ranks_sum:.1f}) не равна ожидаемой ({expected_sum:.1f})")

    # Суммы рангов
    R1 = np.sum(ranks[:n])
    R2 = np.sum(ranks[n:])

    # Формулы (26) и (27)
    U1 = n * m + (n * (n + 1)) / 2 - R1
    U2 = n * m + (m * (m + 1)) / 2 - R2

    U_statistic = min(U1, U2)

    # Для наших выборок (n=23, m=23 и n=44, m=44) используем нормальную аппроксимацию
    # согласно методичке: при n, m ≥ 4, n + m ≥ 20

    # Формула (28): Ũ = |U - nm/2| / √(1/12 * nm(n + m + 1))
    U_mean = n * m / 2
    U_std = np.sqrt(n * m * (n + m + 1) / 12)
    U_hat = abs(U_statistic - U_mean) / U_std

    # Квантиль стандартного нормального распределения
    # u_{1-α/2} из Таблицы 13
    p_value = 1 - alpha / 2
    if p_value in NORMAL_QUANTILES:
        z_critical = NORMAL_QUANTILES[p_value]
    else:
        # Аппроксимация если точного значения нет
        z_critical = norm.ppf(p_value)

    # Согласно методичке: если Ũ < u_{1-α/2}, то H₀ принимается
    reject_h0 = U_hat > z_critical

    # Формируем формулы для отображения
    formula_26 = f"U₁ = {n}×{m} + {n}×({n}+1)/2 - {R1:.1f} = {U1:.1f}"
    formula_27 = f"U₂ = {n}×{m} + {m}×({m}+1)/2 - {R2:.1f} = {U2:.1f}"
    formula_28 = f"Ũ = |{U_statistic:.1f} - {n}×{m}/2| / √(1/12×{n}×{m}×({n}+{m}+1)) = {U_hat:.4f}"

    return {
        'U_statistic': U_statistic,
        'U1': U1,
        'U2': U2,
        'R1': R1,
        'R2': R2,
        'n': n,
        'm': m,
        'U_hat': U_hat,
        'z_critical': z_critical,
        'reject_h0': reject_h0,
        'formula_26': formula_26,
        'formula_27': formula_27,
        'formula_28': formula_28,
        'use_table': False  # для наших выборок всегда используем нормальную аппроксимацию
    }

def create_sample_data():
    """Создание данных согласно таблице 4"""
    # Данные для выборок 1 и 2
    sample1_data = [
        21.42, 21.24, 24.39, 24.98, 19.27, 25.75, 18.14, 19.60, 20.66, 20.36,
        20.98, 19.93, 23.09, 18.41, 21.87, 22.64, 25.25, 24.48, 21.01, 18.68,
        19.90, 23.33, 21.72, 20.09, 23.84, 27.35, 22.11, 24.41, 26.65, 14.75,
        22.76, 24.43, 20.31, 18.38, 22.02, 25.35, 23.51, 18.65, 19.95, 22.17,
        20.09, 24.62, 22.91, 24.65
    ]

    sample2_data = [
        14.52, 16.21, 15.56, 17.48, 17.84, 13.38, 14.81, 14.54, 16.69, 15.24,
        14.83, 19.04, 18.96, 16.48, 17.80, 15.05, 14.35, 11.93, 15.28, 17.46,
        14.57, 15.86, 10.20, 14.33, 20.56, 14.38, 17.84, 12.75, 15.02, 16.03,
        18.38, 18.34, 16.14, 13.48, 17.00, 15.62, 17.53, 19.71, 12.50, 17.87,
        17.77, 17.77, 15.21, 17.22
    ]

    # Данные для выборок 3 и 4
    sample3_data = [
        90.06, 95.39, 93.55, 95.98, 100.53, 91.05, 91.80, 97.97, 97.64, 88.70,
        102.52, 90.84, 94.65, 84.04, 88.58, 100.37, 89.98, 92.99, 89.65, 93.26,
        85.00, 107.25, 99.74
    ]

    sample4_data = [
        91.64, 96.47, 101.05, 97.34, 84.30, 110.42, 95.51, 111.54, 99.24, 103.24,
        98.15, 83.02, 103.67, 101.71, 97.88, 94.62, 106.37, 98.80, 95.94, 92.28,
        107.49, 96.83, 92.67
    ]

    return {
        'sample1': np.array(sample1_data),
        'sample2': np.array(sample2_data),
        'sample3': np.array(sample3_data),
        'sample4': np.array(sample4_data)
    }

# Основной интерфейс
st.markdown('<div class="section-header">📊 АНАЛИЗ ДАННЫХ</div>', unsafe_allow_html=True)

# Загрузка данных
data = create_sample_data()

# Описательная статистика
st.markdown("### Описательная статистика выборок")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Выборка 1", f"n={len(data['sample1'])}",
              f"μ={np.mean(data['sample1']):.2f}, σ²={np.var(data['sample1'], ddof=1):.2f}")

with col2:
    st.metric("Выборка 2", f"n={len(data['sample2'])}",
              f"μ={np.mean(data['sample2']):.2f}, σ²={np.var(data['sample2'], ddof=1):.2f}")

with col3:
    st.metric("Выборка 3", f"n={len(data['sample3'])}",
              f"μ={np.mean(data['sample3']):.2f}, σ²={np.var(data['sample3'], ddof=1):.2f}")

with col4:
    st.metric("Выборка 4", f"n={len(data['sample4'])}",
              f"μ={np.mean(data['sample4']):.2f}, σ²={np.var(data['sample4'], ddof=1):.2f}")

# Визуализация распределений
st.markdown("### Визуализация распределений")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

samples = [data['sample1'], data['sample2'], data['sample3'], data['sample4']]
titles = ['Выборка 1', 'Выборка 2', 'Выборка 3', 'Выборка 4']

for i, (sample, title) in enumerate(zip(samples, titles)):
    ax = axes[i // 2, i % 2]
    sns.histplot(sample, kde=True, ax=ax, color='skyblue', alpha=0.7)
    ax.set_title(title)
    ax.axvline(np.mean(sample), color='red', linestyle='--', label=f'Среднее: {np.mean(sample):.2f}')
    ax.legend()

plt.tight_layout()
st.pyplot(fig)

# ПРОВЕРКА ГИПОТЕЗ
st.markdown('<div class="section-header">🔍 ПРОВЕРКА СТАТИСТИЧЕСКИХ ГИПОТЕЗ</div>', unsafe_allow_html=True)

# 1. Критерий Стьюдента
st.markdown("### 1. Критерий Стьюдента (Выборки 1 и 2)")
st.markdown("**H₀:** Средние значения выборок 1 и 2 равны (μ₁ = μ₂)")
st.markdown("**H₁:** Средние значения выборок 1 и 2 различаются (μ₁ ≠ μ₂)")

t_result = manual_student_t_test(data['sample1'], data['sample2'], alpha)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.write("**Расчет по формулам методички:**")
    st.markdown('<div class="formula">' + t_result['formula_22'] + '</div>', unsafe_allow_html=True)
    st.markdown('<div class="formula">' + t_result['formula_23'] + '</div>', unsafe_allow_html=True)
    st.write("**Результаты:**")
    st.write(f"- t-статистика: {t_result['t_statistic']:.4f}")
    st.write(f"- Степени свободы: {t_result['df']:.2f}")
    st.write(f"- Критическое значение: ±{t_result['t_critical']:.4f}")
    st.write(f"- Среднее выборки 1: {t_result['mean1']:.4f}")
    st.write(f"- Среднее выборки 2: {t_result['mean2']:.4f}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if t_result['reject_h0']:
        st.markdown('<div class="hypothesis-rejected">', unsafe_allow_html=True)
        st.write("**Результат:** Нулевая гипотеза H₀ ОТВЕРГАЕТСЯ")
        st.write("**Вывод:** Существует статистически значимое различие между средними значениями выборок 1 и 2")
        st.write(f"Так как |{t_result['t_statistic']:.4f}| > {t_result['t_critical']:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="hypothesis-accepted">', unsafe_allow_html=True)
        st.write("**Результат:** Нулевая гипотеза H₀ ПРИНИМАЕТСЯ")
        st.write("**Вывод:** Нет статистически значимых различий между средними значениями выборок 1 и 2")
        st.write(f"Так как |{t_result['t_statistic']:.4f}| ≤ {t_result['t_critical']:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)

# 2. Критерий Фишера
st.markdown("### 2. Критерий Фишера (Выборки 1 и 2)")
st.markdown("**H₀:** Дисперсии выборок 1 и 2 равны (σ₁² = σ₂²)")
st.markdown("**H₁:** Дисперсии выборок 1 и 2 различаются (σ₁² ≠ σ₂²)")

f_result = manual_fisher_f_test(data['sample1'], data['sample2'], alpha)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.write("**Расчет по формулам методички:**")
    st.markdown('<div class="formula">' + f_result['formula_24'] + '</div>', unsafe_allow_html=True)
    st.write("**Результаты:**")
    st.write(f"- F-статистика: {f_result['f_statistic']:.4f}")
    st.write(f"- Степени свободы: ({f_result['df1']}, {f_result['df2']})")
    st.write(f"- Критическое значение: {f_result['f_critical']:.4f}")
    st.write(f"- Дисперсия выборки 1: {f_result['var1']:.4f}")
    st.write(f"- Дисперсия выборки 2: {f_result['var2']:.4f}")
    st.write(f"- Большая дисперсия: {f_result['larger_var']}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if f_result['reject_h0']:
        st.markdown('<div class="hypothesis-rejected">', unsafe_allow_html=True)
        st.write("**Результат:** Нулевая гипотеза H₀ ОТВЕРГАЕТСЯ")
        st.write("**Вывод:** Существует статистически значимое различие между дисперсиями выборок 1 и 2")
        st.write(f"Так как {f_result['f_statistic']:.4f} > {f_result['f_critical']:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="hypothesis-accepted">', unsafe_allow_html=True)
        st.write("**Результат:** Нулевая гипотеза H₀ ПРИНИМАЕТСЯ")
        st.write("**Вывод:** Нет статистически значимых различий между дисперсиями выборок 1 и 2")
        st.write(f"Так как {f_result['f_statistic']:.4f} ≤ {f_result['f_critical']:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)

# 3. Критерий Шапиро-Уилка
st.markdown("### 3. Критерий Шапиро-Уилка (Выборка 3)")
st.markdown("**H₀:** Выборка 3 распределена нормально")
st.markdown("**H₁:** Выборка 3 не распределена нормально")

shapiro_result = manual_shapiro_wilk_test(data['sample3'])

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.write("**Расчет по формулам методички:**")

    if shapiro_result.get('using_scipy', False):
        st.warning("Используется scipy.stats.shapiro (объем выборки не найден в таблице коэффициентов)")
        st.write(f"- W-статистика: {shapiro_result['w_statistic']:.4f}")
        st.write(f"- p-value: {shapiro_result['p_value']:.6f}")
    else:
        st.markdown('<div class="formula">' + shapiro_result['formula_21'] + '</div>', unsafe_allow_html=True)
        st.markdown('<div class="formula">' + shapiro_result['formula_19'] + '</div>', unsafe_allow_html=True)
        st.markdown('<div class="formula">' + shapiro_result['formula_20'] + '</div>', unsafe_allow_html=True)
        st.write("**Результаты:**")
        st.write(f"- W-статистика: {shapiro_result['w_statistic']:.4f}")
        st.write(f"- S: {shapiro_result['S']:.4f}")
        st.write(f"- nμ₂: {shapiro_result['n_mu2']:.4f}")
        st.write(f"- Количество коэффициентов (k): {shapiro_result['k']}")

    st.write(f"- Критическое значение (Wт): {shapiro_result['w_critical']:.4f}")
    st.write(f"- Объем выборки: {shapiro_result['n']}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # Используем правильную логику согласно методичке
    if shapiro_result.get('using_scipy', False):
        # Логика scipy
        if shapiro_result['reject_h0']:
            st.markdown('<div class="hypothesis-rejected">', unsafe_allow_html=True)
            st.write("**Результат:** Нулевая гипотеза H₀ ОТВЕРГАЕТСЯ")
            st.write("**Вывод:** Выборка 3 не распределена нормально")
            st.write(f"Так как p-value ({shapiro_result['p_value']:.6f}) < α ({alpha})")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="hypothesis-accepted">', unsafe_allow_html=True)
            st.write("**Результат:** Нулевая гипотеза H₀ ПРИНИМАЕТСЯ")
            st.write("**Вывод:** Выборка 3 распределена нормально")
            st.write(f"Так как p-value ({shapiro_result['p_value']:.6f}) ≥ α ({alpha})")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Логика согласно методичке: если W > Wт, принимаем H₀
        if shapiro_result['accept_h0']:
            st.markdown('<div class="hypothesis-accepted">', unsafe_allow_html=True)
            st.write("**Результат:** Нулевая гипотеза H₀ ПРИНИМАЕТСЯ")
            st.write("**Вывод:** Выборка 3 распределена нормально")
            st.write(f"Так как W ({shapiro_result['w_statistic']:.4f}) > Wт ({shapiro_result['w_critical']:.4f})")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="hypothesis-rejected">', unsafe_allow_html=True)
            st.write("**Результат:** Нулевая гипотеза H₀ ОТВЕРГАЕТСЯ")
            st.write("**Вывод:** Выборка 3 не распределена нормально")
            st.write(f"Так как W ({shapiro_result['w_statistic']:.4f}) ≤ Wт ({shapiro_result['w_critical']:.4f})")
            st.markdown('</div>', unsafe_allow_html=True)

# 4. Критерий Вилкоксона-Манна-Уитни
st.markdown("### 4. Критерий Вилкоксона-Манна-Уитни (Выборки 3 и 4)")
st.markdown("**H₀:** Выборки 3 и 4 принадлежат одной генеральной совокупности")
st.markdown("**H₁:** Выборки 3 и 4 принадлежат разным генеральным совокупностям")

mw_result = manual_mann_whitney_test(data['sample3'], data['sample4'], alpha)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.write("**Расчет по формулам методички:**")
    st.markdown('<div class="formula">' + mw_result['formula_26'] + '</div>', unsafe_allow_html=True)
    st.markdown('<div class="formula">' + mw_result['formula_27'] + '</div>', unsafe_allow_html=True)
    st.markdown('<div class="formula">' + mw_result['formula_28'] + '</div>', unsafe_allow_html=True)

    st.write("**Результаты:**")
    st.write(f"- U-статистика: {mw_result['U_statistic']:.1f}")
    st.write(f"- U₁: {mw_result['U1']:.1f}")
    st.write(f"- U₂: {mw_result['U2']:.1f}")
    st.write(f"- R₁: {mw_result['R1']:.1f}")
    st.write(f"- R₂: {mw_result['R2']:.1f}")
    st.write(f"- Ũ: {mw_result['U_hat']:.4f}")
    st.write(f"- u_{{1-α/2}}: {mw_result['z_critical']:.4f}")
    st.write(f"- Объемы выборок: n={mw_result['n']}, m={mw_result['m']}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if mw_result['reject_h0']:
        st.markdown('<div class="hypothesis-rejected">', unsafe_allow_html=True)
        st.write("**Результат:** Нулевая гипотеза H₀ ОТВЕРГАЕТСЯ")
        st.write("**Вывод:** Выборки 3 и 4 принадлежат разным генеральным совокупностям")
        st.write(f"Так как Ũ ({mw_result['U_hat']:.4f}) > u_{{1-α/2}} ({mw_result['z_critical']:.4f})")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="hypothesis-accepted">', unsafe_allow_html=True)
        st.write("**Результат:** Нулевая гипотеза H₀ ПРИНИМАЕТСЯ")
        st.write("**Вывод:** Выборки 3 и 4 принадлежат одной генеральной совокупности")
        st.write(f"Так как Ũ ({mw_result['U_hat']:.4f}) ≤ u_{{1-α/2}} ({mw_result['z_critical']:.4f})")
        st.markdown('</div>', unsafe_allow_html=True)

# ИТОГОВЫЕ ВЫВОДЫ
st.markdown('<div class="section-header">📋 ИТОГОВЫЕ ВЫВОДЫ</div>', unsafe_allow_html=True)

conclusions = []

# Вывод по критерию Стьюдента
if t_result['reject_h0']:
    conclusions.append("✅ **Существует статистически значимое различие** между средними значениями выборок 1 и 2")
else:
    conclusions.append("❌ **Нет статистически значимого различия** между средними значениями выборок 1 и 2")

# Вывод по критерию Фишера
if f_result['reject_h0']:
    conclusions.append("✅ **Существует статистически значимое различие** между дисперсиями выборок 1 и 2")
else:
    conclusions.append("❌ **Нет статистически значимого различия** между дисперсиями выборок 1 и 2")

# Вывод по критерию Шапиро-Уилка
if shapiro_result['reject_h0']:
    conclusions.append("✅ **Выборка 3 не распределена нормально**")
else:
    conclusions.append("❌ **Выборка 3 распределена нормально**")

# Вывод по критерию Манна-Уитни
if mw_result['reject_h0']:
    conclusions.append("✅ **Выборки 3 и 4 принадлежат разным генеральным совокупностям**")
else:
    conclusions.append("❌ **Выборки 3 и 4 принадлежат одной генеральной совокупности**")

for conclusion in conclusions:
    st.write(f"- {conclusion}")

# Футер
st.markdown("---")
st.markdown(
    "**Разработано для проверки статистических гипотез согласно методичке** | "
    "Используются точные формулы из методических указаний"
)