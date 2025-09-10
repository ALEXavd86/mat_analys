import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import io
import re
from collections import Counter

# Настройки страницы
st.set_page_config(
    page_title="Анализ статистических данных",
    page_icon="📊",
    layout="wide"
)

# Заголовок приложения
st.title("📊 Анализ статистических данных")
st.markdown("""
Эта программа выполняет полный статистический анализ данных из Excel-файла:
1. Расчет основных статистических показателей
2. Отбраковка грубых ошибок
3. Определение доверительного интервала
4. Расчет необходимого количества измерений
5. Построение гистограммы распределения
6. Проверка на нормальности распределения
""")


# Функция для правильного определения моды
def find_mode(data, tolerance=0.001):
    """Находит моду в данных с учетом чисел с плавающей точкой"""
    # Округляем значения для группировки
    rounded_data = np.round(data, decimals=3)
    counter = Counter(rounded_data)
    max_count = max(counter.values())

    # Находим все значения с максимальной частотой
    modes = [value for value, count in counter.items() if count == max_count]

    if max_count == 1:
        return "Все значения уникальные", 1
    elif len(modes) == 1:
        # Находим точное значение моды в исходных данных
        mode_value = modes[0]
        exact_values = [x for x in data if abs(np.round(x, 3) - mode_value) < tolerance]
        return np.mean(exact_values), max_count
    else:
        return f"Мультимодальное: {modes}", max_count


# Функция для коррекции формата чисел
def correct_number_format(value):
    if pd.isna(value):
        return None
    if isinstance(value, str):
        # Заменяем запятые на точки и удаляем лишние пробелы
        value = value.replace(',', '.').strip()
        try:
            return float(value)
        except ValueError:
            return None
    try:
        return float(value)
    except:
        return None


# Загрузка файла
uploaded_file = st.file_uploader("Загрузите Excel-файл с данными", type=['xlsx', 'xls'])

if uploaded_file is not None:
    try:
        # Чтение данных
        df = pd.read_excel(uploaded_file)

        st.info(f"Загружено {len(df)} строк, {len(df.columns)} колонок")

        # Преобразование всех колонок в числовой формат
        for col in df.columns:
            df[col] = df[col].apply(correct_number_format)

        # Выбор колонки для анализа
        numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

        if not numeric_columns:
            st.error("В файле нет числовых колонок для анализа")
        else:
            selected_column = st.selectbox("Выберите колонку для анализа:", numeric_columns)

            # Извлечение данных и удаление NaN
            data = df[selected_column].dropna().values
            n = len(data)

            st.info(f"В колонке '{selected_column}': {n} числовых значений из {len(df)}")

            if n < 3:
                st.error("Недостаточно данных для анализа (минимум 3 значения)")
            else:
                st.success(f"Загружено {n} значений для анализа")
                st.info(f"Диапазон данных: от {min(data):.2f} до {max(data):.2f}")

                # Показать все значения для проверки
                with st.expander("🔍 Просмотр данных"):
                    st.write("Все значения:")
                    for i, val in enumerate(data, 1):
                        st.write(f"{i}: {val:.4f}")

                # Разделение на вкладки
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📈 Основная статистика",
                    "🔍 Отбраковка ошибок",
                    "📏 Доверительный интервал",
                    "📊 Гистограмма",
                    "✅ Проверка нормальности"
                ])

                with tab1:
                    st.header("Основные статистические показатели")

                    # Расчет статистик
                    mean = np.mean(data)
                    median = np.median(data)

                    # Правильное определение моды
                    mode_value, mode_count = find_mode(data)
                    if isinstance(mode_value, str):
                        mode_display = mode_value
                    else:
                        mode_display = f"{mode_value:.4f} (встречается {mode_count} раз)"

                    variance = np.var(data, ddof=1)  # несмещенная оценка
                    std_dev = np.std(data, ddof=1)
                    cv = (std_dev / mean) * 100 if mean != 0 else "Не определен"
                    skewness = stats.skew(data)
                    kurtosis = stats.kurtosis(data)

                    # Отображение результатов
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Количество значений", n)
                        st.metric("Среднее арифметическое", f"{mean:.4f}")
                        st.metric("Медиана", f"{median:.4f}")
                        st.metric("Мода", mode_display)

                    with col2:
                        st.metric("Дисперсия", f"{variance:.4f}")
                        st.metric("Среднеквадратичное отклонение", f"{std_dev:.4f}")
                        st.metric("Коэффициент вариации", f"{cv:.2f}%" if isinstance(cv, float) else cv)
                        st.metric("Коэффициент асимметрии", f"{skewness:.4f}")
                        st.metric("Коэффициент эксцесса", f"{kurtosis:.4f}")


                with tab2:
                    st.header("Отбраковка грубых ошибок")

                    # Критерий Романовского
                    confidence_level = st.slider("Уровень доверия для отбраковки (%)", 90, 99, 95)
                    alpha = 1 - confidence_level / 100

                    # Критическое значение t-распределения
                    t_critical = stats.t.ppf(1 - alpha / 2, n - 2)
                    threshold = t_critical * std_dev

                    # Поиск выбросов
                    outliers = []

                    for i, value in enumerate(data):
                        if abs(value - mean) > threshold:
                            outliers.append((i, value))

                    # Удаление выбросов
                    if outliers:
                        outlier_indices = [idx for idx, val in outliers]
                        clean_data = np.delete(data, outlier_indices)

                        st.warning(f"Найдено {len(outliers)} выбросов:")
                        for idx, value in outliers:
                            st.write(f"Индекс {idx + 1}: {value:.4f} (отклонение: {abs(value - mean):.4f})")

                        st.success(f"После отбраковки осталось {len(clean_data)} значений")

                        # Обновление статистик для очищенных данных
                        mean_clean = np.mean(clean_data)
                        std_dev_clean = np.std(clean_data, ddof=1)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Среднее до очистки", f"{mean:.4f}")
                            st.metric("Среднее после очистки", f"{mean_clean:.4f}")
                        with col2:
                            st.metric("СКО до очистки", f"{std_dev:.4f}")
                            st.metric("СКО после очистки", f"{std_dev_clean:.4f}")
                    else:
                        st.success("Грубые ошибки не обнаружены")
                        clean_data = data
                        mean_clean = mean
                        std_dev_clean = std_dev

                with tab3:
                    st.header("Доверительный интервал и необходимое количество измерений")

                    # Доверительный интервал
                    conf_level = st.slider("Доверительная вероятность (%)", 90, 99, 95, key="conf_int")
                    alpha_conf = 1 - conf_level / 100

                    t_critical_conf = stats.t.ppf(1 - alpha_conf / 2, len(clean_data) - 1)
                    margin_error = t_critical_conf * std_dev_clean / np.sqrt(len(clean_data))

                    lower_bound = mean_clean - margin_error
                    upper_bound = mean_clean + margin_error

                    st.info(f"Доверительный интервал ({conf_level}%): "
                            f"[{lower_bound:.4f}, {upper_bound:.4f}]")
                    st.info(f"Точность измерения: ±{margin_error:.4f}")

                    # Необходимое количество измерений
                    st.subheader("Расчет необходимого количества измерений")

                    desired_error = st.number_input("Желаемая погрешность измерения:",
                                                    min_value=0.001,
                                                    value=float(0.1 * std_dev_clean),
                                                    format="%.4f")

                    if desired_error > 0:
                        required_n = ((t_critical_conf * std_dev_clean) / desired_error) ** 2
                        st.metric("Необходимое количество измерений", f"{int(np.ceil(required_n))}")

                with tab4:
                    st.header("Гистограмма распределения")

                    # Автоматическое определение числа интервалов (правило Стёрджеса)
                    k = int(1 + 3.322 * np.log10(len(clean_data)))

                    fig, ax = plt.subplots(figsize=(8, 5))
                    n_bins = st.slider("Количество интервалов гистограммы", 5, 30, min(k, 15))

                    ax.hist(clean_data, bins=n_bins, alpha=0.7, edgecolor='black', density=True)
                    ax.axvline(mean_clean, color='r', linestyle='--', label=f'Среднее: {mean_clean:.2f}')
                    ax.axvline(median, color='g', linestyle='--', label=f'Медиана: {median:.2f}')

                    # Теоретическая нормальная кривая
                    x = np.linspace(min(clean_data), max(clean_data), 100)
                    pdf = stats.norm.pdf(x, mean_clean, std_dev_clean)
                    ax.plot(x, pdf, 'k-', label='Нормальное распределение', linewidth=1)

                    ax.set_xlabel('Значение')
                    ax.set_ylabel('Плотность вероятности')
                    ax.set_title('Гистограмма распределения')
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)

                    st.pyplot(fig, use_container_width=False)

                with tab5:
                    st.header("Проверка нормальности распределения")

                    # Тест Шапиро-Уилка
                    if len(clean_data) >= 3 and len(clean_data) <= 5000:
                        stat, p_value = stats.shapiro(clean_data)

                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric("Статистика Шапиро-Уилка", f"{stat:.4f}")
                            st.metric("p-value", f"{p_value:.6f}")

                        with col2:
                            alpha_norm = 0.05
                            if p_value > alpha_norm:
                                st.success("✅ Данные соответствуют нормальному распределению")
                            else:
                                st.error("❌ Данные не соответствуют нормальному распределению")

                        # Q-Q plot
                        st.subheader("Q-Q график")

                        fig_qq, ax_qq = plt.subplots(figsize=(6, 4))
                        stats.probplot(clean_data, dist="norm", plot=ax_qq)
                        ax_qq.set_title('Q-Q график для проверки нормальности')
                        st.pyplot(fig_qq, use_container_width=False)

                    else:
                        st.warning("Тест Шапиро-Уилка требует от 3 до 5000 значений для анализа")

                    # Интерпретация
                    st.info("""
                    **Интерпретация результатов:**
                    - **p-value > 0.05**: нельзя отвергнуть гипотезу о нормальности распределения
                    - **Q-Q график**: точки должны лежать примерно на прямой линии
                    - **Гистограмма**: форма должна быть колоколообразной
                    """)

    except Exception as e:
        st.error(f"Ошибка при обработке файла: {str(e)}")
        st.error("Пожалуйста, убедитесь, что файл содержит числовые данные")
        import traceback

        st.text(traceback.format_exc())

else:
    st.info("👆 Пожалуйста, загрузите Excel-файл для начала анализа")