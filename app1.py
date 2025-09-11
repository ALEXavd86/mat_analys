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
                    variance = np.var(data, ddof=1)
                    std_dev = np.std(data, ddof=1)
                    cv = (std_dev / mean) * 100 if mean != 0 else "Не определен"
                    skewness = stats.skew(data)
                    kurtosis = stats.kurtosis(data)


                    # Определение моды через модальный интервал
                    def calculate_mode_from_histogram(data, bins=10):
                        """Вычисляет моду через модальный интервал"""
                        # Строим гистограмму для определения интервалов
                        counts, bin_edges = np.histogram(data, bins=bins)

                        # Находим модальный интервал (интервал с максимальной частотой)
                        modal_interval_idx = np.argmax(counts)
                        modal_interval = (bin_edges[modal_interval_idx], bin_edges[modal_interval_idx + 1])
                        modal_frequency = counts[modal_interval_idx]

                        # Формула для моды в модальном интервале
                        # Mo = L + (f_m - f_{m-1}) / ((f_m - f_{m-1}) + (f_m - f_{m+1})) * h
                        L = modal_interval[0]  # Нижняя граница модального интервала
                        h = modal_interval[1] - modal_interval[0]  # Ширина интервала

                        # Частоты соседних интервалов
                        f_m = modal_frequency  # Частота модального интервала
                        f_prev = counts[modal_interval_idx - 1] if modal_interval_idx > 0 else 0
                        f_next = counts[modal_interval_idx + 1] if modal_interval_idx < len(counts) - 1 else 0

                        # Вычисляем моду
                        try:
                            mode_value = L + ((f_m - f_prev) / ((f_m - f_prev) + (f_m - f_next))) * h
                        except ZeroDivisionError:
                            # Если все частоты равны, берем середину интервала
                            mode_value = L + h / 2

                        return mode_value, modal_interval, modal_frequency


                    # Вычисляем моду
                    mode_value, modal_interval, modal_frequency = calculate_mode_from_histogram(data)

                    # Отображение результатов
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Количество значений", n)
                        st.metric("Среднее арифметическое", f"{mean:.4f}")
                        st.metric("Медиана", f"{median:.4f}")
                        st.metric("Мода (приближенная)", f"{mode_value:.4f}")
                        st.info(f"Модальный интервал: {modal_interval[0]:.2f} - {modal_interval[1]:.2f}")
                        st.info(f"Частота модального интервала: {modal_frequency}")

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
                    st.header("Гистограмма распределения с модальным интервалом")

                    # Автоматическое определение числа интервалов
                    k = int(1 + 3.322 * np.log10(len(clean_data)))
                    n_bins = st.slider("Количество интервалов гистограммы", 5, 20, min(k, 12))

                    fig, ax = plt.subplots(figsize=(8, 5))

                    # Построение гистограммы
                    counts, bin_edges, patches = ax.hist(clean_data, bins=n_bins, alpha=0.7,
                                                         edgecolor='black', density=False,
                                                         color='lightblue')

                    # Находим модальный интервал и вычисляем моду
                    modal_interval_idx = np.argmax(counts)
                    modal_interval = (bin_edges[modal_interval_idx], bin_edges[modal_interval_idx + 1])
                    modal_frequency = counts[modal_interval_idx]

                    # Вычисляем моду по формуле
                    L = modal_interval[0]  # Нижняя граница модального интервала
                    h = modal_interval[1] - modal_interval[0]  # Ширина интервала

                    # Частоты соседних интервалов
                    f_m = modal_frequency
                    f_prev = counts[modal_interval_idx - 1] if modal_interval_idx > 0 else 0
                    f_next = counts[modal_interval_idx + 1] if modal_interval_idx < len(counts) - 1 else 0

                    # Вычисляем моду
                    try:
                        mode_value = L + ((f_m - f_prev) / ((f_m - f_prev) + (f_m - f_next))) * h
                    except ZeroDivisionError:
                        mode_value = L + h / 2  # Середина интервала если все частоты равны

                    # Выделяем модальный интервал цветом
                    patches[modal_interval_idx].set_facecolor('red')
                    patches[modal_interval_idx].set_alpha(0.8)
                    patches[modal_interval_idx].set_edgecolor('darkred')
                    patches[modal_interval_idx].set_linewidth(2)

                    # Подписываем модальный интервал
                    ax.annotate(f'МОДАЛЬНЫЙ ИНТЕРВАЛ\nЧастота: {modal_frequency}',
                                xy=(bin_edges[modal_interval_idx] + h / 2, modal_frequency),
                                xytext=(0, 25),
                                textcoords='offset points',
                                ha='center', va='bottom',
                                fontsize=6, color='darkred', weight='bold',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                                arrowprops=dict(arrowstyle='->', color='darkred', lw=2))

                    # Отмечаем вычисленную моду вертикальной линией
                    ax.axvline(mode_value, color='red', linestyle='-', linewidth=3,
                               label=f'Мода: {mode_value:.2f}')

                    # Подписываем значение моды
                    ax.annotate(f'Мода = {mode_value:.2f}',
                                xy=(mode_value, modal_frequency * 0.8),
                                xytext=(10, 0),
                                textcoords='offset points',
                                ha='left', va='center',
                                fontsize=6, color='red', weight='bold',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

                    # Полигон частот
                    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                    ax.plot(bin_centers, counts, 'ro-', linewidth=2, markersize=6,
                            label='Полигон частот', color='darkred')

                    # Вертикальные линии для среднего и медианы
                    ax.axvline(mean_clean, color='blue', linestyle='--', linewidth=2,
                               label=f'Среднее: {mean_clean:.2f}')
                    ax.axvline(median, color='green', linestyle='--', linewidth=2,
                               label=f'Медиана: {median:.2f}')

                    # Теоретическая нормальная кривая
                    x = np.linspace(min(clean_data), max(clean_data), 100)
                    pdf = stats.norm.pdf(x, mean_clean, std_dev_clean) * len(clean_data) * h
                    ax.plot(x, pdf, 'k-', label='Нормальное распределение', linewidth=2, alpha=0.7)

                    # Добавляем значения частот на столбцы
                    for i, (count, patch) in enumerate(zip(counts, patches)):
                        if count > 0:
                            ax.text(patch.get_x() + patch.get_width() / 2, count + 0.1,
                                    f'{int(count)}', ha='center', va='bottom',
                                    fontsize=6, color='black', weight='bold')

                    # Настройка оформления
                    ax.set_xlabel('Значение', fontsize=5, weight='bold')
                    ax.set_ylabel('Частота', fontsize=5, weight='bold')
                    ax.set_title('Гистограмма распределения с выделенным модальным интервалом',
                                 fontsize=5, weight='bold', pad=20)

                    # Легенда с формулой моды
                    from matplotlib.offsetbox import AnchoredText

                    formula_text = f"Mo = L + (fₘ - fₘ₋₁)/((fₘ - fₘ₋₁) + (fₘ - fₘ₊₁)) × h\n"
                    formula_text += f"= {L:.2f} + ({f_m}-{f_prev})/(({f_m}-{f_prev}) + ({f_m}-{f_next})) × {h:.2f}"
                    anchored_text = AnchoredText(formula_text, loc='upper right',
                                                 frameon=True, prop=dict(size=4))
                    ax.add_artist(anchored_text)

                    ax.legend(loc='upper left', fontsize=5)
                    ax.grid(True, alpha=0.3)
                    ax.set_axisbelow(True)

                    st.pyplot(fig, use_container_width=True)

                    # Дополнительная информация о модальном интервале
                    st.subheader("📋 Информация о модальном интервале")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Мода", f"{mode_value:.4f}")
                    with col2:
                        st.metric("Модальный интервал", f"{modal_interval[0]:.2f} - {modal_interval[1]:.2f}")
                    with col3:
                        st.metric("Частота модального интервала", f"{modal_frequency}")
                    with col4:
                        st.metric("Ширина интервала", f"{h:.2f}")

                    # Таблица для модального интервала и соседних
                    st.subheader("📊 Частоты интервалов вокруг модального")

                    modal_table_data = []
                    for i in range(max(0, modal_interval_idx - 2), min(len(counts), modal_interval_idx + 3)):
                        interval_label = f"{bin_edges[i]:.2f} - {bin_edges[i + 1]:.2f}"
                        is_modal = "✅" if i == modal_interval_idx else ""
                        modal_table_data.append({
                            'Интервал': interval_label,
                            'Частота': int(counts[i]),
                            'Модальный': is_modal
                        })

                    modal_df = pd.DataFrame(modal_table_data)
                    st.dataframe(modal_df, use_container_width=True)

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