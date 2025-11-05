import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import plotly.graph_objects as go


class MultipleCorrelationRegressionAnalysis:
    def __init__(self, data):
        self.data = data
        self.n = len(data)
        self.m = 2  # количество факторов

    def calculate_basic_stats(self):
        """Расчет основных статистических показателей"""
        stats_dict = {}
        for col in self.data.columns:
            stats_dict[col] = {
                'mean': np.mean(self.data[col]),
                'std': np.std(self.data[col], ddof=1),
                'min': np.min(self.data[col]),
                'max': np.max(self.data[col])
            }
        return stats_dict

    def calculate_pairwise_correlations(self):
        """Расчет парных коэффициентов корреляции"""
        y = self.data.iloc[:, 0].values  # результативный показатель
        x1 = self.data.iloc[:, 1].values  # первый фактор
        x2 = self.data.iloc[:, 2].values  # второй фактор

        # Средние значения
        y_mean = np.mean(y)
        x1_mean = np.mean(x1)
        x2_mean = np.mean(x2)

        # Расчет парных коэффициентов корреляции
        r_yx1 = np.sum((x1 - x1_mean) * (y - y_mean)) / (
            np.sqrt(np.sum((x1 - x1_mean) ** 2) * np.sum((y - y_mean) ** 2))
        )

        r_yx2 = np.sum((x2 - x2_mean) * (y - y_mean)) / (
            np.sqrt(np.sum((x2 - x2_mean) ** 2) * np.sum((y - y_mean) ** 2))
        )

        r_x1x2 = np.sum((x1 - x1_mean) * (x2 - x2_mean)) / (
            np.sqrt(np.sum((x1 - x1_mean) ** 2) * np.sum((x2 - x2_mean) ** 2))
        )

        return r_yx1, r_yx2, r_x1x2

    def calculate_partial_correlations(self, r_yx1, r_yx2, r_x1x2):
        """Расчет частных коэффициентов корреляции"""
        r_yx1_x2 = (r_yx1 - r_yx2 * r_x1x2) / np.sqrt(
            (1 - r_yx2 ** 2) * (1 - r_x1x2 ** 2)
        )

        r_yx2_x1 = (r_yx2 - r_yx1 * r_x1x2) / np.sqrt(
            (1 - r_yx1 ** 2) * (1 - r_x1x2 ** 2)
        )

        return r_yx1_x2, r_yx2_x1

    def calculate_multiple_correlations(self, r_yx1, r_yx2, r_x1x2):
        """Расчет множественных коэффициентов корреляции"""
        r_x2_yx1 = np.sqrt(
            (r_yx2 ** 2 + r_x1x2 ** 2 - 2 * r_yx1 * r_yx2 * r_x1x2) /
            (1 - r_yx1 ** 2)
        )

        r_x1_yx2 = np.sqrt(
            (r_yx1 ** 2 + r_x1x2 ** 2 - 2 * r_yx1 * r_yx2 * r_x1x2) /
            (1 - r_yx2 ** 2)
        )

        return r_x2_yx1, r_x1_yx2

    def test_significance_partial(self, r_partial, variable_name):
        """Проверка значимости частных коэффициентов корреляции"""
        t_calculated = r_partial * np.sqrt(self.n - self.m - 1) / np.sqrt(1 - r_partial ** 2)
        t_critical = stats.t.ppf(0.975, self.n - self.m - 1)  # для α=0.05

        is_significant = abs(t_calculated) > t_critical

        return {
            't_calculated': t_calculated,
            't_critical': t_critical,
            'is_significant': is_significant
        }

    def test_significance_multiple(self, r_multiple, variable_name):
        """Проверка значимости множественных коэффициентов корреляции"""
        F_calculated = (r_multiple ** 2 / (1 - r_multiple ** 2)) * (
                (self.n - self.m - 1) / self.m
        )
        F_critical = stats.f.ppf(0.95, self.m, self.n - self.m - 1)  # для α=0.05

        is_significant = F_calculated > F_critical

        return {
            'F_calculated': F_calculated,
            'F_critical': F_critical,
            'is_significant': is_significant
        }

    def linear_regression(self):
        """Построение линейного уравнения регрессии"""
        y = self.data.iloc[:, 0].values
        x1 = self.data.iloc[:, 1].values
        x2 = self.data.iloc[:, 2].values

        # Стандартные отклонения
        S_y = np.std(y, ddof=1)
        S_x1 = np.std(x1, ddof=1)
        S_x2 = np.std(x2, ddof=1)

        # Парные коэффициенты корреляции
        r_yx1, r_yx2, r_x1x2 = self.calculate_pairwise_correlations()

        # Коэффициенты уравнения регрессии
        b = (S_y / S_x1) * (r_yx1 - r_yx2 * r_x1x2) / (1 - r_x1x2 ** 2)
        c = (S_y / S_x2) * (r_yx2 - r_yx1 * r_x1x2) / (1 - r_x1x2 ** 2)
        a = np.mean(y) - b * np.mean(x1) - c * np.mean(x2)

        # Прогнозные значения
        y_pred = a + b * x1 + c * x2
        r2 = r2_score(y, y_pred)

        return {
            'coefficients': {'a': a, 'b': b, 'c': c},
            'equation': f"y = {a:.4f} + {b:.4f}*x1 + {c:.4f}*x2",
            'r2': r2,
            'y_pred': y_pred
        }

    def polynomial_regression(self, degree=2):
        """Построение полиномиальной регрессии"""
        y = self.data.iloc[:, 0].values
        X = self.data.iloc[:, 1:].values

        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)

        y_pred = model.predict(X_poly)
        r2 = r2_score(y, y_pred)

        return {
            'model': model,
            'poly': poly,
            'r2': r2,
            'y_pred': y_pred,
            'degree': degree
        }


def main():
    st.set_page_config(page_title="Множественный корреляционно-регрессионный анализ",
                       layout="wide")

    st.title("🔬 Множественный корреляционно-регрессионный анализ")
    st.write("Анализ взаимосвязей между P, n и Vмех")

    # Данные для варианта 1
    data = {
        'P, кН': [100, 140, 100, 140, 80, 200, 180, 180, 180, 180],
        'n, об/мин': [100, 100, 300, 300, 200, 200, 50, 350, 200, 400],
        'Vмех, м/ч': [3, 5, 4.5, 7, 3, 5, 1, 5, 3, 4.5]
    }

    df = pd.DataFrame(data)
    # Переупорядочиваем столбцы: результативный показатель первый (Vмех)
    df = df[['Vмех, м/ч', 'P, кН', 'n, об/мин']]
    y_col, x1_col, x2_col = 'Vмех, м/ч', 'P, кН', 'n, об/мин'

    st.subheader("📊 Исходные данные")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.dataframe(df)

    with col2:
        st.write("**Описание переменных:**")
        st.write("- **Vмех, м/ч** - результативный показатель")
        st.write("- **P, кН** - факторный показатель 1")
        st.write("- **n, об/мин** - факторный показатель 2")

    # Создаем экземпляр класса анализа
    analysis = MultipleCorrelationRegressionAnalysis(df)

    # Основные статистики
    st.subheader("📈 Основные статистические показатели")
    stats_data = analysis.calculate_basic_stats()

    stats_df = pd.DataFrame({
        'Показатель': ['Среднее', 'Стандартное отклонение', 'Минимум', 'Максимум'],
        y_col: [stats_data[y_col]['mean'], stats_data[y_col]['std'],
                stats_data[y_col]['min'], stats_data[y_col]['max']],
        x1_col: [stats_data[x1_col]['mean'], stats_data[x1_col]['std'],
                 stats_data[x1_col]['min'], stats_data[x1_col]['max']],
        x2_col: [stats_data[x2_col]['mean'], stats_data[x2_col]['std'],
                 stats_data[x2_col]['min'], stats_data[x2_col]['max']]
    })
    st.dataframe(stats_df)

    # Парные корреляции
    st.subheader("🔗 Парные коэффициенты корреляции")
    r_yx1, r_yx2, r_x1x2 = analysis.calculate_pairwise_correlations()

    corr_matrix = pd.DataFrame({
        y_col: [1.0, r_yx1, r_yx2],
        x1_col: [r_yx1, 1.0, r_x1x2],
        x2_col: [r_yx2, r_x1x2, 1.0]
    }, index=[y_col, x1_col, x2_col])

    col1, col2 = st.columns([2, 1])

    with col1:
        st.dataframe(corr_matrix.style.format("{:.4f}"))

    with col2:
        st.write("**Интерпретация:**")
        st.write(f"r(Vмех-P) = {r_yx1:.4f}")
        st.write(f"r(Vмех-n) = {r_yx2:.4f}")
        st.write(f"r(P-n) = {r_x1x2:.4f}")

    # Визуализация корреляционной матрицы
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax,
                square=True, cbar_kws={"shrink": 0.8})
    ax.set_title('Матрица парных корреляций')
    st.pyplot(fig)

    # Частные корреляции
    st.subheader("🎯 Частные коэффициенты корреляции")
    r_yx1_x2, r_yx2_x1 = analysis.calculate_partial_correlations(r_yx1, r_yx2, r_x1x2)

    partial_corr_df = pd.DataFrame({
        'Коэффициент': ['r(Vмех-P|n)', 'r(Vмех-n|P)'],
        'Обозначение': ['Исключено влияние n', 'Исключено влияние P'],
        'Значение': [r_yx1_x2, r_yx2_x1]
    })
    st.dataframe(partial_corr_df.style.format({"Значение": "{:.4f}"}))

    # Проверка значимости частных корреляций
    st.write("**Проверка значимости частных коэффициентов корреляции (α=0.05):**")
    test1 = analysis.test_significance_partial(r_yx1_x2, "Vмех-P|n")
    test2 = analysis.test_significance_partial(r_yx2_x1, "Vмех-n|P")

    significance_df = pd.DataFrame({
        'Коэффициент': ['r(Vмех-P|n)', 'r(Vмех-n|P)'],
        't-расчетное': [test1['t_calculated'], test2['t_calculated']],
        't-критическое': [test1['t_critical'], test2['t_critical']],
        'Значим': ['✅ Да' if test1['is_significant'] else '❌ Нет',
                   '✅ Да' if test2['is_significant'] else '❌ Нет']
    })
    st.dataframe(significance_df.style.format({"t-расчетное": "{:.4f}", "t-критическое": "{:.4f}"}))

    # Множественные корреляции
    st.subheader("🌐 Множественные коэффициенты корреляции")
    r_x2_yx1, r_x1_yx2 = analysis.calculate_multiple_correlations(r_yx1, r_yx2, r_x1x2)

    multiple_corr_df = pd.DataFrame({
        'Коэффициент': ['r(n-VмехP)', 'r(P-Vмехn)'],
        'Обозначение': ['Связь n с Vмех и P', 'Связь P с Vмех и n'],
        'Значение': [r_x2_yx1, r_x1_yx2]
    })
    st.dataframe(multiple_corr_df.style.format({"Значение": "{:.4f}"}))

    # Проверка значимости множественных корреляций
    st.write("**Проверка значимости множественных коэффициентов корреляции (α=0.05):**")
    test_m1 = analysis.test_significance_multiple(r_x2_yx1, "n-VмехP")
    test_m2 = analysis.test_significance_multiple(r_x1_yx2, "P-Vмехn")

    significance_multi_df = pd.DataFrame({
        'Коэффициент': ['r(n-VмехP)', 'r(P-Vмехn)'],
        'F-расчетное': [test_m1['F_calculated'], test_m2['F_calculated']],
        'F-критическое': [test_m1['F_critical'], test_m2['F_critical']],
        'Значим': ['✅ Да' if test_m1['is_significant'] else '❌ Нет',
                   '✅ Да' if test_m2['is_significant'] else '❌ Нет']
    })
    st.dataframe(significance_multi_df.style.format({"F-расчетное": "{:.4f}", "F-критическое": "{:.4f}"}))

    # Линейная регрессия
    st.subheader("📐 Линейное уравнение регрессии")
    linear_result = analysis.linear_regression()

    st.success(f"**Уравнение регрессии:** {linear_result['equation']}")
    st.info(f"**Коэффициент детерминации R²:** {linear_result['r2']:.4f}")

    # Детали коэффициентов
    st.write("**Коэффициенты уравнения:**")
    coef_df = pd.DataFrame({
        'Коэффициент': ['a (свободный член)', 'b (для P)', 'c (для n)'],
        'Значение': [linear_result['coefficients']['a'],
                     linear_result['coefficients']['b'],
                     linear_result['coefficients']['c']],
        'Интерпретация': ['Базовый уровень Vмех', 'Влияние P на Vмех', 'Влияние n на Vмех']
    })
    st.dataframe(coef_df.style.format({"Значение": "{:.4f}"}))

    # Визуализация линейной регрессии
    st.write("**3D визуализация линейной регрессии:**")

    # Создаем 3D график
    fig = go.Figure()

    # Добавляем исходные точки
    fig.add_trace(go.Scatter3d(
        x=df[x1_col], y=df[x2_col], z=df[y_col],
        mode='markers',
        marker=dict(size=8, color='red', symbol='circle'),
        name='Исходные данные',
        text=[f'P={p} кН, n={n} об/мин, Vмех={v} м/ч'
              for p, n, v in zip(df[x1_col], df[x2_col], df[y_col])]
    ))

    # Создаем сетку для поверхности
    x1_range = np.linspace(df[x1_col].min(), df[x1_col].max(), 20)
    x2_range = np.linspace(df[x2_col].min(), df[x2_col].max(), 20)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

    # Рассчитываем z для сетки
    z_grid = (linear_result['coefficients']['a'] +
              linear_result['coefficients']['b'] * x1_grid +
              linear_result['coefficients']['c'] * x2_grid)

    # Добавляем поверхность регрессии
    fig.add_trace(go.Surface(
        x=x1_grid, y=x2_grid, z=z_grid,
        colorscale='Blues',
        opacity=0.7,
        name='Плоскость регрессии',
        showscale=False
    ))

    fig.update_layout(
        title='Линейная регрессия: Vмех = f(P, n)',
        scene=dict(
            xaxis_title='P, кН',
            yaxis_title='n, об/мин',
            zaxis_title='Vмех, м/ч',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        width=800,
        height=600,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    st.plotly_chart(fig)

    # Полиномиальная регрессия
    st.subheader("🔄 Полиномиальная регрессия")

    degree = st.radio("Выберите степень полинома:", [2, 3], horizontal=True)
    poly_result = analysis.polynomial_regression(degree=degree)

    st.success(f"**Полиномиальная регрессия {degree}-й степени**")
    st.info(f"**Коэффициент детерминации R²:** {poly_result['r2']:.4f}")

    # Сравнение моделей
    st.subheader("📊 Сравнение моделей")

    comparison_df = pd.DataFrame({
        'Модель': ['Линейная', f'Полиномиальная ({degree} степень)'],
        'R²': [linear_result['r2'], poly_result['r2']],
        'Улучшение': ['-', f"+{(poly_result['r2'] - linear_result['r2']):.4f}"]
    })

    st.dataframe(comparison_df.style.format({"R²": "{:.4f}"}))

    # Визуализация полиномиальной регрессии
    st.write(f"**3D визуализация полиномиальной регрессии {degree}-й степени:**")

    fig_poly = go.Figure()

    # Добавляем исходные точки
    fig_poly.add_trace(go.Scatter3d(
        x=df[x1_col], y=df[x2_col], z=df[y_col],
        mode='markers',
        marker=dict(size=8, color='red', symbol='circle'),
        name='Исходные данные'
    ))

    # Создаем сетку для поверхности полиномиальной регрессии
    X_test = np.array([[x1, x2] for x1 in x1_range for x2 in x2_range])
    X_test_poly = poly_result['poly'].transform(X_test)
    z_poly = poly_result['model'].predict(X_test_poly)

    z_poly_grid = z_poly.reshape(len(x2_range), len(x1_range))

    fig_poly.add_trace(go.Surface(
        x=x1_grid, y=x2_grid, z=z_poly_grid,
        colorscale='Viridis',
        opacity=0.7,
        name=f'Поверхность регрессии {degree} степени',
        showscale=False
    ))

    fig_poly.update_layout(
        title=f'Полиномиальная регрессия {degree}-й степени: Vмех = f(P, n)',
        scene=dict(
            xaxis_title='P, кН',
            yaxis_title='n, об/мин',
            zaxis_title='Vмех, м/ч',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        width=800,
        height=600,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    st.plotly_chart(fig_poly)

    # Выводы
    st.subheader("🎯 Основные выводы")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Корреляционный анализ:**")
        if abs(r_yx1) > 0.7:
            st.write("✅ Сильная связь между Vмех и P")
        elif abs(r_yx1) > 0.3:
            st.write("⚠️ Умеренная связь между Vмех и P")
        else:
            st.write("❌ Слабая связь между Vмех и P")

        if abs(r_yx2) > 0.7:
            st.write("✅ Сильная связь между Vмех и n")
        elif abs(r_yx2) > 0.3:
            st.write("⚠️ Умеренная связь между Vмех и n")
        else:
            st.write("❌ Слабая связь между Vмех и n")

    with col2:
        st.write("**Регрессионный анализ:**")
        if linear_result['r2'] > 0.7:
            st.write("✅ Хорошее качество линейной модели")
        elif linear_result['r2'] > 0.5:
            st.write("⚠️ Удовлетворительное качество линейной модели")
        else:
            st.write("❌ Низкое качество линейной модели")

        if poly_result['r2'] > linear_result['r2'] + 0.1:
            st.write("✅ Полиномиальная модель значительно лучше")
        elif poly_result['r2'] > linear_result['r2']:
            st.write("⚠️ Полиномиальная модель немного лучше")
        else:
            st.write("❌ Линейная модель предпочтительнее")


if __name__ == "__main__":
    main()