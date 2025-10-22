import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')

# Настройка страницы
st.set_page_config(page_title="Анализ регрессии", page_icon="📊", layout="wide")


def main():
    st.title("📊 Анализ регрессии: Y от X")
    st.markdown("---")

    # Исходные данные
    st.header("1. Исходные данные")

    # Данные из задания
    X_original = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    Y_original = np.array([1.40, 3.00, 7.00, 20.40, 51.60, 102.10, 183.80, 296.20, 426.70, 579.10, 773.20])

    # Создаем DataFrame для отображения
    df = pd.DataFrame({
        'X': X_original,
        'Y': Y_original
    })

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Таблица данных")
        st.dataframe(df.style.format({"X": "{:.0f}", "Y": "{:.1f}"}), use_container_width=True)

    with col2:
        st.subheader("График исходных данных")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(X_original, Y_original, color='blue', s=50, alpha=0.7, label='Данные')
        ax.set_xlabel('X - объясняющая переменная')
        ax.set_ylabel('Y - зависимая переменная')
        ax.set_title('Исходные данные')
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)

    st.markdown("---")

    # Линейная регрессия
    st.header("2. Линейная регрессия")

    # Подготовка данных для регрессии
    X = X_original.reshape(-1, 1)
    Y = Y_original

    # Линейная регрессия
    linear_model = LinearRegression()
    linear_model.fit(X, Y)
    Y_linear_pred = linear_model.predict(X)

    # Коэффициенты
    a_linear = linear_model.coef_[0]
    b_linear = linear_model.intercept_

    # Коэффициент корреляции и R²
    r_linear = np.corrcoef(X_original, Y_original)[0, 1]
    r2_linear = r2_score(Y, Y_linear_pred)

    # Оценка надежности коэффициента корреляции
    n = len(X_original)
    t_statistic = r_linear * np.sqrt(n - 2) / np.sqrt(1 - r_linear ** 2)
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_statistic), n - 2))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Результаты линейной регрессии")
        st.latex(f"Y = {b_linear:.2f} + {a_linear:.2f} \\cdot X")
        st.metric("Коэффициент корреляции (r)", f"{r_linear:.6f}")
        st.metric("Коэффициент детерминации (R²)", f"{r2_linear:.6f}")
        st.metric("t-статистика", f"{t_statistic:.4f}")
        st.metric("p-значение", f"{p_value:.6f}")

        # Интерпретация надежности
        if p_value < 0.05:
            st.success("✅ Коэффициент корреляции статистически значим (p < 0.05)")
        else:
            st.warning("⚠️ Коэффициент корреляции не является статистически значимым")

    with col2:
        st.subheader("График линейной регрессии")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(X_original, Y_original, color='blue', s=50, alpha=0.7, label='Исходные данные')
        ax.plot(X_original, Y_linear_pred, color='red', linewidth=2, label=f'Линейная регрессия (R² = {r2_linear:.4f})')
        ax.set_xlabel('X - объясняющая переменная')
        ax.set_ylabel('Y - зависимая переменная')
        ax.set_title('Линейная регрессия')
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)

    st.markdown("---")

    # Полиномиальные регрессии
    st.header("3. Полиномиальные регрессии")
    st.info("Поскольку R² линейной модели далек от 1, исследуем полиномиальные модели")

    # Создаем полиномиальные признаки
    degrees = [2, 3, 4, 5]
    poly_results = []

    for degree in degrees:
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)

        poly_model = LinearRegression()
        poly_model.fit(X_poly, Y)
        Y_poly_pred = poly_model.predict(X_poly)

        r2_poly = r2_score(Y, Y_poly_pred)

        poly_results.append({
            'degree': degree,
            'model': poly_model,
            'r2': r2_poly,
            'predictions': Y_poly_pred,
            'poly_features': poly
        })

    # Отображаем результаты полиномиальных регрессий
    cols = st.columns(len(degrees))

    for idx, (result, col) in enumerate(zip(poly_results, cols)):
        with col:
            st.subheader(f"Степень {result['degree']}")
            st.metric("R²", f"{result['r2']:.6f}")

            if result['r2'] > 0.999:
                st.success("🎯 Отличное соответствие!")
            elif result['r2'] > 0.99:
                st.info("✅ Хорошее соответствие")

    # Графики полиномиальных регрессий
    st.subheader("Сравнение моделей")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for idx, result in enumerate(poly_results):
        ax = axes[idx]
        ax.scatter(X_original, Y_original, color='blue', s=40, alpha=0.7, label='Данные')

        # Создаем плавную кривую для отображения
        X_smooth = np.linspace(X_original.min(), X_original.max(), 300).reshape(-1, 1)
        X_smooth_poly = result['poly_features'].transform(X_smooth)
        Y_smooth_pred = result['model'].predict(X_smooth_poly)

        ax.plot(X_smooth, Y_smooth_pred, color='red', linewidth=2,
                label=f'Степень {result["degree"]} (R² = {result["r2"]:.4f})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Полиномиальная регрессия степени {result["degree"]}')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    # Выбор лучшей модели
    st.header("4. Рекомендация по модели")

    best_poly = max(poly_results, key=lambda x: x['r2'])

    st.success(f"""
    **Рекомендуемая модель:** Полиномиальная регрессия {best_poly['degree']}-й степени
    - **Коэффициент детерминации R²:** {best_poly['r2']:.6f}
    - **Качество модели:** {'Превосходное' if best_poly['r2'] > 0.999 else 'Очень хорошее'}
    """)

    # Детали лучшей модели
    st.subheader("Уравнение лучшей модели")

    # Получаем коэффициенты для лучшей полиномиальной модели
    coefficients = best_poly['model'].coef_
    intercept = best_poly['model'].intercept_

    # Формируем уравнение
    equation = f"Y = {intercept:.4f}"
    for i in range(1, best_poly['degree'] + 1):
        if coefficients[i] >= 0:
            equation += f" + {coefficients[i]:.4f}·X^{i}"
        else:
            equation += f" - {abs(coefficients[i]):.4f}·X^{i}"

    st.latex(equation)

    # Сравнение всех моделей
    st.subheader("Сравнительная таблица моделей")

    comparison_data = []
    comparison_data.append({
        'Модель': 'Линейная',
        'Степень': 1,
        'R²': r2_linear,
        'Рекомендация': 'Не рекомендуется' if r2_linear < 0.95 else 'Рассмотреть'
    })

    for result in poly_results:
        comparison_data.append({
            'Модель': f'Полиномиальная',
            'Степень': result['degree'],
            'R²': result['r2'],
            'Рекомендация': 'Рекомендуется' if result['r2'] == best_poly['r2'] else 'Рассмотреть'
        })

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df.style.format({'R²': '{:.6f}'}).highlight_max(subset=['R²']),
                 use_container_width=True)


if __name__ == "__main__":
    main()