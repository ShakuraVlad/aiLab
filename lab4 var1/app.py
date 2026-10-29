import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title='Прогноз осадков — Мамдани', layout='wide')


def triangle(x, a, b, c):
    x = np.asarray(x, dtype=float)
    return np.maximum(0, np.minimum(np.where(b == a, 1.0, (x - a) / (b - a) if b != a else 0), np.where(c == b, 1.0, (c - x) / (c - b) if c != b else 0)))


def trapezoid(x, a, b, c, d):
    x = np.asarray(x, dtype=float)
    left = np.ones_like(x) if a == b else np.clip((x-a)/(b-a), 0, 1)
    right = np.ones_like(x) if c == d else np.clip((d-x)/(d-c), 0, 1)
    return np.minimum(left, right)


DOMAINS = {
    'Температура (°C)': np.linspace(-30, 50, 801),
    'Влажность (%)': np.linspace(0, 100, 1001),
    'Давление (гПа)': np.linspace(950, 1050, 1001),
    'Вероятность осадков (%)': np.linspace(0, 100, 1001),
}

SETS = {
    'Температура (°C)': {
        'низкая': lambda x: trapezoid(x, -30, -30, 0, 15),
        'умеренная': lambda x: triangle(x, 5, 20, 35),
        'высокая': lambda x: trapezoid(x, 25, 40, 50, 50),
    },
    'Влажность (%)': {
        'низкая': lambda x: trapezoid(x, 0, 0, 25, 50),
        'средняя': lambda x: triangle(x, 30, 55, 80),
        'высокая': lambda x: trapezoid(x, 60, 85, 100, 100),
    },
    'Давление (гПа)': {
        'низкое': lambda x: trapezoid(x, 950, 950, 980, 1005),
        'среднее': lambda x: triangle(x, 990, 1015, 1040),
        'высокое': lambda x: trapezoid(x, 1025, 1045, 1050, 1050),
    },
    'Вероятность осадков (%)': {
        'низкая': lambda x: trapezoid(x, 0, 0, 20, 45),
        'средняя': lambda x: triangle(x, 25, 50, 75),
        'высокая': lambda x: trapezoid(x, 55, 80, 100, 100),
    },
}

# Каждое условие: (название переменной, лингвистический терм).
# AND = минимум, OR = максимум, импликация = отсечение, агрегация = максимум.
RULES = [
    ('Температура низкая И влажность высокая', 'AND', [('Температура (°C)', 'низкая'), ('Влажность (%)', 'высокая')], 'высокая'),
    ('Температура умеренная И влажность высокая', 'AND', [('Температура (°C)', 'умеренная'), ('Влажность (%)', 'высокая')], 'высокая'),
    ('Температура высокая И давление высокое', 'AND', [('Температура (°C)', 'высокая'), ('Давление (гПа)', 'высокое')], 'низкая'),
    ('Температура высокая И влажность низкая', 'AND', [('Температура (°C)', 'высокая'), ('Влажность (%)', 'низкая')], 'низкая'),
    ('Давление низкое И влажность высокая', 'AND', [('Давление (гПа)', 'низкое'), ('Влажность (%)', 'высокая')], 'высокая'),
    ('Давление среднее И влажность средняя', 'AND', [('Давление (гПа)', 'среднее'), ('Влажность (%)', 'средняя')], 'средняя'),
    ('Температура низкая И давление низкое', 'AND', [('Температура (°C)', 'низкая'), ('Давление (гПа)', 'низкое')], 'высокая'),
    ('Температура умеренная И давление высокое', 'AND', [('Температура (°C)', 'умеренная'), ('Давление (гПа)', 'высокое')], 'низкая'),
    ('Температура высокая И влажность средняя', 'AND', [('Температура (°C)', 'высокая'), ('Влажность (%)', 'средняя')], 'средняя'),
    ('Влажность низкая ИЛИ давление высокое', 'OR', [('Влажность (%)', 'низкая'), ('Давление (гПа)', 'высокое')], 'низкая'),
    ('Влажность высокая И давление среднее', 'AND', [('Влажность (%)', 'высокая'), ('Давление (гПа)', 'среднее')], 'высокая'),
    ('Влажность средняя И давление низкое', 'AND', [('Влажность (%)', 'средняя'), ('Давление (гПа)', 'низкое')], 'средняя'),
]


def mamdani(inputs):
    memberships = {
        name: {term: float(fn(inputs[name])) for term, fn in terms.items()}
        for name, terms in SETS.items() if name in inputs
    }
    output_name = 'Вероятность осадков (%)'
    x = DOMAINS[output_name]
    aggregated = np.zeros_like(x)
    activations = []
    for description, operator, conditions, result in RULES:
        values = [memberships[var][term] for var, term in conditions]
        strength = min(values) if operator == 'AND' else max(values)
        clipped = np.minimum(strength, SETS[output_name][result](x))
        aggregated = np.maximum(aggregated, clipped)
        activations.append((description, result, strength))
    area = np.trapezoid(aggregated, x)
    score = float(np.trapezoid(x * aggregated, x) / area) if area > 0 else None
    return memberships, aggregated, activations, score


def plot_sets(name, value=None, aggregated=None, score=None):
    x = DOMAINS[name]
    fig, ax = plt.subplots(figsize=(8, 3))
    for term, fn in SETS[name].items():
        ax.plot(x, fn(x), label=term.capitalize(), linewidth=2)
    if value is not None:
        ax.axvline(value, color='black', linestyle='--', label=f'Ввод: {value:g}')
    if aggregated is not None:
        ax.fill_between(x, aggregated, color='gray', alpha=0.35, label='Агрегированный результат')
    if score is not None:
        ax.axvline(score, color='red', linestyle=':', linewidth=2, label=f'Центроид: {score:.2f}%')
    ax.set(xlabel=name, ylabel='Степень принадлежности', ylim=(-0.03, 1.08))
    ax.grid(alpha=0.2)
    ax.legend(loc='upper center', ncol=3, fontsize=8)
    fig.tight_layout()
    return fig


st.title('Прогнозирование вероятности осадков')
st.caption('Нечёткая модель Мамдани: фаззификация → правила IF–THEN → агрегация → центроид')
with st.form('weather'):
    c1, c2, c3 = st.columns(3)
    with c1:
        temperature = st.number_input('Температура (°C)', min_value=-30.0, max_value=50.0, value=20.0, step=1.0)
    with c2:
        humidity = st.number_input('Влажность (%)', min_value=0.0, max_value=100.0, value=80.0, step=1.0)
    with c3:
        pressure = st.number_input('Давление (гПа)', min_value=950.0, max_value=1050.0, value=1005.0, step=1.0)
    submitted = st.form_submit_button('Рассчитать прогноз', type='primary')

if submitted:
    inputs = {'Температура (°C)': temperature, 'Влажность (%)': humidity, 'Давление (гПа)': pressure}
    if not all(np.isfinite(v) for v in inputs.values()):
        st.error('Введите корректные конечные числовые значения.')
    else:
        memberships, aggregated, activations, score = mamdani(inputs)
        if score is None:
            st.warning('Ни одно правило не активировано. Измените входные значения.')
        else:
            category = 'Низкая' if score < 35 else 'Средняя' if score < 65 else 'Высокая'
            st.metric('Оценка вероятности осадков', f'{score:.2f}%')
            st.success(f'Категория прогноза: {category}')
            st.info('Это условная оценка нечёткой модели, а не статистически откалиброванная вероятность или реальный метеопрогноз.')
        st.subheader('Фаззификация')
        for name, terms in memberships.items():
            st.write(f"**{name}:** " + ' · '.join(f'{term}: {degree:.3f}' for term, degree in terms.items()))
        st.subheader('Функции принадлежности')
        for name in inputs:
            fig = plot_sets(name, inputs[name])
            st.pyplot(fig)
            plt.close(fig)
        fig = plot_sets('Вероятность осадков (%)', aggregated=aggregated, score=score)
        st.pyplot(fig)
        plt.close(fig)
        st.subheader('Активация правил')
        st.dataframe([{'№': i, 'ЕСЛИ': desc, 'ТО вероятность': result, 'Активация': round(strength, 3)}
                      for i, (desc, result, strength) in enumerate(activations, 1)], hide_index=True, use_container_width=True)
else:
    st.info('Введите параметры и нажмите «Рассчитать прогноз».')

with st.expander('Описание метода'):
    st.write('Фаззификация вычисляет степени принадлежности. AND = min, OR = max. Выход каждого правила обрезается по его силе (min). Все выходы объединяются через max. Число получается как центр тяжести объединённой функции: ∫x·μ(x)dx / ∫μ(x)dx.')
    st.write('Модель использует 12 правил, включая правила с AND и OR. Повторный расчёт доступен без перезапуска.')
