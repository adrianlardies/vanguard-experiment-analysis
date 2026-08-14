import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import skew, kurtosis
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import chi2_contingency

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEANED_DATA_PATH = PROJECT_ROOT / 'data' / 'cleaned' / 'vanguard.csv'

st.markdown('# **Historical Vanguard A/B Test Analysis** 📊')
st.caption('Historical portfolio project; no employment or client relationship with Vanguard is implied.')

st.sidebar.title('Opciones de Análisis')

grupo_seleccionado = st.sidebar.selectbox('Seleccione el grupo', ['Ambos', 'Test', 'Control'])

usuario_tipo = st.sidebar.radio('Seleccione el tipo de usuario', ['Todos', 'Lineal', 'No Lineal'])

paso_seleccionado = st.sidebar.selectbox('Seleccione el paso a analizar', ['Todos', 'start', 'step_1', 'step_2', 'step_3', 'confirm'])

@st.cache_data
def cargar_datos():
    df = pd.read_csv(CLEANED_DATA_PATH, index_col=None)
    return df

df_vanguard = cargar_datos()

df_filtrado = df_vanguard.copy()

if grupo_seleccionado != 'Ambos':
    df_filtrado = df_filtrado[df_filtrado['variation'] == grupo_seleccionado]

if usuario_tipo == 'Lineal':
    df_filtrado = df_filtrado[df_filtrado['lineal'] == True]
elif usuario_tipo == 'No Lineal':
    df_filtrado = df_filtrado[df_filtrado['lineal'] == False]

if paso_seleccionado != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['step'] == paso_seleccionado]

def calculate_completion_rates(df):
    total_clientes_por_variacion = df.drop_duplicates(subset=['client_id', 'variation']).groupby('variation')['client_id'].nunique()

    lineal_true = df[(df['lineal'] == True)].drop_duplicates(subset=['client_id', 'variation'])
    lineal_false = df[(df['lineal'] == False) & (~df['client_id'].isin(lineal_true['client_id']))].drop_duplicates(subset=['client_id', 'variation'])

    completion_rate_lineal_true = lineal_true.groupby('variation')['client_id'].nunique() / total_clientes_por_variacion

    completion_rate_lineal_false = lineal_false.groupby('variation')['client_id'].nunique() / total_clientes_por_variacion

    suma_tasas = completion_rate_lineal_true + completion_rate_lineal_false

    return completion_rate_lineal_true, completion_rate_lineal_false, suma_tasas

def plot_completion_rates(completion_rate_true, completion_rate_false):
    completion_rates = pd.DataFrame({
        'Lineal=True': completion_rate_true,
        'Lineal=False': completion_rate_false
    })

    completion_rates.plot(kind='bar', figsize=(10, 6), color=['#4CAF50', '#FF5733'])

    plt.title('Tasa de Secuencia Lineal y No Lineal por Grupo (Test vs Control)')
    plt.xlabel('Grupo')
    plt.ylabel('Proporción de Clientes')
    plt.xticks(rotation=0)
    plt.legend(title='Condición')
    plt.show()

def plot_avg_time_in_steps(df, step_order):
    df['lineal'] = df['lineal'].astype(bool)
    
    df['step'] = pd.Categorical(df['step'], categories=step_order, ordered=True)

    avg_time_in_step_lineal = df.groupby(['variation', 'lineal', 'step'])['total_time_in_step'].mean().unstack()

    avg_time_in_step_lineal = avg_time_in_step_lineal[step_order]

    plt.figure(figsize=(12, 6))

    for variation, lineal in avg_time_in_step_lineal.index:
        if lineal:
            label = f"{variation} - Lineal"
            marker = 'o'
            linestyle = '-'
        else:
            label = f"{variation} - No Lineal"
            marker = 'x'
            linestyle = '--'
        
        avg_time_in_step_lineal.loc[(variation, lineal), :].T.plot(
            kind='line', marker=marker, linestyle=linestyle, figsize=(12, 6), ax=plt.gca(), label=label
        )

    plt.title('Valor Acumulado Promedio por Paso: Test vs Control (Lineal vs No Lineal)')
    plt.xlabel('Paso')
    plt.ylabel('Valor Acumulado Promedio (segundos)')
    plt.grid(True)
    plt.legend(title='Condición (Test/Control - Lineal/No Lineal)', loc='upper left')
    plt.show()

    return avg_time_in_step_lineal

def plot_error_rate(df, step_order):
    df['error'] = (df['step_diff'] == -1) | (df['step_repeat_count'] > 2)

    error_rate = df.groupby(['variation', 'lineal', 'step'])['error'].mean().unstack()

    error_rate = error_rate[step_order]

    return error_rate

def calculate_and_plot_time_stats(df, step_order):
    grouped_time_stats = df.groupby(['variation', 'step', 'lineal'])['total_time_in_step'].agg(
        mean='mean',
        skew=lambda x: skew(x),
        kurtosis=lambda x: kurtosis(x)
    )

    grouped_time_stats = grouped_time_stats.reindex(step_order, level='step')

    fig, ax = plt.subplots(figsize=(10, 6))
    grouped_time_stats['mean'].unstack().plot(kind='bar', ax=ax)
    ax.set_title('Mean Time per Step: Lineal vs Non-lineal (Test vs Control)')
    ax.set_ylabel('Mean Time (seconds)')
    ax.set_xlabel('Steps')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    grouped_time_stats['skew'].unstack().plot(kind='bar', ax=ax)
    ax.set_title('Skewness per Step: Lineal vs Non-lineal (Test vs Control)')
    ax.set_ylabel('Skewness')
    ax.set_xlabel('Steps')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    grouped_time_stats['kurtosis'].unstack().plot(kind='bar', ax=ax)
    ax.set_title('Kurtosis per Step: Lineal vs Non-lineal (Test vs Control)')
    ax.set_ylabel('Kurtosis')
    ax.set_xlabel('Steps')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()
    
    return grouped_time_stats

def calculate_and_plot_time_per_step(df, step_order):
    df_time_per_step = df.groupby(['variation', 'step'])['total_time_in_step'].mean().unstack()

    df_time_per_step = df_time_per_step[step_order]

    df_time_per_step.T.plot(kind='line', marker='o', figsize=(10, 6))
    plt.title('Valor Acumulado Promedio por Paso: Test vs Control')
    plt.xlabel('Paso')
    plt.ylabel('Valor Acumulado Promedio')
    plt.grid(True)
    plt.show()

    return df_time_per_step

def calculate_pearson_spearman_corr(df):
    pearson_corr = df[['balance', 'activity', 'age', 'num_accounts']].corr(method='pearson')
    spearman_corr = df[['balance', 'activity', 'age', 'num_accounts']].corr(method='spearman')

    plt.figure(figsize=(10, 6))
    sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Pearson Correlation Heatmap')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Spearman Correlation Heatmap')
    plt.show()

    return pearson_corr, spearman_corr

def z_test_completion_rates(df):
    n_test = df[df['variation'] == 'Test']['client_id'].nunique()  # Número de usuarios en Test
    n_control = df[df['variation'] == 'Control']['client_id'].nunique()  # Número de usuarios en Control

    completed_test = df[(df['variation'] == 'Test') & (df['lineal'] == True)]['client_id'].nunique()  # Clientes Test con alguna visita lineal
    completed_control = df[(df['variation'] == 'Control') & (df['lineal'] == True)]['client_id'].nunique()  # Clientes Control con alguna visita lineal

    p_test = completed_test / n_test
    p_control = completed_control / n_control

    count = [completed_test, completed_control]
    nobs = [n_test, n_control]

    stat, p_value = proportions_ztest(count, nobs)

    return stat, p_value

def t_test_time_per_step(df, step):
    time_test_lineal = df[(df['variation'] == 'Test') & (df['lineal'] == True) & (df['step'] == step)]['total_time_in_step']
    time_control_lineal = df[(df['variation'] == 'Control') & (df['lineal'] == True) & (df['step'] == step)]['total_time_in_step']

    time_test_non_lineal = df[(df['variation'] == 'Test') & (df['lineal'] == False) & (df['step'] == step)]['total_time_in_step']
    time_control_non_lineal = df[(df['variation'] == 'Control') & (df['lineal'] == False) & (df['step'] == step)]['total_time_in_step']

    t_stat_lineal, p_value_lineal = stats.ttest_ind(time_test_lineal, time_control_lineal, equal_var=False)

    t_stat_non_lineal, p_value_non_lineal = stats.ttest_ind(time_test_non_lineal, time_control_non_lineal, equal_var=False)

    return (t_stat_lineal, p_value_lineal), (t_stat_non_lineal, p_value_non_lineal)

def mann_whitney_test_variation(df):
    test_group = df[df['variation'] == 'Test']['total_time_in_step']
    control_group = df[df['variation'] == 'Control']['total_time_in_step']

    u_statistic, p_value = stats.mannwhitneyu(test_group, control_group, alternative='two-sided')

    return u_statistic, p_value

def mann_whitney_test_lineal(df):
    lineal_group = df[df['lineal'] == True]['total_time_in_step']
    non_lineal_group = df[df['lineal'] == False]['total_time_in_step']

    u_statistic, p_value = stats.mannwhitneyu(lineal_group, non_lineal_group, alternative='two-sided')

    return u_statistic, p_value

def chi_square_and_cramers_v(df):
    contingency_table = pd.crosstab(df['variation'], df['lineal'])

    chi2, p, dof, ex = chi2_contingency(contingency_table)

    n = contingency_table.sum().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))

    return chi2, p, cramers_v

df_vanguard = cargar_datos()

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Introducción",
    "Tasa de Secuencia Lineal",
    "Tiempos por Paso", 
    "Tasa de Errores", 
    "Estadísticas de Tiempos", 
    "Correlaciones", 
    "Pruebas Estadísticas"
])

with tab1:
    st.markdown('''
    ## Introducción
    
    Este proyecto histórico compara el comportamiento registrado en un experimento A/B entre una **nueva versión** de una experiencia digital (Test) y su **versión anterior** (Control). El análisis describe recorridos, tiempos y errores observados; no implica una relación laboral o de cliente con Vanguard.
    ''')
                
    st.divider()    

    st.markdown('''
    El análisis se centra en tres métricas fundamentales:

    - **Tasa de secuencia lineal**: Analiza el porcentaje de clientes con al menos una visita que cumple la regla histórica de secuencia completa.
    - **Tiempos registrados por evento**: Compara el tiempo acumulado desde el inicio de la visita en cada etapa del proceso.
    - **Tasa de errores**: Mide la frecuencia de los eventos que cumplen la regla histórica de error: un retroceso de un paso o más de dos apariciones del mismo paso en una visita.
    ''')
                
    st.divider()            

    st.markdown('''
    El análisis histórico emplea diversas **pruebas estadísticas** para comparar las diferencias observadas entre ambas versiones:

    - **Pruebas T-test y Mann-Whitney**: Comparan los valores de tiempo acumulado registrados por evento entre los grupos.
    - **Pruebas Chi-square y Cramér's V**: Analizan la relación entre variables categóricas, como el tipo de usuario (lineal o no lineal) y la versión de la web utilizada (Test o Control).
    - **Correlaciones de Pearson y Spearman**: Describen la relación entre variables continuas como la **edad**, el **balance**, la **actividad** (número de logins y llamadas al soporte) y el **número de cuentas**.
    ''')

    st.divider()            

    st.markdown('''
    El objetivo es comparar los patrones observados entre Test y Control. Las métricas permiten describir diferencias en los logs, pero por sí solas no demuestran una mejora global de la experiencia de usuario.
    ''')

with tab2:
    st.subheader("Tasas de Secuencia Lineal")
    
    st.write("""
    Esta métrica histórica mide el porcentaje de clientes con al menos una visita clasificada como `lineal=True`.
    El análisis se divide entre:
    
    - **Lineal=True**: La visita contiene `start → step_1 → step_2 → step_3 → confirm` en ese orden, con cada paso presente una o dos veces y sin otros eventos intercalados.
    - **Lineal=False**: La visita no satisface esa regla; esto incluye recorridos incompletos, desordenados o con más repeticiones.
    """)

    completion_rate_true, completion_rate_false, suma_tasas = calculate_completion_rates(df_filtrado)

    st.write("##### Tasa de clientes con alguna visita lineal")
    st.write(completion_rate_true)

    st.write("##### Tasa de clientes sin visitas lineales")
    st.write(completion_rate_false)

    st.write("##### Suma de tasas para cada grupo (debería ser cercana a 1)")
    st.write(suma_tasas)

    st.divider()

    st.subheader("Visualización")
    
    completion_rates = pd.DataFrame({
        'Lineal=True': completion_rate_true,
        'Lineal=False': completion_rate_false
    })

    completion_rates = completion_rates.reset_index()

    fig = px.bar(completion_rates, x='variation', y=['Lineal=True', 'Lineal=False'], barmode='group', title="Tasas de Secuencia Lineal y No Lineal (Test vs Control)")
    st.plotly_chart(fig)

    st.divider()

    st.write("""
    #### Insights
    - En el grupo **Control**, el **46.05%** de los clientes tuvo al menos una visita lineal, frente al **47.46%** en **Test**. Esta diferencia descriptiva es pequeña.
    
    - El porcentaje de clientes sin ninguna visita lineal también se mantiene cercano entre los grupos. En **Control** es **53.94%**, frente a **52.53%** en **Test**.

    - La clasificación no lineal sigue siendo común en ambas versiones. La métrica no distingue abandono de otros motivos por los que una visita incumple la regla.
    
    - Sería necesario un análisis adicional para distinguir recorridos incompletos, retrocesos y repeticiones, sin inferir motivos de usabilidad a partir de los logs únicamente.
    """)

with tab3:
    st.subheader("Valores Acumulados Promedio por Paso")
    
    st.write("""
    En este apartado se analiza el tiempo acumulado registrado en cada paso del proceso de navegación,
    comparando los grupos Test (nueva versión de la web) y Control (versión antigua), y diferenciando entre usuarios lineales 
    (visitas que satisfacen la regla de secuencia) y no lineales (visitas que no la satisfacen).
    """)

    step_order = ['start', 'step_1', 'step_2', 'step_3', 'confirm']

    avg_time_in_step_lineal = plot_avg_time_in_steps(df_filtrado, step_order)
    
    st.write(avg_time_in_step_lineal)

    st.divider()

    st.subheader("Visualización")

    df_grouped_time = df_filtrado.groupby(['variation', 'step', 'lineal'])['total_time_in_step'].mean().reset_index()

    df_grouped_time['step'] = pd.Categorical(df_grouped_time['step'], categories=step_order, ordered=True)

    fig_time = px.line(
        df_grouped_time, 
        x='step', 
        y='total_time_in_step', 
        color='variation', 
        line_dash='lineal',  # Agregar la variable lineal/no lineal como patrón de línea
        markers=True, 
        title="Valores Acumulados Promedio por Paso (Test vs Control - Lineal/No Lineal)"
    )
    
    st.plotly_chart(fig_time)

    st.divider()

    st.write("""
    #### Insights
    - **Visitas no lineales**: Los eventos de visitas no lineales de **Test** muestran valores medios de tiempo acumulado superiores en los primeros pasos. Los logs no permiten atribuir este patrón a usabilidad, confusión u otra causa.
    
    - **Visitas lineales**: Los eventos de visitas lineales del grupo **Test** muestran menores valores medios de tiempo acumulado que los del grupo **Control**. Es una diferencia descriptiva y no demuestra por sí sola una experiencia más eficiente.
    
    - **Pasos intermedios**: Los valores medios acumulados aumentan en **Step 2** y **Step 3** para ambos grupos. Esta métrica no permite atribuir el patrón a una mayor complejidad de la interfaz.
    
    - **Paso Confirm**: El valor medio acumulado en **Confirm** es menor para las visitas no lineales de Test que para las de Control; no se evalúa la causa de esa diferencia.
    
    - **Conclusión general**: Los patrones varían por paso y clasificación de la visita. No respaldan por sí solos una afirmación de mejora general de eficiencia o experiencia.
    """)

with tab4:
    st.subheader("Tasas de Error")
    
    st.write("""
    La tasa histórica de error mide la proporción de eventos que cumplen una de dos reglas: un retroceso de un paso
    (`step_diff == -1`) o más de dos apariciones del mismo paso dentro de una visita. No captura otras posibles
    dificultades ni identifica sus causas.
    """)

    step_order = ['start', 'step_1', 'step_2', 'step_3', 'confirm']

    error_rate = plot_error_rate(df_filtrado, step_order)
    
    error_rate_reset = error_rate.reset_index()

    error_rate_melted = error_rate_reset.melt(id_vars=['variation', 'lineal'], value_vars=step_order, var_name='step', value_name='error_rate')

    error_rate_filtered = error_rate_melted[error_rate_melted['lineal'] == False]

    st.write(error_rate_filtered)

    st.divider()

    st.subheader("Visualización")
    fig_error = px.line(
        error_rate_melted, 
        x='step', 
        y='error_rate', 
        color='variation', 
        line_dash='lineal',  # Diferenciar lineales y no lineales con líneas diferentes
        markers=True,
        title="Tasa de Errores por Grupo (Test vs Control - Lineal/No Lineal)",
        labels={"error_rate": "Tasa de Error"}
    )
    st.plotly_chart(fig_error)

    st.divider()

    st.write("""
    #### Insights
    - **Paso Start y visitas no lineales**: La tasa histórica de error es mayor en **Test** (34.62%) que en **Control** (23.37%). Es una diferencia descriptiva; los logs no identifican barreras ni explican la causa.
    
    - **Evolución de las tasas de error**: En Test, las tasas históricas son **20.85%** en **Step 1**, **21.29%** en **Step 2** y **7.91%** en **Step 3**. En Control son **15.18%**, **22.89%** y **10.38%**, respectivamente. Son comparaciones descriptivas de eventos clasificados como error.
    
    - **Paso Confirm**: La tasa histórica de error es ligeramente superior en **Test** (**8.78%**) que en **Control** (**7.68%**). La métrica no permite atribuir la diferencia a confusión u otra causa.
    
    - **Usuarios lineales**: No se detectan errores definidos por esta regla en las visitas lineales, un resultado esperado por la relación entre ambas definiciones. Esto no demuestra que la experiencia estuviera libre de otros problemas.
    
    - **Conclusión general**: La tasa de error definida por el proyecto es mayor en **Start** para las visitas no lineales de Test. Los logs no permiten concluir que la interfaz sea menos intuitiva ni identificar la causa.
    """)

with tab5:
    st.subheader("Estadísticas de Tiempos: Media, Skewness y Kurtosis")
    
    st.write("""
    Además del valor medio, se describe la forma de la distribución histórica de `total_time_in_step`
    en cada paso mediante las siguientes métricas:
    
    - **Media**: El valor acumulado promedio registrado para los eventos del paso.
    - **Skewness (asimetría)**: Describe la asimetría de los valores acumulados respecto a la media.
    - **Kurtosis (curtosis)**: Describe el peso de las colas de la distribución y la presencia de valores extremos.
    """)

    st.divider()

    step_order = ['start', 'step_1', 'step_2', 'step_3', 'confirm']

    grouped_time_stats = calculate_and_plot_time_stats(df_filtrado, step_order)

    grouped_time_stats = grouped_time_stats.reset_index()

    st.write(grouped_time_stats)

    st.divider()

    st.subheader("Visualización")

    fig_mean = px.bar(grouped_time_stats, x='step', y='mean', color='variation', 
                      barmode='group', title='Media de Valores Acumulados por Grupo y Paso',
                      labels={'mean': 'Valor acumulado promedio (s)', 'step': 'Paso'})
    st.plotly_chart(fig_mean)

    fig_skewness = px.bar(grouped_time_stats, x='step', y='skew', color='variation', 
                          barmode='group', title='Skewness por Grupo y Paso',
                          labels={'skew': 'Skewness', 'step': 'Paso'})
    st.plotly_chart(fig_skewness)

    fig_kurtosis = px.bar(grouped_time_stats, x='step', y='kurtosis', color='variation', 
                          barmode='group', title='Kurtosis por Grupo y Paso',
                          labels={'kurtosis': 'Kurtosis', 'step': 'Paso'})
    st.plotly_chart(fig_kurtosis)

    st.divider()

    st.markdown("""
    ### Insights

    1. **Media (mean)**:
   - En general, los eventos de **visitas no lineales** muestran valores medios acumulados superiores en todos los pasos, especialmente en los intermedios y finales. En **Test**, por ejemplo, la media es 194 segundos en **step_3**, frente a 157 segundos para las visitas lineales.
   - En los primeros pasos (**start** y **step_1**), las visitas lineales de **Control** registran medias de 12 y 37 segundos, frente a 7 y 30 segundos en **Test**. Los patrones cambian en pasos posteriores y no respaldan una conclusión global sobre rapidez.
   - En **confirm**, las visitas lineales de **Control** tienen un valor medio acumulado superior (290 segundos) al de **Test** (250 segundos). Es una diferencia descriptiva de esta métrica.

    2. **Skewness (asimetría)**:
   - En ambos grupos, la **skewness** es muy alta en los primeros pasos (**start** y **step_1**), especialmente para las **visitas no lineales**, lo que refleja distribuciones con una cola derecha pronunciada. En **Control**, el paso **start** tiene una skewness de 53 para esas visitas.
   - La asimetría disminuye en pasos posteriores, aunque sigue siendo alta en **confirm** para las visitas no lineales de **Test** (9.27). La distribución no permite identificar dificultades ni sus causas.

    3. **Kurtosis**:
   - Los valores extremadamente altos de **kurtosis** en los primeros pasos, especialmente para las **visitas no lineales**, reflejan colas pesadas y valores extremos. Por ejemplo, **start** en **Control** tiene una kurtosis de 5255 para esas visitas.
   - En el grupo **Test**, la **kurtosis** es alta en **step_1** y **confirm**, con valores de 403 y 280 respectivamente, lo que refleja colas pesadas o valores extremos sin identificar su causa.
   - Las **visitas lineales** de ambos grupos muestran una **kurtosis más baja** que las no lineales, una descripción de la forma de estas distribuciones que no demuestra una experiencia más consistente.
    """)

with tab6:
    st.subheader("Correlaciones entre Variables")
    
    st.write("""
    En esta sección se analizan las correlaciones entre diferentes variables relacionadas con los clientes, 
    como **balance**, **edad**, **actividad** (suma de logins y llamadas a soporte), y **número de cuentas**. 
    El objetivo es describir asociaciones bivariadas en los datos. Estas correlaciones no establecen influencia,
    capacidad predictiva ni causalidad.
    """)

    st.divider()
    
    st.write("""
    ### Correlaciones de Pearson y Spearman
    Se utilizan dos métodos para calcular las correlaciones:
    
    - **Correlación de Pearson**: Mide la relación lineal entre dos variables continuas. 
      Un valor cercano a 1 indica una fuerte correlación positiva, mientras que un valor cercano a -1 indica 
      una fuerte correlación negativa.
    - **Correlación de Spearman**: Mide la correlación monótona entre dos variables, 
      lo que permite detectar relaciones que no son estrictamente lineales.
    """)

    st.divider()

    pearson_corr, spearman_corr = calculate_pearson_spearman_corr(df_filtrado)

    st.subheader("Matriz de Correlación de Pearson")
    st.write(pearson_corr)

    st.subheader("Matriz de Correlación de Spearman")
    st.write(spearman_corr)

    st.divider()

    st.subheader("Visualización")

    fig_pearson = px.imshow(pearson_corr, text_auto=True, aspect="auto", title="Correlación de Pearson")
    st.plotly_chart(fig_pearson)

    fig_spearman = px.imshow(spearman_corr, text_auto=True, aspect="auto", title="Correlación de Spearman")
    st.plotly_chart(fig_spearman)

    st.divider()

    st.markdown("""
    ### Insights

    1. **Balance y Número de Cuentas**:
   - La correlación entre **balance** y **número de cuentas** es positiva tanto en Pearson (**0.26**) como en Spearman (**0.33**). El análisis no incluye una prueba de significancia para estas correlaciones.

    2. **Balance y Edad**:
   - **Balance** y **edad** muestran una correlación positiva, con un valor de Spearman de **0.35**. La diferencia respecto a Pearson describe los datos, pero no prueba que una relación sea más sólida.

    3. **Balance y Actividad**:
   - La correlación entre **balance** y **actividad** es positiva pero baja: Pearson **0.17** y Spearman **0.29**. No se evaluó capacidad predictiva.

    4. **Actividad y Número de Cuentas**:
   - **Actividad** y **número de cuentas** presentan valores de **0.24** tanto en Pearson como en Spearman, una asociación positiva de magnitud limitada.

    5. **Edad y Actividad**:
   - La correlación entre **edad** y **actividad** es muy baja tanto en Pearson (**0.08**) como en Spearman (**0.08**). No se realizó una prueba que permita afirmar impacto o ausencia de impacto.
    """)

with tab7:
    st.subheader("Pruebas Estadísticas")

    st.write("""
    El análisis histórico compara **Test** (nueva versión) y **Control** (versión anterior) mediante varias pruebas.
    Sus valores p describen la compatibilidad de las métricas observadas —como secuencia lineal y valores acumulados—
    con la hipótesis nula bajo los supuestos de cada prueba; no establecen importancia práctica ni causalidad.
    """)

    st.divider()

    st.write("""
    ### Tipos de Pruebas Realizadas:
    
    - **Prueba Z**: Comparación de las tasas de clientes con alguna visita lineal entre Test y Control.
    - **T-test**: Comparación de los valores acumulados en `confirm` entre Test y Control, para filas lineales y no lineales.
    - **Prueba Mann-Whitney**: Evaluación de diferencias en los valores acumulados de las filas de eventos.
    - **Prueba Chi-square**: Evaluación de la asociación entre las variables categóricas, como Test/Control y Lineal/No Lineal.
    - **Cramér's V**: Medición de la fuerza de la asociación entre variables categóricas.
    """)

    st.divider()

    st.markdown("#### 1. Prueba Z para Tasas de Secuencia Lineal")
    stat, p_value = z_test_completion_rates(df_vanguard)
    st.write(f"**Estadístico Z**: {stat:.3f}")
    st.write(f"**Valor p**: {p_value:.5f}")
    
    if p_value < 0.05:
        st.write("La diferencia en las tasas de clientes con alguna visita lineal es **estadísticamente significativa** bajo esta prueba.")
    else:
        st.write("La diferencia en las tasas de clientes con alguna visita lineal **no es estadísticamente significativa** bajo esta prueba.")

    st.markdown("""
    ##### Insight
    - El estadístico Z muestra una diferencia significativa en la proporción histórica de clientes con alguna visita lineal (valor p **0.0015**). Esto indica asociación, no prueba por sí solo una mejora de experiencia.
    """)

    st.divider()

    st.markdown("#### 2. T-test para Valores Acumulados en Confirm")
    step = 'confirm'
    (t_stat_lineal, p_value_lineal), (t_stat_non_lineal, p_value_non_lineal) = t_test_time_per_step(df_vanguard, step)

    st.write("**Resultados para usuarios lineales:**")
    st.write(f"**Estadístico t**: {t_stat_lineal:.2f}")
    st.write(f"**Valor p**: {p_value_lineal:.2e}")
    if p_value_lineal < 0.05:
        st.write("La diferencia en los valores acumulados de `confirm` para filas lineales es **estadísticamente significativa** bajo esta prueba.")
    else:
        st.write("La diferencia en los valores acumulados de `confirm` para filas lineales **no es estadísticamente significativa** bajo esta prueba.")
    
    st.write("**Resultados para usuarios no lineales:**")
    st.write(f"**Estadístico t**: {t_stat_non_lineal:.2f}")
    st.write(f"**Valor p**: {p_value_non_lineal:.2e}")
    if p_value_non_lineal < 0.05:
        st.write("La diferencia en los valores acumulados de `confirm` para filas no lineales es **estadísticamente significativa** bajo esta prueba.")
    else:
        st.write("La diferencia en los valores acumulados de `confirm` para filas no lineales **no es estadísticamente significativa** bajo esta prueba.")

    st.markdown("""
    ##### Insight
    - Para las filas **lineales**, el valor p es muy bajo (**2.18e-40**) bajo esta prueba histórica.
    - Para las filas **no lineales**, el valor p también es muy bajo (**8.11e-21**). Estas observaciones se repiten dentro de visitas y clientes, por lo que la independencia es limitada.
    """)

    st.divider()

    st.markdown("#### 3. Prueba Mann-Whitney U para Valores Acumulados por Evento")
    u_statistic_variation, p_value_variation = mann_whitney_test_variation(df_vanguard)
    st.write(f"**U-statistic (Test vs Control)**: {u_statistic_variation:,.0f}")
    st.write(f"**P-value (Test vs Control)**: {p_value_variation:.2e}")
    
    if p_value_variation < 0.05:
        st.write("La diferencia entre los grupos Test y Control es **estadísticamente significativa**.")
    else:
        st.write("No hay una diferencia estadísticamente significativa entre los grupos Test y Control.")
    
    u_statistic_lineal, p_value_lineal = mann_whitney_test_lineal(df_vanguard)
    st.write(f"**U-statistic (Lineal vs No Lineal)**: {u_statistic_lineal:,.0f}")
    st.write(f"**P-value (Lineal vs No Lineal)**: {p_value_lineal:.2e}")
    
    if p_value_lineal < 0.05:
        st.write("La diferencia entre usuarios lineales y no lineales es **estadísticamente significativa**.")
    else:
        st.write("No hay una diferencia estadísticamente significativa entre usuarios lineales y no lineales.")

    st.markdown("""
    ##### Insight
    - La prueba Mann-Whitney U muestra una diferencia en los valores acumulados por evento entre Test y Control (valor p **1.59e-08**) bajo sus supuestos.
    - La comparación entre filas **lineales** y **no lineales** también produce un valor p muy bajo (**7.29e-277**). Las filas se repiten dentro de visitas y clientes, por lo que la independencia es limitada.
    """)

    st.divider()

    st.markdown("#### 4. Prueba Chi-square y Cramér's V")
    chi2, p_value, cramers_v = chi_square_and_cramers_v(df_vanguard)
    st.write(f"**Chi-square statistic**: {chi2:.2f}")
    st.write(f"**P-value**: {p_value:.2e}")
    st.write(f"**Cramér's V**: {cramers_v:.4f}")
    
    if p_value < 0.05:
        st.write("La asociación entre 'variation' y 'lineal' es **estadísticamente significativa**.")
    else:
        st.write("No hay una asociación estadísticamente significativa entre 'variation' y 'lineal'.")

    st.markdown("""
    ##### Insight
    - El estadístico Chi-square es **133.73** con un valor p de **6.26e-31**, lo que indica que hay una asociación significativa entre los grupos Test/Control y la categorización lineal/no lineal.
    - El valor de **Cramér's V** es **0.0205**, lo que indica que aunque la asociación es estadísticamente significativa, la fuerza de la relación entre las variables es débil.
    """)