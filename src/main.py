import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import skew, kurtosis
from scipy.stats import chi2_contingency
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'

# Función para cargar los datos desde archivos CSV
def cargar_datos():
    df_final_demo = pd.read_csv(RAW_DATA_DIR / 'df_final_demo.txt')
    df_final_experiment_clients = pd.read_csv(RAW_DATA_DIR / 'df_final_experiment_clients.txt')
    df_pt_1 = pd.read_csv(RAW_DATA_DIR / 'df_pt_1.txt')
    df_pt_2 = pd.read_csv(RAW_DATA_DIR / 'df_pt_2.txt')
    
    # Concatenar df_pt_1 y df_pt_2
    df_pt = pd.concat([df_pt_1, df_pt_2], ignore_index=True)
    
    return df_final_demo, df_final_experiment_clients, df_pt

# Función para limpiar los datos eliminando duplicados y valores nulos
def limpiar_datos(df_final_demo, df_final_experiment_clients, df_pt):
    df_final_demo = df_final_demo.dropna().drop_duplicates()
    df_final_experiment_clients = df_final_experiment_clients.dropna()
    df_pt = df_pt.drop_duplicates()
    
    return df_final_demo, df_final_experiment_clients, df_pt

# Función para filtrar por client_id comunes en las tablas
def filtrar_por_clientes_comunes(df_final_demo, df_final_experiment_clients, df_pt):
    client_ids_demo = set(df_final_demo['client_id'])
    client_ids_experiment = set(df_final_experiment_clients['client_id'])
    client_ids_pt = set(df_pt['client_id'])
    
    # Intersección de client_ids en común
    client_ids_comunes = client_ids_demo.intersection(client_ids_experiment).intersection(client_ids_pt)

    # Filtrar las tablas por client_ids en común
    df_final_demo = df_final_demo[df_final_demo['client_id'].isin(client_ids_comunes)]
    df_final_experiment_clients = df_final_experiment_clients[df_final_experiment_clients['client_id'].isin(client_ids_comunes)]
    df_pt = df_pt[df_pt['client_id'].isin(client_ids_comunes)]

    return df_final_demo, df_final_experiment_clients, df_pt

# Convertir tipos de datos en df_final_demo
def convertir_tipos_datos(df_final_demo, df_pt):
    # Convertir columnas en df_final_demo a enteros
    df_final_demo['clnt_tenure_yr'] = df_final_demo['clnt_tenure_yr'].astype(int)
    df_final_demo['clnt_tenure_mnth'] = df_final_demo['clnt_tenure_mnth'].astype(int)
    df_final_demo['num_accts'] = df_final_demo['num_accts'].astype(int)
    df_final_demo['calls_6_mnth'] = df_final_demo['calls_6_mnth'].astype(int)
    df_final_demo['logons_6_mnth'] = df_final_demo['logons_6_mnth'].astype(int)
    
    # Convertir date_time a formato datetime en df_pt
    df_pt['date_time'] = pd.to_datetime(df_pt['date_time'])
    
    return df_final_demo, df_pt

# Renombrar columnas y fusionar datasets
def renombrar_columnas_y_fusionar(df_final_experiment_clients, df_final_demo, df_pt):
    # Renombrar las columnas de df_pt
    df_pt = df_pt.rename(columns={'process_step': 'step'})
    
    # Renombrar columnas en df_final_demo
    df_final_demo = df_final_demo.rename(columns={
        'clnt_tenure_yr': 'tenure_years',
        'clnt_tenure_mnth': 'tenure_months',
        'clnt_age': 'age',
        'gendr': 'gender',
        'num_accts': 'num_accounts',
        'bal': 'balance',
        'calls_6_mnth': 'calls_6_months',
        'logons_6_mnth': 'logons_6_months'
    })
    
    # Renombrar df_final_experiment_clients a df_final y renombrar su columna 'Variation' a 'variation'
    df_final_experiment_clients = df_final_experiment_clients.rename(columns={'Variation': 'variation'})
    
    # Fusionar df_final_demo y df_final_experiment_clients en un único DataFrame (df_final)
    df_final = pd.merge(df_final_experiment_clients, df_final_demo, on='client_id', how='inner')
    
    return df_final, df_pt

# Función para calcular medias y percentiles de las métricas clave
def calcular_medias_percentiles(df_final):
    mean_tenure = df_final['tenure_years'].mean()
    mean_balance = df_final['balance'].mean()
    mean_num_accounts = df_final['num_accounts'].mean()
    mean_activity = (df_final['calls_6_months'] + df_final['logons_6_months']).mean()

    percentile_90_tenure = df_final['tenure_years'].quantile(0.90)
    percentile_90_balance = df_final['balance'].quantile(0.90)
    percentile_90_num_accounts = df_final['num_accounts'].quantile(0.90)
    percentile_90_activity = (df_final['calls_6_months'] + df_final['logons_6_months']).quantile(0.90)
    
    return mean_tenure, mean_balance, mean_num_accounts, mean_activity, percentile_90_tenure, percentile_90_balance, percentile_90_num_accounts, percentile_90_activity

# Función para clasificar los clientes en diferentes grupos
def clasificar_clientes(df_final, medias, percentiles):
    mean_tenure, mean_balance, mean_num_accounts, mean_activity = medias
    percentile_90_tenure, percentile_90_balance, percentile_90_num_accounts, percentile_90_activity = percentiles
    
    df_above_mean = df_final[
        (df_final['tenure_years'] > mean_tenure) &
        (df_final['balance'] > mean_balance) &
        (df_final['num_accounts'] > mean_num_accounts) &
        ((df_final['calls_6_months'] + df_final['logons_6_months']) > mean_activity)
    ]

    df_top_10 = df_final[
        (df_final['tenure_years'] >= percentile_90_tenure) &
        (df_final['balance'] >= percentile_90_balance) &
        (df_final['num_accounts'] >= percentile_90_num_accounts) &
        ((df_final['calls_6_months'] + df_final['logons_6_months']) >= percentile_90_activity)
    ]

    df_final['group'] = 'General'
    df_final.loc[df_final['client_id'].isin(df_top_10['client_id']), 'group'] = 'Top 10%'
    df_final.loc[
        (df_final['tenure_years'] > mean_tenure) &
        (df_final['balance'] > mean_balance) &
        (df_final['num_accounts'] > mean_num_accounts) &
        ((df_final['calls_6_months'] + df_final['logons_6_months']) > mean_activity) &
        (df_final['group'] == 'General'),
        'group'
    ] = 'Above Mean'

    df_final['activity'] = df_final['calls_6_months'] + df_final['logons_6_months']
    df_final['age_group'] = pd.cut(df_final['age'], bins=[0, 35, 55, 100], labels=['Young', 'Middle-aged', 'Senior'])
    df_final['balance_category'] = pd.qcut(df_final['balance'], q=3, labels=['Low', 'Medium', 'High'])
    df_final['activity_category'] = pd.qcut(df_final['activity'], q=3, labels=['Low', 'Medium', 'High'])
    df_final['tenure_category'] = pd.cut(df_final['tenure_years'], bins=[0, 5, 10, 20, 50], labels=['0-5', '6-10', '11-20', '21+'])

    return df_final

# Función para analizar pasos en df_pt
def analizar_pasos(df_pt):
    df_pt = df_pt.sort_values(by=['client_id', 'visit_id', 'date_time'])
    
    step_order = {'start': 1, 'step_1': 2, 'step_2': 3, 'step_3': 4, 'confirm': 5}
    df_pt['step_order'] = df_pt['step'].map(step_order)
    
    df_pt['step_diff'] = df_pt.groupby(['client_id', 'visit_id'])['step_order'].diff().fillna(0)
    df_pt['non_linear'] = df_pt['step_diff'].apply(lambda x: x < 0)
    df_pt['step_repeat_count'] = df_pt.groupby(['client_id', 'visit_id', 'step']).cumcount() + 1
    df_pt['time_diff'] = df_pt.groupby(['client_id', 'visit_id'])['date_time'].diff().fillna(pd.Timedelta(seconds=0))
    df_pt['time_diff'] = df_pt['time_diff'].dt.total_seconds()
    df_pt['total_time_in_step'] = df_pt.groupby(['client_id', 'visit_id', 'step_repeat_count'])['time_diff'].cumsum()

    max_repeats = df_pt.groupby(['client_id', 'visit_id'])['step_repeat_count'].max() <= 2
    no_retrocessions = ~(df_pt.groupby(['client_id', 'visit_id'])['step_diff'].apply(lambda x: (x == -1).any()))
    
    def follows_correct_sequence_with_repeats(group):
        steps = group['step_order'].tolist()
        correct_sequence = [1, 2, 3, 4, 5]
        valid_steps = []
        for step in correct_sequence:
            count = steps.count(step)
            if count == 0:
                return False
            valid_steps.extend([step] * min(count, 2))
        return steps == valid_steps

    correct_order = df_pt.groupby(['client_id', 'visit_id']).apply(follows_correct_sequence_with_repeats)
    is_lineal = max_repeats & no_retrocessions & correct_order
    df_pt = df_pt.merge(is_lineal.rename('lineal'), on=['client_id', 'visit_id'])

    return df_pt

# Función para construir el DataFrame final
def construir_df_vanguard(df_final, df_pt):
    # Fusionar df_pt con df_final basado en 'client_id'
    df_vanguard = pd.merge(df_pt, df_final.drop_duplicates(subset='client_id'), on='client_id', how='left')

    # Eliminar columnas duplicadas si existen
    df_vanguard = df_vanguard.drop(columns=['variation_y', 'lineal_y'], errors='ignore')
    df_vanguard = df_vanguard.rename(columns={'variation_x': 'variation', 'lineal_x': 'lineal'})

    return df_vanguard

def calculate_completion_rates(df):
    """
    Calcula la tasa de finalización (lineal=True) y de no finalización (lineal=False) para el dataframe dado.
    
    :param df: DataFrame con los datos de clientes (df_vanguard).
    :return: Tasa de finalización (lineal=True), Tasa de no finalización (lineal=False), Suma de tasas.
    """
    # Total de clientes únicos por variación
    total_clientes_por_variacion = df.drop_duplicates(subset=['client_id', 'variation']).groupby('variation')['client_id'].nunique()

    # Asegurarse de que no hay solapamiento entre clientes "lineal=True" y "lineal=False"
    lineal_true = df[(df['lineal'] == True)].drop_duplicates(subset=['client_id', 'variation'])
    lineal_false = df[(df['lineal'] == False) & (~df['client_id'].isin(lineal_true['client_id']))].drop_duplicates(subset=['client_id', 'variation'])

    # Calculamos la tasa de finalización para los que completaron el proceso de manera lineal
    completion_rate_lineal_true = lineal_true.groupby('variation')['client_id'].nunique() / total_clientes_por_variacion

    # Calculamos la tasa de "no finalización" para los que no completaron el proceso de manera lineal
    completion_rate_lineal_false = lineal_false.groupby('variation')['client_id'].nunique() / total_clientes_por_variacion

    # Suma de tasas
    suma_tasas = completion_rate_lineal_true + completion_rate_lineal_false

    return completion_rate_lineal_true, completion_rate_lineal_false, suma_tasas

def plot_completion_rates(completion_rate_true, completion_rate_false):
    """
    Genera un gráfico de barras comparando las tasas de finalización (lineal=True) y no finalización (lineal=False)
    para los grupos Test y Control.
    
    :param completion_rate_true: Series con la tasa de finalización (lineal=True) por grupo.
    :param completion_rate_false: Series con la tasa de no finalización (lineal=False) por grupo.
    """
    # Crear un DataFrame para las tasas
    completion_rates = pd.DataFrame({
        'Lineal=True': completion_rate_true,
        'Lineal=False': completion_rate_false
    })

    # Generar el gráfico de barras
    completion_rates.plot(kind='bar', figsize=(10, 6), color=['#4CAF50', '#FF5733'])

    # Añadir etiquetas y título
    plt.title('Tasa de Finalización y No Finalización por Grupo (Test vs Control)')
    plt.xlabel('Grupo')
    plt.ylabel('Proporción de Clientes')
    plt.xticks(rotation=0)
    plt.legend(title='Condición')
    plt.show()

def plot_avg_time_in_steps(df, step_order):
    """
    Genera un gráfico de líneas que compara el tiempo promedio en cada paso por variación (Test/Control) y 
    si el cliente completó el proceso de manera lineal o no.
    
    :param df: DataFrame con los datos de tiempos por paso (df_vanguard).
    :param step_order: Lista con el orden correcto de los pasos.
    :return: DataFrame con los tiempos promedio calculados.
    """
    # Calcular el tiempo total promedio en cada paso por tipo de web (Test/Control) y por si el cliente fue lineal o no
    avg_time_in_step_lineal = df.groupby(['variation', 'lineal', 'step'])['total_time_in_step'].mean().unstack()

    # Reordenar las columnas para reflejar el orden correcto de los pasos
    avg_time_in_step_lineal = avg_time_in_step_lineal[step_order]

    # Graficar comparando Test/Control y si fueron lineales o no
    plt.figure(figsize=(12, 6))

    # Gráfico para los que completaron de manera lineal (lineal=True)
    avg_time_in_step_lineal.loc[(slice(None), True), :].T.plot(kind='line', marker='o', figsize=(12, 6), ax=plt.gca(), label="Lineal")

    # Gráfico para los que no completaron de manera lineal (lineal=False)
    avg_time_in_step_lineal.loc[(slice(None), False), :].T.plot(kind='line', marker='x', linestyle='--', figsize=(12, 6), ax=plt.gca(), label="No Lineal")

    # Añadir etiquetas y título
    plt.title('Tiempo Promedio en Cada Paso: Lineal vs No Lineal (Test vs Control)')
    plt.xlabel('Paso')
    plt.ylabel('Tiempo Promedio (segundos)')
    plt.grid(True)
    plt.legend(title='Condición (Lineal/No Lineal)', loc='upper left')
    plt.show()

    return avg_time_in_step_lineal

def plot_error_rate(df, step_order):
    """
    Genera un gráfico de líneas que compara la tasa de errores por paso, comparando entre Test/Control y 
    si el cliente fue lineal o no.
    
    :param df: DataFrame con los datos de errores por paso (df_vanguard).
    :param step_order: Lista con el orden correcto de los pasos.
    :return: DataFrame con las tasas de error calculadas.
    """
    # Definir una nueva columna para marcar los errores (retrocesos o repeticiones mayores a 2)
    df['error'] = (df['step_diff'] == -1) | (df['step_repeat_count'] > 2)

    # Calcular la tasa de errores por cada grupo (Test/Control), si es lineal/no lineal, y por paso
    error_rate = df.groupby(['variation', 'lineal', 'step'])['error'].mean().unstack()

    # Reordenar las columnas para reflejar el orden correcto de los pasos
    error_rate = error_rate[step_order]

    # Graficar la tasa de errores comparando Test/Control y si fueron lineales o no
    plt.figure(figsize=(12, 6))

    # Gráfico para los que completaron de manera lineal (lineal=True)
    error_rate.loc[(slice(None), True), :].T.plot(kind='line', marker='o', figsize=(12, 6), ax=plt.gca(), label="Lineal")

    # Gráfico para los que no completaron de manera lineal (lineal=False)
    error_rate.loc[(slice(None), False), :].T.plot(kind='line', marker='x', linestyle='--', figsize=(12, 6), ax=plt.gca(), label="No Lineal")

    # Añadir etiquetas y título
    plt.title('Tasa de Errores en Cada Paso: Lineal vs No Lineal (Test vs Control)')
    plt.xlabel('Paso')
    plt.ylabel('Tasa de Errores')
    plt.grid(True)
    plt.legend(title='Condición (Lineal/No Lineal)', loc='upper left')
    plt.show()

    return error_rate

def calculate_and_plot_time_stats(df, step_order):
    """
    Calcula la media, skewness y kurtosis de los tiempos por paso, 
    y genera gráficos de barras con los resultados.
    
    :param df: DataFrame con los datos de tiempos por paso (df_vanguard).
    :param step_order: Lista con el orden correcto de los pasos.
    :return: DataFrame con las estadísticas calculadas.
    """
    # Agrupar por 'variation', 'step' y 'lineal' para calcular media, skewness y kurtosis
    grouped_time_stats = df.groupby(['variation', 'step', 'lineal'])['total_time_in_step'].agg(
        mean='mean',
        skew=lambda x: skew(x),
        kurtosis=lambda x: kurtosis(x)
    )

    # Reindexar el nivel de 'step' en el orden deseado
    grouped_time_stats = grouped_time_stats.reindex(step_order, level='step')

    # Graficar la media (mean)
    fig, ax = plt.subplots(figsize=(10, 6))
    grouped_time_stats['mean'].unstack().plot(kind='bar', ax=ax)
    ax.set_title('Mean Time per Step: Lineal vs Non-lineal (Test vs Control)')
    ax.set_ylabel('Mean Time (seconds)')
    ax.set_xlabel('Steps')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

    # Graficar el skewness
    fig, ax = plt.subplots(figsize=(10, 6))
    grouped_time_stats['skew'].unstack().plot(kind='bar', ax=ax)
    ax.set_title('Skewness per Step: Lineal vs Non-lineal (Test vs Control)')
    ax.set_ylabel('Skewness')
    ax.set_xlabel('Steps')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

    # Graficar la kurtosis
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
    """
    Calcula el tiempo promedio en cada paso por variación (Test/Control) y genera un gráfico de líneas.
    
    :param df: DataFrame con los datos.
    :param step_order: Lista con el orden correcto de los pasos.
    :return: DataFrame con los tiempos promedio por paso.
    """
    # Calcular el tiempo promedio en cada paso por variación
    df_time_per_step = df.groupby(['variation', 'step'])['total_time_in_step'].mean().unstack()

    # Reordenar los pasos
    df_time_per_step = df_time_per_step[step_order]

    # Graficar los tiempos promedio por paso
    df_time_per_step.T.plot(kind='line', marker='o', figsize=(10, 6))
    plt.title('Tiempo Promedio por Paso: Test vs Control')
    plt.xlabel('Paso')
    plt.ylabel('Tiempo Promedio')
    plt.grid(True)
    plt.show()

    # Retornar el DataFrame con los tiempos promedio
    return df_time_per_step

def calculate_pearson_spearman_corr(df):
    """
    Calcula las correlaciones de Pearson y Spearman y genera los heatmaps correspondientes.
    
    :param df: DataFrame con las columnas 'balance', 'activity', 'age', 'num_accounts'.
    :return: Dos DataFrames con las correlaciones de Pearson y Spearman.
    """
    # Pearson Correlation
    pearson_corr = df[['balance', 'activity', 'age', 'num_accounts']].corr(method='pearson')

    # Spearman Correlation
    spearman_corr = df[['balance', 'activity', 'age', 'num_accounts']].corr(method='spearman')

    # Visualizar con heatmaps para Pearson
    plt.figure(figsize=(10, 6))
    sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Pearson Correlation Heatmap')
    plt.show()

    # Visualizar con heatmaps para Spearman
    plt.figure(figsize=(10, 6))
    sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Spearman Correlation Heatmap')
    plt.show()

    # Retornar los DataFrames de las correlaciones
    return pearson_corr, spearman_corr

def z_test_completion_rates(df):
    """
    Realiza una prueba Z de proporciones para comparar las tasas de finalización entre los grupos Test y Control.
    
    :param df: DataFrame con los datos, incluyendo 'variation', 'client_id', y 'lineal'.
    :return: El estadístico Z y el valor p.
    """
    # Tasas de finalización en Test y Control
    n_test = df[df['variation'] == 'Test']['client_id'].nunique()  # Número de usuarios en Test
    n_control = df[df['variation'] == 'Control']['client_id'].nunique()  # Número de usuarios en Control

    completed_test = df[(df['variation'] == 'Test') & (df['lineal'] == True)]['client_id'].nunique()  # Usuarios que completaron el proceso en Test
    completed_control = df[(df['variation'] == 'Control') & (df['lineal'] == True)]['client_id'].nunique()  # Usuarios que completaron el proceso en Control

    # Proporciones
    p_test = completed_test / n_test
    p_control = completed_control / n_control

    # Prueba Z
    count = [completed_test, completed_control]
    nobs = [n_test, n_control]

    stat, p_value = proportions_ztest(count, nobs)

    return stat, p_value

def t_test_time_per_step(df, step):
    """
    Realiza un T-test para comparar los tiempos de finalización entre Test y Control,
    separados para usuarios lineales y no lineales en el paso especificado.
    
    :param df: DataFrame con los datos.
    :param step: El paso ('step') en el que realizar el análisis (ej: 'confirm').
    :return: Estadísticos t y valores p para usuarios lineales y no lineales.
    """
    # Filtrar los datos para usuarios lineales y no lineales en Test y Control
    time_test_lineal = df[(df['variation'] == 'Test') & (df['lineal'] == True) & (df['step'] == step)]['total_time_in_step']
    time_control_lineal = df[(df['variation'] == 'Control') & (df['lineal'] == True) & (df['step'] == step)]['total_time_in_step']

    time_test_non_lineal = df[(df['variation'] == 'Test') & (df['lineal'] == False) & (df['step'] == step)]['total_time_in_step']
    time_control_non_lineal = df[(df['variation'] == 'Control') & (df['lineal'] == False) & (df['step'] == step)]['total_time_in_step']

    # Realizar el T-test para los usuarios lineales
    t_stat_lineal, p_value_lineal = stats.ttest_ind(time_test_lineal, time_control_lineal, equal_var=False)

    # Realizar el T-test para los usuarios no lineales
    t_stat_non_lineal, p_value_non_lineal = stats.ttest_ind(time_test_non_lineal, time_control_non_lineal, equal_var=False)

    return (t_stat_lineal, p_value_lineal), (t_stat_non_lineal, p_value_non_lineal)

def mann_whitney_test_variation(df):
    """
    Realiza la prueba Mann-Whitney U para comparar los tiempos de finalización entre los grupos Test y Control.
    
    :param df: DataFrame con los datos.
    :return: Estadístico U y valor p.
    """
    # Filtrar los datos por grupos (Test y Control)
    test_group = df[df['variation'] == 'Test']['total_time_in_step']
    control_group = df[df['variation'] == 'Control']['total_time_in_step']

    # Aplicar la prueba Mann-Whitney U
    u_statistic, p_value = stats.mannwhitneyu(test_group, control_group, alternative='two-sided')

    return u_statistic, p_value

def mann_whitney_test_lineal(df):
    """
    Realiza la prueba Mann-Whitney U para comparar los tiempos de finalización entre usuarios lineales y no lineales.
    
    :param df: DataFrame con los datos.
    :return: Estadístico U y valor p.
    """
    # Filtrar los datos por grupos (Lineal y No lineal)
    lineal_group = df[df['lineal'] == True]['total_time_in_step']
    non_lineal_group = df[df['lineal'] == False]['total_time_in_step']

    # Aplicar la prueba Mann-Whitney U
    u_statistic, p_value = stats.mannwhitneyu(lineal_group, non_lineal_group, alternative='two-sided')

    return u_statistic, p_value

def chi_square_and_cramers_v(df):
    """
    Realiza la prueba Chi-square y calcula Cramér's V para evaluar la asociación entre 'variation' y 'lineal'.
    
    :param df: DataFrame con los datos.
    :return: Chi-square statistic, p-value, y Cramér's V.
    """
    # Crear la tabla de contingencia entre 'variation' y 'lineal'
    contingency_table = pd.crosstab(df['variation'], df['lineal'])

    # Aplicar la prueba Chi-square
    chi2, p, dof, ex = chi2_contingency(contingency_table)

    # Calcular Cramér's V
    n = contingency_table.sum().sum()  # Número total de observaciones
    cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))

    return chi2, p, cramers_v


def main():
    """Rebuild the core analysis dataset in memory from the raw files."""
    df_final_demo, df_final_experiment_clients, df_pt = cargar_datos()
    df_final_demo, df_final_experiment_clients, df_pt = limpiar_datos(
        df_final_demo, df_final_experiment_clients, df_pt
    )
    df_final_demo, df_final_experiment_clients, df_pt = filtrar_por_clientes_comunes(
        df_final_demo, df_final_experiment_clients, df_pt
    )
    df_final_demo, df_pt = convertir_tipos_datos(df_final_demo, df_pt)
    df_final, df_pt = renombrar_columnas_y_fusionar(
        df_final_experiment_clients, df_final_demo, df_pt
    )
    metrics = calcular_medias_percentiles(df_final)
    df_final = clasificar_clientes(df_final, metrics[:4], metrics[4:])
    df_pt = analizar_pasos(df_pt)
    df_vanguard = construir_df_vanguard(df_final, df_pt)

    print(
        f"Rebuilt core dataset in memory: {len(df_vanguard):,} rows, "
        f"{len(df_vanguard.columns)} columns."
    )
    print("No files were written.")


if __name__ == '__main__':
    main()