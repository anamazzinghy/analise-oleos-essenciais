import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import logging

# Configuração de logging para depuração
logging.basicConfig(level=logging.DEBUG)

# Função para carregar espectros GC-MS
def carregar_espectros(caminho):
    """
    Carrega os dados espectrais de óleos essenciais de um arquivo CSV.

    Parâmetros:
    caminho (str): Caminho para o arquivo CSV.

    Retorno:
    pd.DataFrame: Dados espectrais carregados.
    """
    try:
        return pd.read_csv(caminho)
    except Exception as e:
        st.error(f"Erro ao carregar os dados do arquivo {caminho}: {e}")
        logging.error(f"Erro ao carregar arquivo {caminho}: {e}")
        return None

# Função para pré-processar os dados
def preprocessar_espectros(dados):
    """
    Normaliza os dados espectrais para análise comparativa.

    Parâmetros:
    dados (pd.DataFrame): Dados espectrais originais.

    Retorno:
    pd.DataFrame: Dados normalizados.
    """
    try:
        # Normalização por soma total (percentual de cada composto)
        dados_normalizados = dados.div(dados.sum(axis=1), axis=0)
        return dados_normalizados
    except Exception as e:
        st.error(f"Erro ao processar os dados espectrais: {e}")
        logging.error(f"Erro ao normalizar os dados: {e}")
        return None

# Função para calcular similaridade entre espectros
def calcular_similaridade(espectro_teste, banco_referencia):
    """
    Calcula a similaridade do espectro de teste com os espectros de referência.

    Parâmetros:
    espectro_teste (pd.Series): Espectro de teste.
    banco_referencia (pd.DataFrame): Banco de dados de espectros puros.

    Retorno:
    list: Lista de similaridades com cada espectro de referência.
    """
    similaridades = []
    try:
        for _, referencia in banco_referencia.iterrows():
            similaridade = cosine_similarity(
                [espectro_teste.values], [referencia.values]
            )[0][0]
            similaridades.append(similaridade)
    except Exception as e:
        st.error(f"Erro ao calcular similaridade: {e}")
        logging.error(f"Erro ao calcular similaridade: {e}")
    return similaridades

# Função para analisar PCA (para visualização gráfica)
def plot_pca(dados, labels):
    """
    Plota os dados espectrais em um gráfico PCA para visualização de padrões.

    Parâmetros:
    dados (pd.DataFrame): Dados normalizados para PCA.
    labels (list): Classificações associadas aos dados.
    """
    try:
        pca = PCA(n_components=2)
        componentes = pca.fit_transform(dados)
        plt.figure(figsize=(8, 6))
        for i, label in enumerate(set(labels)):
            indices = [j for j, l in enumerate(labels) if l == label]
            plt.scatter(
                componentes[indices, 0],
                componentes[indices, 1],
                label=label
            )
        plt.title("Análise PCA dos Espectros")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Erro ao gerar o gráfico PCA: {e}")
        logging.error(f"Erro ao plotar PCA: {e}")

# Função para detectar compostos adulterantes
def detectar_adulterantes(espectro_teste, banco_adulterantes):
    """
    Verifica se compostos adulterantes estão presentes no espectro.

    Parâmetros:
    espectro_teste (pd.Series): Espectro de teste.
    banco_adulterantes (pd.DataFrame): Banco de dados de adulterantes.

    Retorno:
    list: Lista de adulterantes detectados.
    """
    adulterantes_detectados = []
    try:
        for _, adulterante in banco_adulterantes.iterrows():
            if any(espectro_teste.values >= adulterante.values):
                adulterantes_detectados.append(adulterante.name)
    except Exception as e:
        st.error(f"Erro ao detectar adulterantes: {e}")
        logging.error(f"Erro ao detectar adulterantes: {e}")
    return adulterantes_detectados

# Função para determinar a pureza do óleo essencial
def determinar_pureza(similaridade, adulterantes_detectados):
    """
    Classifica o óleo essencial com base na pureza.

    Parâmetros:
    similaridade (float): Similaridade máxima com o banco de referência.
    adulterantes_detectados (list): Lista de adulterantes encontrados.

    Retorno:
    str: Resultado da classificação.
    """
    if similaridade >= 0.9 and not adulterantes_detectados:
        return "Alta potencial de pureza"
    else:
        return "Alerta: Produto possivelmente adulterado"

# Pipeline principal
def pipeline(caminho_teste, caminho_referencia, caminho_adulterantes):
    """
    Executa o pipeline completo de validação de pureza de óleos essenciais.

    Parâmetros:
    caminho_teste (str): Caminho para o arquivo CSV com dados da amostra.
    caminho_referencia (str): Caminho para o arquivo CSV do banco de referência.
    caminho_adulterantes (str): Caminho para o arquivo CSV do banco de adulterantes.
    """
    # Carregando os dados
    st.write("Carregando dados...")
    espectros_teste = carregar_espectros(caminho_teste)
    banco_referencia = carregar_espectros(caminho_referencia)
    banco_adulterantes = carregar_espectros(caminho_adulterantes)

    if espectros_teste is None or banco_referencia is None or banco_adulterantes is None:
        return

    # Pré-processando os dados
    st.write("Pré-processando dados...")
    espectros_teste = preprocessar_espectros(espectros_teste)
    banco_referencia = preprocessar_espectros(banco_referencia)
    banco_adulterantes = preprocessar_espectros(banco_adulterantes)

    # Calculando similaridades
    st.write("Calculando similaridades...")
    resultados = []
    classificacoes = []
    for _, espectro in espectros_teste.iterrows():
        similaridades = calcular_similaridade(espectro, banco_referencia)
        max_similaridade = max(similaridades)
        adulterantes = detectar_adulterantes(espectro, banco_adulterantes)
        status = determinar_pureza(max_similaridade, adulterantes)
        resultados.append({
            "similaridade": max_similaridade,
            "status": status,
            "adulterantes": adulterantes
        })
        classificacoes.append(status)

    # Exibindo resultados
    st.write("Visualizando com PCA...")
    plot_pca(pd.concat([banco_referencia, espectros_teste]), classificacoes)

    st.write("Resultados Finais:")
    for i, res in enumerate(resultados):
        st.write(f"Amostra {i + 1}: Similaridade={res['similaridade']:.2f}, Status={res['status']}")
        if res["adulterantes"]:
            st.write(f"  - Adulterantes detectados: {', '.join(res['adulterantes'])}")

# Exemplo de execução no Streamlit
st.title("Avaliação de Óleos Essenciais")
caminho_teste = st.text_input("Caminho para o arquivo CSV de amostras de teste:")
caminho_referencia = st.text_input("Caminho para o arquivo CSV do banco de referência:")
caminho_adulterantes = st.text_input("Caminho para o arquivo CSV do banco de adulterantes:")
if st.button("Executar Pipeline"):
    pipeline(caminho_teste, caminho_referencia, caminho_adulterantes)

caminho_teste = st.text_input("Caminho para o arquivo CSV de amostras de teste:")
caminho_referencia = st.text_input("Caminho para o arquivo CSV do banco de referência:")
caminho_adulterantes = st.text_input("Caminho para o arquivo CSV do banco de adulterantes:")
if st.button("Executar Pipeline"):
    pipeline(caminho_teste, caminho_referencia, caminho_adulterantes)
