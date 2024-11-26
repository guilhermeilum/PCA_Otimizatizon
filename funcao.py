import numpy as np

import pandas as pd
from matplotlib import pyplot as plt
from pca import pca
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error,
    silhouette_score,
    v_measure_score,
)  # Importação da função para cálculo do erro quadrático médio
from time import time
import os
from scatterd import scatterd
from scipy.stats import mode


def scatter_copia(
    xs=None,
    ys=None,
    zs=None,
    labels=None,
    c=[0, 0.1, 0.4],
    s=150,
    marker="o",
    edgecolor="#000000",
    jitter=None,
    alpha=0.8,
    gradient=None,
    density=False,
    density_on_top=False,
    fontcolor=[0, 0, 0],
    fontsize=18,
    fontweight="normal",
    cmap="tab20c",
    legend=None,
    figsize=(25, 15),
    dpi=100,
    visible=True,
    fig=None,
    ax=None,
    grid=True,
    verbose=3,
):
    """Scatter 2d plot.

    Parameters
    ----------
    labels : array-like, default: None
        Label for each sample. The labeling is used for coloring the samples.
    c: list/array of RGB colors for each sample.
        The marker colors. Possible values:
            * A scalar or sequence of n numbers to be mapped to colors using cmap and norm.
            * A 2D array in which the rows are RGB or RGBA.
            * A sequence of colors of length n.
            * A single color format string.
    s: Int or list/array (default: 50)
        Size(s) of the scatter-points.
        [20, 10, 50, ...]: In case of list: should be same size as the number of PCs -> .results['PC']
        50: all points get this size.
    marker: list/array of strings (default: 'o').
        Marker for the samples.
            * 'x' : All data points get this marker
            * ('.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X') : Specify per sample the marker type.
    jitter : float, default: None
        Add jitter to data points as random normal data. Values of 0.01 is usually good for one-hot data seperation.
    PC : tupel, default: None
        Plot the selected Principal Components. Note that counting starts from 0. PC1=0, PC2=1, PC3=2, etc.
        None : Take automatically the first 2 components and 3 in case d3=True.
        [0, 1] : Define the PC for 2D.
        [0, 1, 2] : Define the PCs for 3D
    SPE : Bool, default: False
        Show the outliers based on SPE/DmodX method.
            * None : Auto detect. If outliers are detected. it is set to True.
            * True : Show outliers
            * False : Do not show outliers
    HT2 : Bool, default: False
        Show the outliers based on the hotelling T2 test.
            * None : Auto detect. If outliers are detected. it is set to True.
            * True : Show outliers
            * False : Do not show outliers
    alpha: float or array-like of floats (default: 1).
        The alpha blending value ranges between 0 (transparent) and 1 (opaque).
        1: All data points get this alpha
        [1, 0.8, 0.2, ...]: Specify per sample the alpha
    gradient : String, (default: None)
        Hex color to make a lineair gradient for the scatterplot.
        '#FFFFFF'
    density : Bool (default: False)
        Include the kernel density in the scatter plot.
    density_on_top : bool, (default: False)
        True : The density is the highest layer.
        False : The density is the lowest layer.
    fontsize : String, optional
        The fontsize of the y labels that are plotted in the graph. The default is 16.
    fontcolor: list/array of RGB colors with same size as X (default : None)
        None : Use same colorscheme as for c
        '#000000' : If the input is a single color, all fonts will get this color.
    cmap : String, optional, default: 'Set1'
        Colormap. If set to None, no points are shown.
    title : str, default: None
        Title of the figure.
        None: Automatically create title text based on results.
        '' : Remove all title text.
        'title text' : Add custom title text.
    legend : int, default: None
        None: Set automatically based on number of labels.
        False : No legend.
        True : Best position.
        1 : 'upper right'
        2 : 'upper left'
        3 : 'lower left'
        4 : 'lower right'
    figsize : (int, int), optional, default: (25, 15)
        (width, height) in inches.
    visible : Bool, default: True
        Visible status of the Figure. When False, figure is created on the background.
    Verbose : int (default : 3)
        Print to screen. 0: None, 1: Error, 2: Warning, 3: Info, 4: Debug, 5: Trace

    Returns
    -------
    tuple containing (fig, ax)

    """

    # Set parameters based on intuition
    if c is None:
        s = 0
    if cmap is None:
        s = 0
    if alpha is None:
        alpha = 0.8

    fig, ax = scatterd(
        x=xs,
        y=ys,
        z=zs,
        s=s,
        c=c,
        labels=labels,
        edgecolor=edgecolor,
        alpha=alpha,
        marker=marker,
        jitter=jitter,
        density=density,
        opaque_type="per_class",
        density_on_top=density_on_top,
        gradient=gradient,
        cmap=cmap,
        legend=legend,
        fontcolor=fontcolor,
        fontsize=fontsize,
        fontweight=fontweight,
        grid=grid,
        dpi=dpi,
        figsize=figsize,
        visible=visible,
        fig=fig,
        ax=ax,
        verbose=verbose,
    )

    # Return
    return (fig, ax)


def seleciona_ciclo(dados, ciclo, intervalo_pontos=81):
    dados_novos = {}
    if isinstance(ciclo, list):
        for nome, dado in dados.items():
            ciclos = []
            for i in range(0, len(dado), intervalo_pontos):
                ciclos.append(
                    dado.iloc[0 + i : intervalo_pontos + i].reset_index(drop=True)
                )
            for ciclo_ in ciclo:
                dados_novos[nome + f"{ciclo_}"] = ciclos[ciclo_]
    else:
        for nome, dado in dados.items():
            ciclos = []
            for i in range(0, len(dado), intervalo_pontos):
                ciclos.append(
                    dado.iloc[0 + i : intervalo_pontos + i].reset_index(drop=True)
                )
            dados_novos[nome] = ciclos[ciclo]
    return dados_novos


def separa_dados(df_dict):
    dict_incremento = {}
    dict_decremento = {}
    for nome, df in df_dict.items():
        incremento = df.iloc[: len(df) // 2 + 1]
        decremento = df.iloc[len(df) // 2 :]
        dict_incremento[nome] = incremento
        dict_decremento[nome] = decremento
    return dict_incremento, dict_decremento


def corta_dados(df_dict, intervalo_list=None, especifico=None):
    dados_cortados = {}
    if intervalo_list is not None and especifico is not None:
        raise AssertionError(
            "'intervalo' e 'especifico' foram fornecidos ao mesmo tempo."
        )
    elif intervalo_list is not None:
        for nome, df in df_dict.items():
            df_list = []
            for intervalo in intervalo_list:
                cond1 = df["V"] >= intervalo[0]
                cond2 = df["V"] <= intervalo[1]
                df_list.append(df[cond1 & cond2])
            df = pd.concat(df_list)
            df = df.drop_duplicates()
            dados_cortados[nome] = df.sort_values("V", ignore_index=True)
    elif especifico is not None:
        for nome, df in df_dict.items():
            dados_cortados[nome] = df[df["V"].isin(especifico)]
    else:
        raise ValueError("Falta o intervalo, ou os valores especificos.")

    return dados_cortados


def cria_dados(dados, tipo="amostra"):
    ids_lista = []
    if tipo == "feature":
        primeiro_ciclo = int(list(dados.keys())[0][-1])
        ciclos = []
        for nome, dado in dados.items():
            ids = dado[["IDS_Oupt01__--0.2000"]].T
            v = dado["V"].values
            ciclo = int(nome[-1])
            if ciclo == primeiro_ciclo and len(ciclos) != 0:
                dado_concat = pd.concat(ciclos, axis=1)
                dado_concat = dado_concat.loc[:, ~dado_concat.columns.duplicated()]
                ids_lista.append(dado_concat)
                ciclos = []

            if "parte" in dado:
                v = [str(v_) + "_" + tipo for v_, tipo in zip(v, dado["parte"])]
            v = [f"{v_}_{ciclo}" for v_ in v]

            ids.columns = v
            ids.index = [nome]

            eletrodo = nome.split("\\")[2]

            amostra = nome.split("\\")[-1].split("--")[0].split("_")
            label = amostra[0]
            ids["eletrodo"] = eletrodo
            ids["label"] = label
            ciclos.append(ids.reset_index(drop=True))
        dados_final = pd.concat(ids_lista)
        dados_final = dados_final.loc[:, ~dados_final.columns.duplicated()]

    elif tipo == "amostra":
        tem_ciclo = False
        for nome, dado in dados.items():
            ids = dado[["IDS_Oupt01__--0.2000"]].T
            v = dado["V"].values

            if "parte" in dado:
                v = [str(v_) + " " + f"({tipo})" for v_, tipo in zip(v, dado["parte"])]

            ids.columns = v
            ids.index = [nome]

            eletrodo = nome.split("\\")[2]

            amostra = nome.split("\\")[-1].split("--")[0].split("_")
            label = amostra[0]
            ids["eletrodo"] = eletrodo
            ids["label"] = label
            if nome[-1].isdigit():
                ids["ciclo"] = nome[-1]
                tem_ciclo = True
            ids_lista.append(ids)

        dados_final = pd.concat(ids_lista)
    dados_final = dados_final.reset_index(drop=True)
    dados_final["replicata"] = dados_final.groupby(["eletrodo", "label"]).cumcount()

    # Filtrando os dados para cada eletrodo
    quimico = dados_final[dados_final["eletrodo"] == "Q1"].copy()
    eletrico = dados_final[dados_final["eletrodo"] == "E2"].copy()
    termico = dados_final[dados_final["eletrodo"] == "T3"].copy()

    if tem_ciclo:
        # Primeiro merge entre 'quimico' e 'eletrico' com os sufixos (Q) e (E) usando 'inner'
        df_merged = quimico.merge(
            eletrico,
            on=["label", "replicata", "ciclo"],
            how="inner",
            suffixes=[" (Q)", " (E)"],
        )
        df_merged.rename(
            columns={
                "label (Q)": "label",
                "replicata (Q)": "replicata",
                "ciclo (Q)": "ciclo",
            }
        )
        termico = termico.add_suffix(" (T)")
        termico = termico.rename(
            columns={
                "label (T)": "label",
                "replicata (T)": "replicata",
                "ciclo (T)": "ciclo",
            }
        )
        # Segundo merge com 'termico', adicionando o sufixo  (T), também com 'inner'
        df_merged = df_merged.merge(
            termico, on=["label", "replicata", "ciclo"], how="inner", suffixes=[" (T)"]
        )
    else:
        # Primeiro merge entre 'quimico' e 'eletrico' com os sufixos  (Q) e  (E) usando 'inner'
        df_merged = quimico.merge(
            eletrico,
            on=["label", "replicata"],
            how="inner",
            suffixes=[" (Q)", " (E)"],
        )
        df_merged.rename(columns={"label (Q)": "label", "replicata (Q)": "replicata"})
        termico = termico.add_suffix(" (T)")
        termico = termico.rename(
            columns={"label (T)": "label", "replicata (T)": "replicata"}
        )
        # Segundo merge com 'termico', adicionando o sufixo  (T), também com 'inner'
        df_merged = df_merged.merge(
            termico, on=["label", "replicata"], how="inner", suffixes=[" (T)"]
        )

    # Removendo as colunas auxiliares de merge, se não forem mais necessárias
    df_final = df_merged.select_dtypes(include=["number"])

    # Extraindo os labels
    labels = df_merged["label"]

    return df_final, labels


class PCAKMeansOptimizer:
    """
    Classe para realizar a otimização de parâmetros do PCA e KMeans usando Optuna.

    Esta classe organiza os parâmetros necessários, facilita o reuso de dados e métodos,
    e encapsula o fluxo de trabalho de processamento, clustering e otimização.

    Parâmetros:
    ----------
    Dados : qualquer tipo
        Dados utilizados no processamento.
    """

    def __init__(self, dados):
        self.dados = dados
        self.tempo_pca = []
        self.tempo_kmeans = []
        volts = (
            cria_dados(self.dados)[0].dropna(axis=1).drop("replicata", axis=1).columns
        )
        volts = set(float(v.split(" ")[0]) for v in volts)
        self.volts = sorted(list(volts))

    def corta_dados(self, intervalo_list):
        """
        Corta os dados com base nos intervalos fornecidos.
        """
        return corta_dados(self.dados, intervalo_list)

    def modelo_pca(self, intervalo_index):
        """
        Aplica PCA aos dados normalizados e realiza clustering com KMeans.

        Parâmetros:
        ----------
        intervalo_index : list
            Lista de index de intervalos a serem considerados nos dados.

        Retorna:
        -------
        labels : pd.Series
            Rótulos reais dos dados.
        kmeans : KMeans
            Modelo KMeans treinado.
        modelo_data : dict
            Dados transformados pelo PCA para o conjunto de treino.
        df : pd.DataFrame
            Dados originais organizados e processados.
        model : pca
            Objeto PCA ajustado aos dados de treino.
        pca_data_treino : np.ndarray
            Dados transformados pelo PCA para o conjunto de treino.
        labels_treino : pd.Series
            Rótulos do conjunto de treino.
        pca_data_val : np.ndarray
            Dados transformados pelo PCA para o conjunto de validação.
        labels_val : pd.Series
            Rótulos do conjunto de validação.
        """

        if intervalo_index is not None:
            # Filtra os dados com base nos intervalos fornecidos
            intervalos = [[self.volts[i], self.volts[j]] for i, j in intervalo_index]
            dados_cortados = self.corta_dados(intervalos)
        else:
            dados_cortados = self.dados
        df, labels = cria_dados(dados_cortados)
        df = df.dropna(axis=1)
        # Verifica o número de replicatas únicas
        replicatas_unicas = df["replicata"].unique()
        num_replicatas = len(replicatas_unicas)

        # Checa se o número de replicatas é múltiplo de 3
        if num_replicatas % 3 != 0:
            raise ValueError(
                f"O número de replicatas ({num_replicatas}) não é múltiplo de 3. "
                "A distribuição proporcional não pode ser garantida."
            )

        # Fixar replicatas para o conjunto de teste
        replicatas_teste_fixas = replicatas_unicas[
            : num_replicatas // 3
        ]  # Seleciona um bloco fixo para o teste

        # Define replicatas restantes para treino e validação
        replicatas_restantes = np.setdiff1d(replicatas_unicas, replicatas_teste_fixas)

        # Divide as replicatas restantes entre treino e validação
        treino_idx = replicatas_restantes[: len(replicatas_restantes) // 2]
        val_idx = replicatas_restantes[len(replicatas_restantes) // 2 :]

        # Filtra os dados e separa as labels para treino, validação e teste
        treino = df[df["replicata"].isin(treino_idx)].drop("replicata", axis=1)
        labels_treino = labels[df["replicata"].isin(treino_idx)]

        val = df[df["replicata"].isin(val_idx)].drop("replicata", axis=1).values
        labels_val = labels[df["replicata"].isin(val_idx)]

        teste = (
            df[df["replicata"].isin(replicatas_teste_fixas)]
            .drop("replicata", axis=1)
            .values
        )
        labels_teste = labels[df["replicata"].isin(replicatas_teste_fixas)]

        start_pca = time()
        # Aplica PCA aos dados de treino
        model = pca(3, verbose=0, normalize=True, random_state=42)
        modelo_data = model.fit_transform(
            treino, col_labels=treino.columns, row_labels=labels_treino
        )
        pca_data_treino = modelo_data["PC"].copy()

        # Transforma os dados de validação e teste com o modelo PCA ajustado
        pca_data_val = model.transform(
            val, col_labels=treino.columns, row_labels=labels_val
        )
        pca_data_teste = model.transform(
            teste, col_labels=treino.columns, row_labels=labels_teste
        )
        end_pca = time()
        self.tempo_pca.append(end_pca - start_pca)
        # Ajusta o modelo KMeans nos componentes principais do treino
        start_Kmeans = time()
        kmeans = KMeans(
            n_clusters=len(labels.unique()),
            random_state=42,
        )
        kmeans.fit(modelo_data["PC"])
        end_Kmeans = time()
        self.tempo_kmeans.append(end_Kmeans - start_Kmeans)

        return (
            labels,
            kmeans,
            modelo_data,
            df,
            model,
            pca_data_treino,
            labels_treino,
            pca_data_val,
            labels_val,
            pca_data_teste,
            labels_teste,
        )

    def objective(self, trial):
        """
        Função de objetivo para otimização de hiperparâmetros usando Optuna.

        Realiza a busca pelos melhores hiperparâmetros do PCA e KMeans com base em métricas de agrupamento.

        Parâmetros:
        ----------
        trial : optuna.Trial
            Objeto que define o conjunto de hiperparâmetros em um trial.

        Retorna:
        -------
        metrica : float
            Métrica de avaliação do agrupamento (v_measure_score).
        n_carregamentos : int
            Número de carregamentos gerados pelo PCA.
        """

        # Gera os intervalos dinamicamente com base nos parâmetros do trial
        intervalos = []
        intervalos_n = trial.suggest_int("intervalos", 1, 10)
        for n in range(intervalos_n):
            min_index = trial.suggest_int(f"min_index_{n}", 0, len(self.volts) - 2)

            max_index = trial.suggest_int(
                f"max_index_{n}", min_index, len(self.volts) - 1
            )

            intervalos.append([min_index, max_index])

        # Chama a função de PCA e clustering
        (
            labels,
            kmeans,
            modelo_data,
            _,
            model,
            pca_data_treino,
            labels_treino,
            pca_data_val,
            labels_val,
            pca_data_teste,
            labels_teste,
        ) = self.modelo_pca(intervalos)

        # Predição nos dados de validação
        clusters = kmeans.predict(np.concatenate([pca_data_treino, pca_data_val]))
        metrica = v_measure_score(np.concatenate([labels_treino, labels_val]), clusters)
        n_voltagens = len(model.results["loadings"].columns)
        silhouette = silhouette_score(
            np.concatenate([pca_data_treino, pca_data_val]), clusters, random_state=42
        )
        silhouette_reais = silhouette_score(
            np.concatenate([pca_data_treino, pca_data_val]),
            np.concatenate([labels_treino, labels_val]),
            random_state=42,
        )

        # Retorna a métrica e o número de dimensões principais (carregamentos)
        return metrica, n_voltagens, silhouette, silhouette_reais


def arrumar_fig(ax, x_label=None, y_label=None, z_label=None, loc=None, font_size=None):
    """
    Ajusta propriedades visuais de um gráfico matplotlib, incluindo rótulos dos eixos,
    tamanhos de fonte e localização da legenda.

    Args:
        ax (matplotlib.axes.Axes): Objeto Axes do matplotlib para o qual as propriedades
            serão ajustadas.
        x_label (str, optional): Rótulo do eixo X. Se não especificado, usa o rótulo
            atual do eixo X.
        y_label (str, optional): Rótulo do eixo Y. Se não especificado, usa o rótulo
            atual do eixo Y.
        z_label (str, optional): Rótulo do eixo Z (apenas para gráficos 3D).
            Se não especificado, usa o rótulo atual do eixo Z.
        loc (int, optional): Localização da legenda. Deve ser um dos seguintes valores:
            - 1: "upper left" (superior esquerdo)
            - 2: "upper right" (superior direito)
            - 3: "lower left" (inferior esquerdo)
            - 4: "lower right" (inferior direito)
            Se não especificado, mantém a localização padrão da legenda.

    Raises:
        None

    Funcionalidade:
        - Define os tamanhos das fontes para os rótulos dos eixos (X, Y, e Z) e para os ticks.
        - Ajusta o tamanho da fonte e a localização da legenda, se existir.
        - Para gráficos 3D, ajusta o rótulo e os ticks do eixo Z.
        - Tenta ajustar a proporção da borda da caixa (box aspect) em gráficos 3D.

    Nota:
        - A funcionalidade para o eixo Z é ignorada se o objeto `ax` não suportar gráficos 3D.
        - Se `loc` for fornecido, o mapeamento numérico para localização da legenda é:
            1 -> "upper left", 2 -> "upper right", 3 -> "lower left", 4 -> "lower right".

    Example:
        ```python
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        arrumar_fig(ax, x_label="Eixo X", y_label="Eixo Y", loc=1)
        ```
    """

    if x_label is None:
        x_label = ax.get_xlabel()
    if y_label is None:
        y_label = ax.get_ylabel()
    # Ajuste para os rótulos dos eixos
    ax.set_xlabel(
        x_label, fontsize=font_size
    )  # Define o tamanho da fonte para o eixo x
    ax.set_ylabel(
        y_label, fontsize=font_size
    )  # Define o tamanho da fonte para o eixo y
    # Ajuste para os ticks dos eixos
    ax.tick_params(
        axis="x", labelsize=font_size
    )  # Define o tamanho da fonte para os ticks do eixo x
    ax.tick_params(
        axis="y", labelsize=font_size
    )  # Define o tamanho da fonte para os ticks do eixo y

    # Ajuste para a legenda
    if ax.get_legend() is not None:
        loc_dict = {
            1: "upper left",
            2: "upper right",
            3: "lower left",
            4: "lower right",
        }
        if loc is None:
            ax.legend(fontsize=font_size)  # Define o tamanho da fonte para a legenda
        else:
            ax.legend(fontsize=font_size, loc=loc_dict[loc])
    # ajuste 3d
    try:
        # ajuste borda
        ax.set_box_aspect(None, zoom=0.85)

        if z_label is None:
            z_label = ax.get_zlabel()
        ax.set_zlabel(
            z_label, fontsize=font_size
        )  # Define o tamanho da fonte para o eixo z
        ax.tick_params(
            axis="z", labelsize=font_size
        )  # Define o tamanho da fonte para os ticks do eixo y
    except:
        pass


def plot_comparacao(
    treino_real,
    treino_predito,
    teste_real,
    teste_predito,
    unidade="mmol",
    label_real="Valores Reais",
    label_predito="Valores Previstos",
    path=None,
    path_csv=None,
    suffix=None,
    font_size=15,
):
    """
    Plota a comparação entre os valores reais e previstos para os dados de treino e teste,
    calcula o RMSE e salva os valores reais e previstos em um arquivo CSV.

    Parâmetros:
    - treino_real: array ou Series com os valores reais do treino.
    - treino_predito: array ou Series com os valores ajustados do treino.
    - teste_real: array ou Series com os valores reais do teste.
    - teste_predito: array ou Series com os valores previstos do teste.
    - label_real: (opcional) rótulo do eixo X, padrão: 'Valores Reais'.
    - label_predito: (opcional) rótulo do eixo Y, padrão: 'Valores Previstos'.
    - path: (opcional) caminho para salvar o gráfico.
    - arquivo_csv: (opcional) nome do arquivo CSV para salvar os valores reais e previstos.
    - font_size: (opcional) Tamanho da fonte dos textos.

    Retorno:
    - rmse_treino: RMSE do conjunto de treino.
    - rmse_teste: RMSE do conjunto de teste.
    """
    # Cálculo do RMSE
    rmse_treino = np.sqrt(mean_squared_error(treino_real, treino_predito))
    rmse_teste = np.sqrt(mean_squared_error(teste_real, teste_predito))
    print(f"Erro Quadrático Médio Raiz (RMSE) para o conjunto de treino: {rmse_treino}")
    print(f"Erro Quadrático Médio Raiz (RMSE) para o conjunto de teste: {rmse_teste}")

    # Criação da figura e dos eixos
    fig, ax = plt.subplots(figsize=(8, 8))

    # Dados de treino: valores reais e ajustados pelo modelo
    ax.scatter(
        treino_real,  # Valores reais do treino
        treino_predito,  # Valores ajustados pelo modelo
        color="blue",  # Cor azul para o treino
        label="Treino",  # Legenda
        alpha=0.7,  # Transparência
        edgecolor="k",  # Borda preta nos pontos
    )

    # Dados de teste: valores reais e previstos
    ax.scatter(
        teste_real,  # Valores reais do teste
        teste_predito,  # Valores previstos pelo modelo
        color="red",  # Cor vermelha para o teste
        label="Teste",  # Legenda
        alpha=0.7,  # Transparência
        edgecolor="k",  # Borda preta nos pontos
    )

    # Linha de identidade para referência
    limites = [
        min(min(treino_real), min(teste_real)),
        max(max(treino_real), max(teste_real)),
    ]
    ax.plot(
        limites, limites, "--", color="gray", label="Linha de Identidade"
    )  # Linha de identidade

    # Configurações do gráfico
    ax.set_xlabel(label_real + f"({unidade})", fontsize=font_size, labelpad=10)
    ax.set_ylabel(label_predito + f"({unidade})", fontsize=font_size, labelpad=10)
    ax.tick_params(axis="both", labelsize=font_size)
    ax.legend(fontsize=font_size, frameon=True, loc="best")  # Adiciona a legenda
    ax.grid(visible=True, linestyle="--", alpha=0.7)

    # Adicionar RMSE no gráfico
    texto_rmse = (
        f"RMSE (Treino): {rmse_treino:.2f} {unidade}\n"
        f"RMSE (Teste): {rmse_teste:.2f} {unidade}"
    )
    ax.text(
        0.05,
        0.95,
        texto_rmse,
        transform=ax.transAxes,
        fontsize=font_size,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Ajustar layout e exibir o gráfico
    fig.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=600)
    plt.close()

    # Salvar os valores reais e previstos em um DataFrame
    if path_csv is not None:
        df_treino = pd.DataFrame(
            {
                "Treino_Real": treino_real,
                "Treino_Predito": treino_predito,
            }
        )
        df_teste = pd.DataFrame(
            {
                "Teste_Real": teste_real,
                "Teste_Predito": teste_predito,
            }
        )
        if suffix is None:
            df_treino.to_csv(os.path.join(path_csv, "treino.csv"), index=False)
            df_teste.to_csv(os.path.join(path_csv, "teste.csv"), index=False)
        else:
            df_treino.to_csv(
                os.path.join(path_csv, f"treino_{suffix}.csv"), index=False
            )
            df_teste.to_csv(os.path.join(path_csv, f"teste_{suffix}.csv"), index=False)

    # Retornar os RMSEs
    return rmse_treino, rmse_teste


def corta_min(dict_):
    dict_neg = {}
    dict_pos = {}
    for nome, dado in dict_.items():
        min_index = dado["IDS_Oupt01__--0.2000"].idxmin()
        v_min = dado["V"].loc[min_index]
        dado_neg = dado[dado["V"] <= v_min]
        dado_pos = dado[dado["V"] >= v_min]
        dict_neg[nome] = dado_neg
        dict_pos[nome] = dado_pos
    return dict_neg, dict_pos
