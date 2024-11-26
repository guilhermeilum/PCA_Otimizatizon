# %%
# Importação das bibliotecas necessárias
import os
import warnings  # Para suprimir warnings desnecessários
from glob import glob  # Para localizar arquivos em um diretório

# Importação de bibliotecas de machine learning, visualização e análise estatística
import joblib  # Para salvar e carregar modelos de otimização e aprendizado
import numpy as np  # Operações numéricas com arrays
import optuna  # Framework para otimização de hiperparâmetros
import pandas as pd  # Manipulação e análise de dados tabulares
import statsmodels.api as sm  # Modelos estatísticos (OLS e afins)
from formulaic import Formula  # Fórmulas para criar matrizes de predição
from matplotlib import pyplot as plt  # Para criação de gráficos
from optuna.samplers import TPESampler  # Sampler para otimização de hiperparâmetros
from pysr import PySRRegressor

# Importação de ferramentas do sklearn para clustering, redução de dimensionalidade e escalonamento
from sklearn.metrics import v_measure_score  # Métrica de avaliação para clusters
from sympy import Eq, simplify, symbols, latex

# Importação de funções auxiliares específicas do projeto
from funcao import *

# Ignorar mensagens de warnings para melhorar a legibilidade do notebook
warnings.filterwarnings("ignore")
import scienceplots

plt.style.use(["science"])

import pathlib

import matplotlib

matplotlib.use("agg")
optuna.logging.set_verbosity(optuna.logging.WARNING)
# Se estiver usando intel isso irá deixar mais rapido, baixe o scikit-learn-intelex
from sklearnex import patch_sklearn

patch_sklearn()

# %% [markdown]
# # Carregamento de Arquivos
#
# Nesta etapa, definimos o caminho da pasta contendo os arquivos `.txt` e criamos uma lista com todos os arquivos presentes na pasta e suas subpastas. Essa lista será utilizada para processar os dados posteriormente.

# %%
# Caminho da pasta onde estão os arquivos .txt
pasta = r"medidas"

# Lista todos os arquivos .txt na pasta e subpastas
arquivos_txt = glob(rf"{pasta}\**\*.txt", recursive=True)

# %% [markdown]
# # Processamento dos Dados
#
# Aqui, os dados dos arquivos `.txt` são carregados em um dicionário, onde a chave é o nome do arquivo e o valor é o conteúdo processado como um DataFrame. Em seguida, selecionamos o segundo ciclo dos dados utilizando a função `seleciona_ciclo`. Finalmente, dividimos os dados em dois conjuntos: `incremento` e `decremento`, representando os diferentes estágios do ciclo.
#

# %%
# Carregamento dos dados dos arquivos e remoção de colunas com valores ausentes
dados = {
    nome.replace("mmol", " mmol"): pd.read_table(nome, sep="\t").dropna(axis=1)
    for nome in arquivos_txt
}
# Seleciona os dados do segundo ciclo
dados_ciclo = seleciona_ciclo(dados, [1, 2])
# Separa os dados do ciclo em incremento e decremento
incremento, decremento = separa_dados(dados_ciclo)
dados_juntos = {}
for nome in dados_ciclo.keys():
    df_incremneto = incremento[nome].copy()
    df_decremento = decremento[nome].copy()
    df_incremneto["parte"] = ["I" for _ in range(len(df_incremneto))]
    df_decremento["parte"] = ["D" for _ in range(len(df_decremento))]
    dados_juntos[nome] = pd.concat([df_incremneto, df_decremento])

incremento_neg, incremento_pos = corta_min(incremento)
decremento_neg, decremento_pos = corta_min(decremento)


# %% [markdown]
# # Aplicação do Modelo PCA e KMeans sem Parâmetros Otimizados
# Nesta célula, utilizamos os parâmetros otimizados do Optuna para configurar os intervalos e executar o modelo de PCA e KMeans. Posteriormente, avaliamos o desempenho nos conjuntos de teste e de treino.
#
def main(nome, dado):
    OUTPUT = "saida"

    # NOME_ESTUDO = input("O nome do estudo, será usado para salvar.")
    NOME_ESTUDO = nome
    path_modelos = os.path.join(OUTPUT, NOME_ESTUDO, "modelos")
    path_imagens = os.path.join(OUTPUT, NOME_ESTUDO, "imagens")
    path_dados = os.path.join(OUTPUT, NOME_ESTUDO, "dados")

    pathlib.Path(path_modelos).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_imagens).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_dados).mkdir(parents=True, exist_ok=True)
    formula_path = open(os.path.join(path_dados, "formulas.txt"), "w")

    # %%
    optimizer = PCAKMeansOptimizer(dado)

    # Executa o modelo PCA e KMeans com os intervalos e parâmetros otimizados
    (
        labels,  # Rótulos originais dos dados
        kmeans,  # Modelo KMeans ajustado
        modelo_data,  # Dados transformados pelo PCA (treino)
        df_pca,  # Dados originais organizados
        model,  # Objeto PCA ajustado
        pca_data_treino,  # Dados transformados do conjunto de treino
        labels_treino,  # Rótulos do conjunto de treino
        pca_data_val,  # Dados transformados do conjunto de validação
        labels_val,  # Rótulos do conjunto de validação
        pca_data_teste,  # Dados transformados do conjunto de teste
        labels_teste,  # Rótulos do conjunto de teste
    ) = optimizer.modelo_pca(None)

    # Obtém os dados transformados pelo PCA para o conjunto de treino
    data = model.results["PC"]

    # Realiza predição no conjunto de teste utilizando o modelo KMeans ajustado
    clusters_teste = kmeans.predict(pca_data_teste)

    # Calcula e exibe a métrica de agrupamento no conjunto de teste
    print(
        "A métrica para o conjunto de teste encontrada foi:",
        v_measure_score(labels_teste, clusters_teste),
    )

    # Realiza predição no conjunto de treino utilizando o modelo KMeans ajustado
    clusters = kmeans.predict(data)

    # Renomeia os clusters para facilitar a interpretação nos gráficos
    clusters_nome = np.array([f"clusters {cluster}" for cluster in clusters])

    # Calcula e exibe a métrica combinando treino, validação e teste
    print(
        "A métrica combinando treino, validação e teste foi:",
        v_measure_score(data.index, clusters),
    )

    # %% [markdown]
    # ## Visualização dos Resultados do PCA
    # Abaixo estão os gráficos gerados para interpretar os resultados do PCA, incluindo a variância explicada, a dispersão dos dados nos componentes principais, e as representações em 3D e biplots com e sem os clusters nomeados.

    # %%
    # Gráfico de variância explicada pelos componentes principais
    # Útil para entender quanto da variância total é capturada por cada componente
    fig, ax = model.plot(figsize=[10, 8], title="")
    arrumar_fig(
        ax,
        "Componentes principais",
        r"Vari$\text{â}$ncia explicada percentual",
        font_size=15,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(path_imagens, "varianca_dados_total.png"), dpi=600)
    plt.close()
    # Extrair variança para uso nos plots
    var_pc1, var_pc2, var_pc3 = [
        str(numero).replace(".", ",")
        for numero in (model.results["variance_ratio"] * 100).round(1)
    ]

    # %% [markdown]
    # ### Dispersão dos Dados no Espaço dos Componentes Principais
    # Estes gráficos mostram a distribuição dos dados no espaço projetado pelos primeiros componentes principais.
    #

    # %%
    with plt.style.context(["science", "scatter"]):
        # Gráfico de dispersão dos dados sem labels
        fig, ax = model.scatter(figsize=[10, 8], title="")
        arrumar_fig(
            ax,
            rf"PC1 ({var_pc1}\% var. expl.)",
            rf"PC2 ({var_pc2}\% var. expl.)",
            font_size=15,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(path_imagens, "scatter_dados_total.png"), dpi=600)
        plt.close()
    with plt.style.context(["science", "scatter"]):
        # Gráfico de dispersão dos dados com labels dos clusters
        fig, ax = model.scatter(labels=clusters_nome, figsize=[10, 8], title="")
        arrumar_fig(
            ax,
            rf"PC1 ({var_pc1}\% var. expl.)",
            rf"PC2 ({var_pc2}\% var. expl.)",
            font_size=15,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(path_imagens, "scatter_cluster_dados_total.png"), dpi=600
        )
        plt.close()

    # %% [markdown]
    # ### Representação 3D dos Dados
    # Estas visualizações em 3D permitem explorar a separação entre os clusters e a distribuição no espaço dos componentes principais.
    #

    # %%
    with plt.style.context(["science", "scatter"]):
        # Gráfico de dispersão 3D dos dados sem labels
        fig, ax = model.scatter3d(figsize=[10, 8], title="")
        arrumar_fig(
            ax,
            rf"PC1 ({var_pc1}\% var. expl.)",
            rf"PC2 ({var_pc2}\% var. expl.)",
            rf"PC3 ({var_pc3}\% var. expl.)",
            loc=1,
            font_size=15,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(path_imagens, "scatter3d_dados_total.png"), dpi=600)
        plt.close()

    with plt.style.context(["science", "scatter"]):
        # Gráfico de dispersão 3D dos dados com labels dos clusters
        fig, ax = model.scatter3d(labels=clusters_nome, figsize=[10, 8], title="")
        arrumar_fig(
            ax,
            rf"PC1 ({var_pc1}\% var. expl.)",
            rf"PC2 ({var_pc2}\% var. expl.)",
            rf"PC3 ({var_pc3}\% var. expl.)",
            loc=1,
            font_size=15,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(path_imagens, "scatter_cluster_3d_dados_total.png"), dpi=600
        )
        plt.close()

    # %% [markdown]
    # ### Biplots: Relação Entre Componentes e Variáveis Originais
    # Os biplots mostram como as variáveis originais contribuem para os componentes principais. Isso ajuda a interpretar os eixos no espaço projetado.
    #

    # %%
    with plt.style.context(["science", "scatter"]):
        # Biplot no espaço 2D dos componentes principais sem labels
        fig, ax = model.biplot(
            figsize=[10, 8], n_feat=5, arrowdict={"fontsize": 12}, title=""
        )
        arrumar_fig(
            ax,
            rf"PC1 ({var_pc1}\% var. expl.)",
            rf"PC2 ({var_pc2}\% var. expl.)",
            font_size=15,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(path_imagens, "biplot_dados_total.png"), dpi=600)
        plt.close()

    with plt.style.context(["science", "scatter"]):
        # Biplot no espaço 2D com labels dos clusters
        fig, ax = model.biplot(
            labels=clusters_nome,
            figsize=[10, 8],
            n_feat=5,
            arrowdict={"fontsize": 12},
            title="",
        )
        arrumar_fig(
            ax,
            rf"PC1 ({var_pc1}\% var. expl.)",
            rf"PC2 ({var_pc2}\% var. expl.)",
            font_size=15,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(path_imagens, "biplot_cluster_dados_total.png"), dpi=600
        )
        plt.close()

    # %% [markdown]
    # ### Biplots 3D
    # Uma visualização tridimensional das contribuições das variáveis originais para os componentes principais.
    #

    # %%
    with plt.style.context(["science", "scatter"]):
        # Biplot 3D sem labels
        fig, ax = model.biplot3d(
            figsize=[10, 8], n_feat=5, arrowdict={"fontsize": 12}, title=""
        )
        arrumar_fig(
            ax,
            rf"PC1 ({var_pc1}\% var. expl.)",
            rf"PC2 ({var_pc2}\% var. expl.)",
            rf"PC3 ({var_pc3}\% var. expl.)",
            loc=1,
            font_size=15,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(path_imagens, "bibplot3d_dados_total.png"), dpi=600)
        plt.close()

    with plt.style.context(["science", "scatter"]):
        # Biplot 3D com labels dos clusters
        fig, ax = model.biplot3d(
            labels=clusters_nome,
            figsize=[10, 8],
            n_feat=5,
            arrowdict={"fontsize": 12},
            title="",
        )
        arrumar_fig(
            ax,
            rf"PC1 ({var_pc1}\% var. expl.)",
            rf"PC2 ({var_pc2}\% var. expl.)",
            rf"PC3 ({var_pc3}\% var. expl.)",
            loc=1,
            font_size=15,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(path_imagens, "biplot_cluster3d_dados_total.png"), dpi=600
        )
        plt.close()

    # %% [markdown]
    # ## Construção do Modelo de Regressão Linear
    # Nesta seção, realizamos o ajuste de um modelo de regressão linear utilizando os três primeiros componentes principais como preditores. As labels foram extraídas e transformadas a partir dos índices dos dados, e o modelo foi ajustado com Mínimos Quadrados Ordinários (OLS).

    # %%
    # Copia os dados principais dos componentes principais (PCs)
    df_ols_pca = model.results["PC"].copy()

    # Criação das labels a partir dos índices, convertendo para float
    # e padronizando os valores para serem utilizados como variáveis dependentes.
    labels = [
        float(index.removesuffix("mmol").replace(",", ".").replace("U", "0"))
        for index in df_ols_pca.index
    ]
    df_ols_pca["labels"] = labels

    # Divisão dos dados em treino e teste (restante)
    treino_pca = df_ols_pca.iloc[: (len(df_ols_pca) // 3) * 2, :]
    teste_pca = df_ols_pca.iloc[(len(df_ols_pca) // 3) * 2 :, :]

    # Fórmula para a regressão utilizando os três primeiros componentes principais
    representacao = "labels ~ PC1 + PC2 + PC3"

    # Criação das matrizes de predição e variável dependente para o modelo
    y_treino, X_treino = Formula(representacao).get_model_matrix(treino_pca)

    # Ajuste do modelo de regressão linear com mínimos quadrados ordinários (OLS)
    modelo = sm.OLS(y_treino, X_treino)
    resultado = modelo.fit()

    # Exibição do resumo do modelo ajustado
    print(resultado.summary())

    # %% [markdown]
    # ## Avaliação do Modelo de Regressão Linear com Conjunto de Teste
    # Nesta etapa, realizamos a comparação entre os valores reais e ajustados/previstos para os conjuntos de treino e teste, utilizando o modelo de regressão linear ajustado. Também calculamos a métrica **Erro Quadrático Médio Raiz (RMSE)** para o conjunto de teste.
    #

    # %%
    # Criação das matrizes de predição e variável dependente para o conjunto de teste
    y_teste, X_teste = Formula(representacao).get_model_matrix(teste_pca)

    # Realização das previsões para o conjunto de teste
    predicoes_teste = resultado.predict(X_teste)

    # Extraindo os coeficientes e o intercepto do modelo
    coeficientes = resultado.params  # Coeficientes estimados pelo modelo

    # Extraindo os nomes das variáveis
    variaveis = coeficientes.index  # Inclui o intercepto

    # Construindo a fórmula da regressão no formato simbólico
    # Criação de variáveis simbólicas correspondentes
    simbolos = {var: symbols(var) for var in variaveis if var != "Intercept"}
    formula_simbolica = sum(
        round(coef, 3) * (simbolos[var] if var != "Intercept" else 1)
        for var, coef in coeficientes.items()
    )

    # Exibição da fórmula utilizando sympy
    # display(Eq(symbols("y"), formula_simbolica))

    print(
        "Equação linear total: ",
        latex(Eq(symbols("y"), formula_simbolica)),
        file=formula_path,
    )

    # Plotando a comparação entre valores reais e previstos para os conjuntos de treino e teste
    plot_comparacao(
        treino_real=treino_pca["labels"],
        treino_predito=resultado.fittedvalues,
        teste_real=teste_pca["labels"],
        teste_predito=predicoes_teste,
        label_real="Valores Reais",
        label_predito="Valores Previstos",
        path=os.path.join(path_imagens, "linear_dados_total.png"),
        path_csv=path_dados,
        suffix="linear",
    )

    # %% [markdown]
    # ## Seleção de Features e Ajuste de Modelo de Regressão Linear
    # Nesta etapa, utilizamos as features mais importantes derivadas do modelo PCA para realizar a seleção de variáveis preditoras, limitando-as a um percentual ajustável pelo usuário. O percentual é definido por meio de entrada direta e aplicado à lista de features ordenadas pela importância. A partir das features selecionadas, preparamos o DataFrame, ajustamos as labels e dividimos os dados em conjuntos de treino e teste. Em seguida, construímos a fórmula da regressão e geramos as matrizes de predição e variável dependente. Por fim, ajustamos um modelo de regressão linear utilizando Mínimos Quadrados Ordinários (OLS) e exibimos um resumo estatístico do modelo ajustado.

    # %%
    # # Seleção das features mais importantes baseadas no PCA
    # featues_importance = model.results["topfeat"]["feature"].copy()

    # # Percentual das features a serem utilizadas, definido pelo usuário
    # percent = float(input("Percentual dos dados para usar"))
    # feature_percent = featues_importance[: int(len(featues_importance) * percent)]

    # # Copia os dados principais baseados nas features mais importantes do PCA
    # df_ols_antes = df_pca.drop("replicata", axis=1).copy()
    # df_ols_antes = df_ols_antes[feature_percent]

    # # Criação de uma string com as colunas para a fórmula da regressão
    # colunas = "+".join(df_ols_antes.columns)

    # # Adiciona as labels como variável dependente no DataFrame
    # df_ols_antes["labels"] = labels

    # # Divisão dos dados em treino e teste
    # treino_antes = df_ols_antes.iloc[: (len(df_ols_antes) // 3) * 2, :]
    # teste_antes = df_ols_antes.iloc[(len(df_ols_antes) // 3) * 2 :, :]

    # # Criação da fórmula de regressão baseada nas features selecionadas
    # representacao = f"labels ~ {colunas}"

    # # Criação das matrizes de predição e variável dependente para o modelo
    # y_treino_antes, X_treino_antes = Formula(representacao).get_model_matrix(treino_antes)

    # # Ajuste do modelo de regressão linear com mínimos quadrados ordinários (OLS)
    # modelo = sm.OLS(y_treino_antes, X_treino_antes)
    # resultado = modelo.fit()

    # # Exibição do resumo do modelo ajustado
    # print(resultado.summary())

    # %% [markdown]
    #
    # ## Avaliação do Modelo de Regressão Linear
    # Nesta etapa, avaliamos o desempenho do modelo ajustado utilizando o conjunto de teste. Para isso, realizamos as previsões com base nas variáveis independentes do conjunto de teste e comparamos os valores previstos com os valores reais. Adicionalmente, extraímos os coeficientes estimados pelo modelo e construímos uma representação simbólica da equação da regressão para melhor interpretabilidade. Por fim, geramos um gráfico de comparação entre os valores reais e previstos, tanto para o conjunto de treino quanto para o de teste.

    # %%
    # # Criação das matrizes de predição e variável dependente para o conjunto de teste
    # y_teste_antes, X_teste_antes = Formula(representacao).get_model_matrix(teste_antes)

    # # Realização das previsões para o conjunto de teste
    # predicoes_teste_antes = resultado.predict(X_teste_antes)

    # # Extraindo os coeficientes e o intercepto do modelo
    # coeficientes_antes = resultado.params  # Coeficientes estimados pelo modelo

    # # Extraindo os nomes das variáveis
    # variaveis_antes = coeficientes_antes.index  # Inclui o intercepto

    # # Construindo a fórmula da regressão no formato simbólico
    # # Criação de variáveis simbólicas correspondentes
    # simbolos_antes = {var: symbols(var) for var in variaveis_antes if var != "Intercept"}
    # formula_simbolica_antes = sum(
    #     round(coef, 3) * (simbolos_antes[var] if var != "Intercept" else 1)
    #     for var, coef in coeficientes_antes.items()
    # )

    # # Exibição da fórmula utilizando sympy
    # display(Eq(symbols("y"), formula_simbolica_antes))
    # print("\n\n\n")
    # print_latex(Eq(symbols("y"), formula_simbolica_antes))

    # # Plotando a comparação entre valores reais e previstos para os conjuntos de treino e teste
    # plot_comparacao(
    #     treino_real=treino_antes["labels"],
    #     treino_predito=resultado.fittedvalues,
    #     teste_real=teste_antes["labels"],
    #     teste_predito=predicoes_teste_antes,
    #     label_real="Valores Reais",
    #     label_predito="Valores Previstos",
    # )

    # %% [markdown]
    # ## Ajuste do Modelo com PySRRegressor
    # Nesta etapa, utilizamos o modelo PySRRegressor para ajustar os dados de treino com base nos três primeiros componentes principais como preditores. Este modelo realiza a busca por equações simbólicas que melhor representem os dados. As operações permitidas para a construção das equações incluem operadores binários, como soma e multiplicação, e operadores unários, como cosseno, exponencial, seno, e outras funções matemáticas básicas.
    #
    # Vale ressaltar que, devido ao uso de paralelismo no processo de busca, os resultados não são determinísticos, podendo variar entre execuções.

    # %%
    # Separação das variáveis independentes (PC1, PC2, PC3) e dependentes (labels) para treino e teste
    X_treino = treino_pca[["PC1", "PC2", "PC3"]]
    y_treino = treino_pca["labels"]
    X_teste = teste_pca[["PC1", "PC2", "PC3"]]
    y_teste = teste_pca["labels"]

    try:
        # Tentativa de carregar o modelo salvo anteriormente, caso queria rodar novamente, apagar o modelo_pysr
        modelo_pysr = joblib.load(
            os.path.join(path_modelos, "modelo_pysr_dados_total.pkl")
        )
    except:
        # Caso o modelo não esteja salvo, ele será instanciado e treinado
        modelo_pysr = PySRRegressor(
            binary_operators=["+", "*", "/"],  # Operadores binários permitidos
            unary_operators=[
                "cos",
                "exp",
                "sin",
                "neg",
                "square",
            ],  # Operadores unários permitidos
            temp_equation_file=True,  # Arquivo temporário para salvar as equações
            random_state=42,  # Reprodutibilidade parcial
            populations=20,  # Número de populações
            population_size=100,  # Tamanho de cada população
            niterations=1000,  # Número total de iterações
        )

        # Ajustando o modelo aos dados de treino
        modelo_pysr.fit(X_treino, y_treino)

        # Salvando o modelo ajustado para uso futuro
        joblib.dump(
            modelo_pysr, os.path.join(path_modelos, "modelo_pysr_dados_total.pkl")
        )

    # Simplificação da melhor expressão encontrada pelo PySR
    print("Expressão encontrada pelo PySR:")
    expressao = simplify(modelo_pysr.sympy())
    # display(expressao)
    print("\n\n\n")
    print("Equação genetica: ", latex(expressao), file=formula_path)

    # Plotando a comparação entre valores reais e previstos para os conjuntos de treino e teste
    plot_comparacao(
        treino_real=y_treino,
        treino_predito=modelo_pysr.predict(X_treino),
        teste_real=y_teste,
        teste_predito=modelo_pysr.predict(X_teste),
        label_real="Valores Reais",
        label_predito="Valores Previstos",
        path=os.path.join(path_imagens, "genetico_dados_total.png"),
        path_csv=path_dados,
        suffix="genetico_dados_total",
    )

    # %% [markdown]
    # ## Ajuste do Modelo com PySRRegressor sem PCA
    # Nesta etapa, utilizamos o modelo PySRRegressor para ajustar os dados de treino com base nas variáveis selecionadas no conjunto de dados original. As colunas foram renomeadas temporariamente para facilitar o processamento pelo modelo, mas a equação final foi traduzida de volta para os nomes originais das variáveis.
    #
    # O modelo busca encontrar equações simbólicas que melhor representem os dados, utilizando operadores binários como soma, multiplicação e divisão, além de operadores unários como cosseno, exponencial, seno, e outras funções matemáticas básicas.
    #
    # O modelo tenta carregar um ajuste anterior salvo. Caso esse arquivo não exista, o modelo é treinado com os dados atuais, e o ajuste resultante é salvo para futuras utilizações.
    #
    # Após o treinamento, a melhor equação encontrada pelo modelo é simplificada e exibida utilizando os nomes originais das colunas, permitindo fácil interpretação. Por fim, comparamos os valores reais e previstos para os conjuntos de treino e teste.

    # %%
    # # Separação das variáveis independentes (features) e dependentes (labels) para treino e teste
    # X_treino = treino_antes.drop("labels", axis=1).copy()
    # dict_colunas = {
    #     f"x_{i}": j for i, j in enumerate(X_treino.columns)
    # }  # Renomeando colunas
    # y_treino = treino_antes["labels"]
    # X_teste = teste_antes.drop("labels", axis=1).copy()
    # y_teste = teste_antes["labels"]

    # # Renomeando colunas dos conjuntos de treino e teste
    # X_treino.columns = dict_colunas.keys()
    # X_teste.columns = dict_colunas.keys()

    # try:
    #     # Tentativa de carregar o modelo salvo anteriormente
    #     modelo_pysr_antes = joblib.load(
    #         os.path.join(path_modelos, "modelo_pysr_original_dados_total.pkl")
    #     )
    # except:
    #     # Caso o modelo não esteja salvo, ele será instanciado e treinado
    #     modelo_pysr_antes = PySRRegressor(
    #         binary_operators=["+", "*", "/"],  # Operadores binários permitidos
    #         unary_operators=[
    #             "cos",
    #             "exp",
    #             "sin",
    #             "neg",
    #             "square",
    #         ],  # Operadores unários permitidos
    #         temp_equation_file=True,  # Arquivo temporário para salvar as equações
    #         random_state=42,  # Reprodutibilidade parcial
    #         populations=20,  # Número de populações
    #         population_size=100,  # Tamanho de cada população
    #         niterations=1000,  # Número total de iterações
    #     )

    #     # Ajustando o modelo aos dados de treino
    #     modelo_pysr_antes.fit(X_treino, y_treino)

    #     # Salvando o modelo ajustado para uso futuro
    #     joblib.dump(
    #         modelo_pysr_antes,
    #         os.path.join(path_modelos, "modelo_pysr_original_dados_total.pkl"),
    #     )

    # # Simplificação da melhor expressão encontrada pelo PySR
    # expressao = simplify(modelo_pysr_antes.sympy())

    # # Substituindo os nomes genéricos pelas colunas originais
    # expressao_original = expressao.subs(
    #     {symbols(k): symbols(v) for k, v in dict_colunas.items()}
    # )
    # display(expressao_original)

    # # Plotando a comparação entre valores reais e previstos para os conjuntos de treino e teste
    # plot_comparacao(
    #     treino_real=y_treino,
    #     treino_predito=modelo_pysr_antes.predict(X_treino),
    #     teste_real=y_teste,
    #     teste_predito=modelo_pysr_antes.predict(X_teste),
    #     label_real="Valores Reais",
    #     label_predito="Valores Previstos",
    # )

    # %% [markdown]
    # ## Salva dados de PCA

    # %%
    model.results["loadings"].to_csv(
        os.path.join(path_dados, "loadings_dados_total.csv")
    )
    model.results["PC"].to_csv(os.path.join(path_dados, "PC_dados_total.csv"))
    model.results["topfeat"].to_csv(os.path.join(path_dados, "topfeat_dados_total.csv"))

    # 2. Combinar 'explained_var' e 'variance_ratio' em um DataFrame
    explained_var_df = pd.DataFrame(
        {
            "PC": ["PC1", "PC2", "PC3"],
            "explained_var": model.results["explained_var"],
            "variance_ratio": model.results["variance_ratio"],
        }
    )

    # Salvar o novo DataFrame como CSV
    explained_var_df.to_csv(
        os.path.join(path_dados, "explained_variance_dados_total.csv"), index=False
    )

    print("Arquivos CSV criados:")
    print("- loadings_dados_total.csv")
    print("- PC_dados_total.csv")
    print("- explained_variance_dados_total.csv")

    # %% [markdown]
    # # Otimização com PCA
    #
    # Aqui inicializamos o otimizador `PCAKMeansOptimizer` com os dados de incremento. Caso um estudo Optuna previamente salvo (`study.pkl`) exista, ele será carregado. Caso contrário, um novo estudo será configurado para otimização. Os resultados do estudo são salvos e exportados como um DataFrame para análise posterior. Se quiser ter uma nova otimização é nescessario apagar ou renomear o `study.pkl` existente.
    #

    # %%
    # Inicializa a classe PCAKMeansOptimizer com os dados de incremento
    optimizer = PCAKMeansOptimizer(dado)

    # Tenta carregar um estudo salvo previamente
    try:
        study = joblib.load(os.path.join(path_modelos, "study.pkl"))
    except:
        sampler = TPESampler(seed=10)
        # Configura um novo estudo Optuna caso o arquivo não exista
        study = optuna.create_study(
            sampler=sampler,
            directions=[
                "maximize",
                "minimize",
                "maximize",
                "maximize",
            ],  # Maximiza e minimiza as métricas desejadas
        )

        # Realiza a otimização
        for _ in range(3):
            study.optimize(
                optimizer.objective,  # Função objetivo a ser otimizada
                n_trials=100,  # Número de tentativas
                n_jobs=12,  # Número de jobs em paralelo
                show_progress_bar=True,  # Exibe barra de progresso
                # gc_after_trial=True,  # Força coleta de lixo após cada tentativa
            )

        # Salva o estudo em um arquivo
        joblib.dump(
            study, os.path.join(path_modelos, "study.pkl")
        )  # Para debug é desativada

    # Exporta os resultados do estudo para um DataFrame
    df = study.trials_dataframe()

    # %% [markdown]
    # ## Ordenação dos Resultados do Estudo
    #
    # Ordena o DataFrame contendo os resultados das tentativas do estudo Optuna com base na métrica de interesse principal (`values_0`) em ordem decrescente. Isso facilita a identificação dos melhores resultados.
    #

    # %%
    # Ordena os resultados do estudo com base na métrica de interesse principal
    df_sorted = df.sort_values("values_0", ascending=False)
    melhores = df_sorted.iloc[:10].sort_values("values_3", ascending=False)
    melhores = melhores[
        melhores["values_3"] == melhores["values_3"].iloc[0]
    ].sort_values("values_1")

    # Exibe os 20 melhores resultados para conferência
    df_sorted.head(20)

    # %% [markdown]
    # ## Exibição dos Melhores Ensaios
    #
    # Este loop percorre os melhores ensaios (`best_trials`) identificados pelo estudo Optuna, exibindo o número do ensaio, os parâmetros utilizados e os valores das métricas obtidas. Isso ajuda a interpretar os resultados e identificar configurações promissoras.
    #

    # %%
    # Itera pelos melhores ensaios do estudo e exibe detalhes
    for trial in study.best_trials:
        print("Ensaio com maior acurácia: ")
        print(f"\tnúmero: {trial.number}")  # Número do ensaio
        print(f"\tparâmetros: {trial.params}")  # Parâmetros utilizados
        print(f"\tvalores: {trial.values}")  # Valores das métricas obtidas
        print("---------------------------")  # Separador para melhor visualização

    # %% [markdown]
    # ## Aplicação do Modelo PCA e KMeans com Parâmetros Otimizados
    # Nesta célula, utilizamos os parâmetros otimizados do Optuna para configurar os intervalos e executar o modelo de PCA e KMeans. Posteriormente, avaliamos o desempenho nos conjuntos de teste e de treino.
    #

    # %%
    # Obtém os parâmetros do segundo trial do estudo
    # Alterar o índice para o trial desejado
    # params = study.trials[int(input("Qual trial usar?"))].params
    params = study.trials[melhores.iloc[0]["number"]].params

    # Constrói a lista de intervalos com base nos parâmetros otimizados
    # O parâmetro "intervalos" define o número de intervalos, e "min_trial" e "max_trial" delimitam os valores
    intervalo_list = [
        [params.pop(f"min_index_{i}"), params.pop(f"max_index_{i}")]
        for i in range(params.pop("intervalos"))
    ]

    # Executa o modelo PCA e KMeans com os intervalos e parâmetros otimizados
    (
        labels,  # Rótulos originais dos dados
        kmeans,  # Modelo KMeans ajustado
        modelo_data,  # Dados transformados pelo PCA (treino)
        df_pca,  # Dados originais organizados
        model,  # Objeto PCA ajustado
        pca_data_treino,  # Dados transformados do conjunto de treino
        labels_treino,  # Rótulos do conjunto de treino
        pca_data_val,  # Dados transformados do conjunto de validação
        labels_val,  # Rótulos do conjunto de validação
        pca_data_teste,  # Dados transformados do conjunto de teste
        labels_teste,  # Rótulos do conjunto de teste
    ) = optimizer.modelo_pca(intervalo_list)

    # Obtém os dados transformados pelo PCA para o conjunto de treino
    data = model.results["PC"]

    # Realiza predição no conjunto de teste utilizando o modelo KMeans ajustado
    clusters_teste = kmeans.predict(pca_data_teste)

    # Calcula e exibe a métrica de agrupamento no conjunto de teste
    print(
        "A métrica para o conjunto de teste encontrada foi:",
        v_measure_score(labels_teste, clusters_teste),
    )

    # Realiza predição no conjunto de treino utilizando o modelo KMeans ajustado
    clusters = kmeans.predict(data)

    # Renomeia os clusters para facilitar a interpretação nos gráficos
    clusters_nome = np.array([f"clusters {cluster}" for cluster in clusters])

    # Calcula e exibe a métrica combinando treino, validação e teste
    print(
        "A métrica combinando treino, validação e teste foi:",
        v_measure_score(data.index, clusters),
    )

    # %% [markdown]
    # ## Visualização dos Resultados do PCA
    # Abaixo estão os gráficos gerados para interpretar os resultados do PCA, incluindo a variância explicada, a dispersão dos dados nos componentes principais, e as representações em 3D e biplots com e sem os clusters nomeados.

    # %%
    # Gráfico de variância explicada pelos componentes principais
    # Útil para entender quanto da variância total é capturada por cada componente
    fig, ax = model.plot(figsize=[10, 8], title="")
    arrumar_fig(
        ax,
        "Componentes principais",
        r"Vari$\text{â}$ncia explicada percentual",
        font_size=15,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(path_imagens, "varianca.png"), dpi=600)
    plt.close()
    # Extrair variança para uso nos plots
    var_pc1, var_pc2, var_pc3 = [
        str(numero).replace(".", ",")
        for numero in (model.results["variance_ratio"] * 100).round(1)
    ]

    # %% [markdown]
    # ### Dispersão dos Dados no Espaço dos Componentes Principais
    # Estes gráficos mostram a distribuição dos dados no espaço projetado pelos primeiros componentes principais.
    #

    # %%
    with plt.style.context(["science", "scatter"]):
        # Gráfico de dispersão dos dados sem labels
        fig, ax = model.scatter(figsize=[10, 8], title="")
        arrumar_fig(
            ax,
            rf"PC1 ({var_pc1}\% var. expl.)",
            rf"PC2 ({var_pc2}\% var. expl.)",
            font_size=15,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(path_imagens, "scatter.png"), dpi=600)
        plt.close()
    with plt.style.context(["science", "scatter"]):
        # Gráfico de dispersão dos dados com labels dos clusters
        fig, ax = model.scatter(labels=clusters_nome, figsize=[10, 8], title="")
        arrumar_fig(
            ax,
            rf"PC1 ({var_pc1}\% var. expl.)",
            rf"PC2 ({var_pc2}\% var. expl.)",
            font_size=15,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(path_imagens, "scatter_cluster.png"), dpi=600)
        plt.close()

    # %% [markdown]
    # ### Representação 3D dos Dados
    # Estas visualizações em 3D permitem explorar a separação entre os clusters e a distribuição no espaço dos componentes principais.
    #

    # %%
    with plt.style.context(["science", "scatter"]):
        # Gráfico de dispersão 3D dos dados sem labels
        fig, ax = model.scatter3d(figsize=[10, 8], title="")
        arrumar_fig(
            ax,
            rf"PC1 ({var_pc1}\% var. expl.)",
            rf"PC2 ({var_pc2}\% var. expl.)",
            rf"PC3 ({var_pc3}\% var. expl.)",
            loc=1,
            font_size=15,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(path_imagens, "scatter3d.png"), dpi=600)
        plt.close()

    with plt.style.context(["science", "scatter"]):
        # Gráfico de dispersão 3D dos dados com labels dos clusters
        fig, ax = model.scatter3d(labels=clusters_nome, figsize=[10, 8], title="")
        arrumar_fig(
            ax,
            rf"PC1 ({var_pc1}\% var. expl.)",
            rf"PC2 ({var_pc2}\% var. expl.)",
            rf"PC3 ({var_pc3}\% var. expl.)",
            loc=1,
            font_size=15,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(path_imagens, "scatter_cluster_3d.png"), dpi=600)
        plt.close()

    # %% [markdown]
    # ### Biplots: Relação Entre Componentes e Variáveis Originais
    # Os biplots mostram como as variáveis originais contribuem para os componentes principais. Isso ajuda a interpretar os eixos no espaço projetado.
    #

    # %%
    with plt.style.context(["science", "scatter"]):
        # Biplot no espaço 2D dos componentes principais sem labels
        fig, ax = model.biplot(
            figsize=[10, 8], n_feat=5, arrowdict={"fontsize": 12}, title=""
        )
        arrumar_fig(
            ax,
            rf"PC1 ({var_pc1}\% var. expl.)",
            rf"PC2 ({var_pc2}\% var. expl.)",
            font_size=15,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(path_imagens, "biplot.png"), dpi=600)
        plt.close()

    with plt.style.context(["science", "scatter"]):
        # Biplot no espaço 2D com labels dos clusters
        fig, ax = model.biplot(
            labels=clusters_nome,
            figsize=[10, 8],
            n_feat=5,
            arrowdict={"fontsize": 12},
            title="",
        )
        arrumar_fig(
            ax,
            rf"PC1 ({var_pc1}\% var. expl.)",
            rf"PC2 ({var_pc2}\% var. expl.)",
            font_size=15,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(path_imagens, "biplot_cluster.png"), dpi=600)
        plt.close()

    # %% [markdown]
    # ### Biplots 3D
    # Uma visualização tridimensional das contribuições das variáveis originais para os componentes principais.
    #

    # %%
    with plt.style.context(["science", "scatter"]):
        # Biplot 3D sem labels
        fig, ax = model.biplot3d(
            figsize=[10, 8], n_feat=5, arrowdict={"fontsize": 12}, title=""
        )
        arrumar_fig(
            ax,
            rf"PC1 ({var_pc1}\% var. expl.)",
            rf"PC2 ({var_pc2}\% var. expl.)",
            rf"PC3 ({var_pc3}\% var. expl.)",
            loc=1,
            font_size=15,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(path_imagens, "bibplot3d.png"), dpi=600)
        plt.close()

    with plt.style.context(["science", "scatter"]):
        # Biplot 3D com labels dos clusters
        fig, ax = model.biplot3d(
            labels=clusters_nome,
            figsize=[10, 8],
            n_feat=5,
            arrowdict={"fontsize": 12},
            title="",
        )
        arrumar_fig(
            ax,
            rf"PC1 ({var_pc1}\% var. expl.)",
            rf"PC2 ({var_pc2}\% var. expl.)",
            rf"PC3 ({var_pc3}\% var. expl.)",
            loc=1,
            font_size=15,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(path_imagens, "biplot_cluster3d.png"), dpi=600)
        plt.close()

    # %% [markdown]
    # ## Construção do Modelo de Regressão Linear
    # Nesta seção, realizamos o ajuste de um modelo de regressão linear utilizando os três primeiros componentes principais como preditores. As labels foram extraídas e transformadas a partir dos índices dos dados, e o modelo foi ajustado com Mínimos Quadrados Ordinários (OLS).

    # %%
    # Copia os dados principais dos componentes principais (PCs)
    df_ols_pca = model.results["PC"].copy()

    # Criação das labels a partir dos índices, convertendo para float
    # e padronizando os valores para serem utilizados como variáveis dependentes.
    labels = [
        float(index.removesuffix("mmol").replace(",", ".").replace("U", "0"))
        for index in df_ols_pca.index
    ]
    df_ols_pca["labels"] = labels

    # Divisão dos dados em treino e teste (restante)
    treino_pca = df_ols_pca.iloc[: (len(df_ols_pca) // 3) * 2, :]
    teste_pca = df_ols_pca.iloc[(len(df_ols_pca) // 3) * 2 :, :]

    # Fórmula para a regressão utilizando os três primeiros componentes principais
    representacao = "labels ~ PC1 + PC2 + PC3"

    # Criação das matrizes de predição e variável dependente para o modelo
    y_treino, X_treino = Formula(representacao).get_model_matrix(treino_pca)

    # Ajuste do modelo de regressão linear com mínimos quadrados ordinários (OLS)
    modelo = sm.OLS(y_treino, X_treino)
    resultado = modelo.fit()

    # Exibição do resumo do modelo ajustado
    print(resultado.summary())

    # %% [markdown]
    # ## Avaliação do Modelo de Regressão Linear com Conjunto de Teste
    # Nesta etapa, realizamos a comparação entre os valores reais e ajustados/previstos para os conjuntos de treino e teste, utilizando o modelo de regressão linear ajustado. Também calculamos a métrica **Erro Quadrático Médio Raiz (RMSE)** para o conjunto de teste.
    #

    # %%
    # Criação das matrizes de predição e variável dependente para o conjunto de teste
    y_teste, X_teste = Formula(representacao).get_model_matrix(teste_pca)

    # Realização das previsões para o conjunto de teste
    predicoes_teste = resultado.predict(X_teste)

    # Extraindo os coeficientes e o intercepto do modelo
    coeficientes = resultado.params  # Coeficientes estimados pelo modelo

    # Extraindo os nomes das variáveis
    variaveis = coeficientes.index  # Inclui o intercepto

    # Construindo a fórmula da regressão no formato simbólico
    # Criação de variáveis simbólicas correspondentes
    simbolos = {var: symbols(var) for var in variaveis if var != "Intercept"}
    formula_simbolica = sum(
        round(coef, 3) * (simbolos[var] if var != "Intercept" else 1)
        for var, coef in coeficientes.items()
    )

    # Exibição da fórmula utilizando sympy
    # display(Eq(symbols("y"), formula_simbolica))
    print(
        "Equação linear otimização: ",
        latex(Eq(symbols("y"), formula_simbolica)),
        file=formula_path,
    )
    # Plotando a comparação entre valores reais e previstos para os conjuntos de treino e teste
    plot_comparacao(
        treino_real=treino_pca["labels"],
        treino_predito=resultado.fittedvalues,
        teste_real=teste_pca["labels"],
        teste_predito=predicoes_teste,
        label_real="Valores Reais",
        label_predito="Valores Previstos",
        path=os.path.join(path_imagens, "linear.png"),
        path_csv=path_dados,
        suffix="linear",
    )

    # %% [markdown]
    # ## Seleção de Features e Ajuste de Modelo de Regressão Linear
    # Nesta etapa, utilizamos as features mais importantes derivadas do modelo PCA para realizar a seleção de variáveis preditoras, limitando-as a um percentual ajustável pelo usuário. O percentual é definido por meio de entrada direta e aplicado à lista de features ordenadas pela importância. A partir das features selecionadas, preparamos o DataFrame, ajustamos as labels e dividimos os dados em conjuntos de treino e teste. Em seguida, construímos a fórmula da regressão e geramos as matrizes de predição e variável dependente. Por fim, ajustamos um modelo de regressão linear utilizando Mínimos Quadrados Ordinários (OLS) e exibimos um resumo estatístico do modelo ajustado.

    # %%
    # # Seleção das features mais importantes baseadas no PCA
    # featues_importance = model.results["topfeat"]["feature"].copy()

    # # Percentual das features a serem utilizadas, definido pelo usuário
    # percent = float(input("Percentual dos dados para usar"))
    # feature_percent = featues_importance[: int(len(featues_importance) * percent)]

    # # Copia os dados principais baseados nas features mais importantes do PCA
    # df_ols_antes = df_pca.drop("replicata", axis=1).copy()
    # df_ols_antes = df_ols_antes[feature_percent]

    # # Criação de uma string com as colunas para a fórmula da regressão
    # colunas = "+".join(df_ols_antes.columns)

    # # Adiciona as labels como variável dependente no DataFrame
    # df_ols_antes["labels"] = labels

    # # Divisão dos dados em treino e teste
    # treino_antes = df_ols_antes.iloc[: (len(df_ols_antes) // 3) * 2, :]
    # teste_antes = df_ols_antes.iloc[(len(df_ols_antes) // 3) * 2 :, :]

    # # Criação da fórmula de regressão baseada nas features selecionadas
    # representacao = f"labels ~ {colunas}"

    # # Criação das matrizes de predição e variável dependente para o modelo
    # y_treino_antes, X_treino_antes = Formula(representacao).get_model_matrix(treino_antes)

    # # Ajuste do modelo de regressão linear com mínimos quadrados ordinários (OLS)
    # modelo = sm.OLS(y_treino_antes, X_treino_antes)
    # resultado = modelo.fit()

    # # Exibição do resumo do modelo ajustado
    # print(resultado.summary())

    # %% [markdown]
    #
    # ## Avaliação do Modelo de Regressão Linear
    # Nesta etapa, avaliamos o desempenho do modelo ajustado utilizando o conjunto de teste. Para isso, realizamos as previsões com base nas variáveis independentes do conjunto de teste e comparamos os valores previstos com os valores reais. Adicionalmente, extraímos os coeficientes estimados pelo modelo e construímos uma representação simbólica da equação da regressão para melhor interpretabilidade. Por fim, geramos um gráfico de comparação entre os valores reais e previstos, tanto para o conjunto de treino quanto para o de teste.

    # %%
    # # Criação das matrizes de predição e variável dependente para o conjunto de teste
    # y_teste_antes, X_teste_antes = Formula(representacao).get_model_matrix(teste_antes)

    # # Realização das previsões para o conjunto de teste
    # predicoes_teste_antes = resultado.predict(X_teste_antes)

    # # Extraindo os coeficientes e o intercepto do modelo
    # coeficientes_antes = resultado.params  # Coeficientes estimados pelo modelo

    # # Extraindo os nomes das variáveis
    # variaveis_antes = coeficientes_antes.index  # Inclui o intercepto

    # # Construindo a fórmula da regressão no formato simbólico
    # # Criação de variáveis simbólicas correspondentes
    # simbolos_antes = {var: symbols(var) for var in variaveis_antes if var != "Intercept"}
    # formula_simbolica_antes = sum(
    #     round(coef, 3) * (simbolos_antes[var] if var != "Intercept" else 1)
    #     for var, coef in coeficientes_antes.items()
    # )

    # # Exibição da fórmula utilizando sympy
    # display(Eq(symbols("y"), formula_simbolica_antes))

    # # Plotando a comparação entre valores reais e previstos para os conjuntos de treino e teste
    # plot_comparacao(
    #     treino_real=treino_antes["labels"],
    #     treino_predito=resultado.fittedvalues,
    #     teste_real=teste_antes["labels"],
    #     teste_predito=predicoes_teste_antes,
    #     label_real="Valores Reais",
    #     label_predito="Valores Previstos",
    # )

    # %% [markdown]
    # ## Ajuste do Modelo com PySRRegressor
    # Nesta etapa, utilizamos o modelo PySRRegressor para ajustar os dados de treino com base nos três primeiros componentes principais como preditores. Este modelo realiza a busca por equações simbólicas que melhor representem os dados. As operações permitidas para a construção das equações incluem operadores binários, como soma e multiplicação, e operadores unários, como cosseno, exponencial, seno, e outras funções matemáticas básicas.
    #
    # Vale ressaltar que, devido ao uso de paralelismo no processo de busca, os resultados não são determinísticos, podendo variar entre execuções.

    # %%
    # Separação das variáveis independentes (PC1, PC2, PC3) e dependentes (labels) para treino e teste
    X_treino = treino_pca[["PC1", "PC2", "PC3"]]
    y_treino = treino_pca["labels"]
    X_teste = teste_pca[["PC1", "PC2", "PC3"]]
    y_teste = teste_pca["labels"]

    try:
        # Tentativa de carregar o modelo salvo anteriormente, caso queria rodar novamente, apagar o modelo_pysr
        modelo_pysr = joblib.load(os.path.join(path_modelos, "modelo_pysr.pkl"))
    except:
        # Caso o modelo não esteja salvo, ele será instanciado e treinado
        modelo_pysr = PySRRegressor(
            binary_operators=["+", "*", "/"],  # Operadores binários permitidos
            unary_operators=[
                "cos",
                "exp",
                "sin",
                "neg",
                "square",
            ],  # Operadores unários permitidos
            temp_equation_file=True,  # Arquivo temporário para salvar as equações
            random_state=42,  # Reprodutibilidade parcial
            populations=20,  # Número de populações
            population_size=100,  # Tamanho de cada população
            niterations=1000,  # Número total de iterações
        )

        # Ajustando o modelo aos dados de treino
        modelo_pysr.fit(X_treino, y_treino)

        # Salvando o modelo ajustado para uso futuro
        joblib.dump(modelo_pysr, os.path.join(path_modelos, "modelo_pysr.pkl"))

    # Simplificação da melhor expressão encontrada pelo PySR
    print("Expressão encontrada pelo PySR:")
    expressao = simplify(modelo_pysr.sympy())
    # display(expressao)
    print("Equação genetico otimização: ", latex(expressao), file=formula_path)

    # Plotando a comparação entre valores reais e previstos para os conjuntos de treino e teste
    plot_comparacao(
        treino_real=y_treino,
        treino_predito=modelo_pysr.predict(X_treino),
        teste_real=y_teste,
        teste_predito=modelo_pysr.predict(X_teste),
        label_real="Valores Reais",
        label_predito="Valores Previstos",
        path=os.path.join(path_imagens, "genetico.png"),
        path_csv=path_dados,
        suffix="genetico",
    )

    # %% [markdown]
    # ## Ajuste do Modelo com PySRRegressor
    # Nesta etapa, utilizamos o modelo PySRRegressor para ajustar os dados de treino com base nas variáveis selecionadas no conjunto de dados original. As colunas foram renomeadas temporariamente para facilitar o processamento pelo modelo, mas a equação final foi traduzida de volta para os nomes originais das variáveis.
    #
    # O modelo busca encontrar equações simbólicas que melhor representem os dados, utilizando operadores binários como soma, multiplicação e divisão, além de operadores unários como cosseno, exponencial, seno, e outras funções matemáticas básicas.
    #
    # O modelo tenta carregar um ajuste anterior salvo. Caso esse arquivo não exista, o modelo é treinado com os dados atuais, e o ajuste resultante é salvo para futuras utilizações.
    #
    # Após o treinamento, a melhor equação encontrada pelo modelo é simplificada e exibida utilizando os nomes originais das colunas, permitindo fácil interpretação. Por fim, comparamos os valores reais e previstos para os conjuntos de treino e teste.

    # %%
    # # Separação das variáveis independentes (features) e dependentes (labels) para treino e teste
    # X_treino = treino_antes.drop("labels", axis=1).copy()
    # dict_colunas = {
    #     f"x_{i}": j for i, j in enumerate(X_treino.columns)
    # }  # Renomeando colunas
    # y_treino = treino_antes["labels"]
    # X_teste = teste_antes.drop("labels", axis=1).copy()
    # y_teste = teste_antes["labels"]

    # # Renomeando colunas dos conjuntos de treino e teste
    # X_treino.columns = dict_colunas.keys()
    # X_teste.columns = dict_colunas.keys()

    # try:
    #     # Tentativa de carregar o modelo salvo anteriormente
    #     modelo_pysr_antes = joblib.load(
    #         os.path.join(path_modelos, "modelo_pysr_original.pkl")
    #     )
    # except:
    #     # Caso o modelo não esteja salvo, ele será instanciado e treinado
    #     modelo_pysr_antes = PySRRegressor(
    #         binary_operators=["+", "*", "/"],  # Operadores binários permitidos
    #         unary_operators=[
    #             "cos",
    #             "exp",
    #             "sin",
    #             "neg",
    #             "square",
    #         ],  # Operadores unários permitidos
    #         temp_equation_file=True,  # Arquivo temporário para salvar as equações
    #         random_state=42,  # Reprodutibilidade parcial
    #         populations=20,  # Número de populações
    #         population_size=100,  # Tamanho de cada população
    #         niterations=1000,  # Número total de iterações
    #     )

    #     # Ajustando o modelo aos dados de treino
    #     modelo_pysr_antes.fit(X_treino, y_treino)

    #     # Salvando o modelo ajustado para uso futuro
    #     joblib.dump(
    #         modelo_pysr_antes, os.path.join(path_modelos, "modelo_pysr_original.pkl")
    #     )

    # # Simplificação da melhor expressão encontrada pelo PySR
    # expressao = simplify(modelo_pysr_antes.sympy())

    # # Substituindo os nomes genéricos pelas colunas originais
    # expressao_original = expressao.subs(
    #     {symbols(k): symbols(v) for k, v in dict_colunas.items()}
    # )
    # display(expressao_original)

    # # Plotando a comparação entre valores reais e previstos para os conjuntos de treino e teste
    # plot_comparacao(
    #     treino_real=y_treino,
    #     treino_predito=modelo_pysr_antes.predict(X_treino),
    #     teste_real=y_teste,
    #     teste_predito=modelo_pysr_antes.predict(X_teste),
    #     label_real="Valores Reais",
    #     label_predito="Valores Previstos",
    # )

    # %% [markdown]
    # ## Salva dados de PCA

    # %%
    model.results["loadings"].to_csv(os.path.join(path_dados, "loadings.csv"))
    model.results["PC"].to_csv(os.path.join(path_dados, "PC.csv"))
    model.results["topfeat"].to_csv(os.path.join(path_dados, "topfeat.csv"))

    # 2. Combinar 'explained_var' e 'variance_ratio' em um DataFrame
    explained_var_df = pd.DataFrame(
        {
            "PC": ["PC1", "PC2", "PC3"],
            "explained_var": model.results["explained_var"],
            "variance_ratio": model.results["variance_ratio"],
        }
    )

    # Salvar o novo DataFrame como CSV
    explained_var_df.to_csv(
        os.path.join(path_dados, "explained_variance.csv"), index=False
    )

    print("Arquivos CSV criados:")
    print("- loadings.csv")
    print("- PC.csv")
    print("- explained_variance.csv")

    # %% [markdown]
    # ## Fecha arquivo formula.txt

    # %%
    formula_path.close()

    # %%


# Separa os dados do ciclo em incremento e decremento
incremento, decremento = separa_dados(dados_ciclo)
dados_juntos = {}
for nome in dados_ciclo.keys():
    df_incremneto = incremento[nome].copy()
    df_decremento = decremento[nome].copy()
    df_incremneto["parte"] = ["I" for _ in range(len(df_incremneto))]
    df_decremento["parte"] = ["D" for _ in range(len(df_decremento))]
    dados_juntos[nome] = pd.concat([df_incremneto, df_decremento])

incremento_neg, incremento_pos = corta_min(incremento)
decremento_neg, decremento_pos = corta_min(decremento)

if __name__ == "__main__":
    for nome, dado in zip(
        [
            "incremento",
            "decremento",
            "dados_juntos",
            "incremento_neg",
            "incremento_pos",
            "decremento_neg",
            "decremento_pos",
        ],
        [
            incremento,
            decremento,
            dados_juntos,
            incremento_neg,
            incremento_pos,
            decremento_neg,
            decremento_pos,
        ],
    ):
        main(nome, dado)
