[![DOI](https://zenodo.org/badge/894568363.svg)](https://doi.org/10.5281/zenodo.14224620)

# Análise e Otimização de Dados Voltamétricos  

Este repositório contém o código desenvolvido para a análise e otimização de dados obtidos por meio de técnicas de voltametria. A metodologia utiliza ferramentas avançadas de aprendizado de máquina, redução de dimensionalidade e otimização de parâmetros, com o objetivo de processar dados de correntes associados a potenciais crescentes, decrescentes e suas combinações.  

## Principais Funcionalidades  

- **Redução de dimensionalidade com PCA**: Aplicação de Análise de Componentes Principais para identificar padrões significativos e simplificar os dados.  
- **Otimização de intervalos de potencial com Optuna**: Seleção eficiente de variáveis baseada em métricas como medida V e índice de silhueta.  
- **Normalização de dados**: Uso de StandardScaler para garantir uniformidade na escala das variáveis antes da análise.  
- **Clusterização e análise de agrupamentos**: Aplicação de K-means e avaliação com métricas de desempenho de clusters.  
- **Modelos preditivos**: Desenvolvimento de ajustes lineares com Statsmodels e regressão simbólica com PySR para explorar relações entre dados transformados e labels originais.  
- **Visualizações gráficas**: Geração de biplots e gráficos de dispersão em 2D e 3D para facilitar a interpretação dos resultados.  
- **Comparação de métodos**: Análise de resultados com e sem otimização para validar o impacto do processo.  

## Tecnologias Utilizadas  

- [**Optuna**](https://optuna.org/) (AKIBA et al., 2019): Otimização de hiperparâmetros.  
- [**Scikit-Learn**](https://scikit-learn.org/) (PEDREGOSA et al., 2011): Pré-processamento, clusterização e suporte ao PCA.  
- [**PCA Library**](https://pypi.org/project/PCA/) (TASKESEN, 2024): Implementação simplificada do PCA.  
- [**Statsmodels**](https://www.statsmodels.org/) (SEABOLD; PERKTOLD, 2010): Regressão linear.  
- [**PySR**](https://pysr.readthedocs.io/) (CRANMER, 2023): Regressão simbólica.  

## Status do Projeto  

Este projeto está em **fase de desenvolvimento**. Atualizações regulares serão feitas, incluindo melhorias no código, adição de novas funcionalidades e organização do repositório.  

## Referências  

1. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework*.  
2. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*.  
3. Taskesen, E. (2024). *PCA Library*.  
4. Seabold, S., & Perktold, J. (2010). *Statsmodels: Econometric and Statistical Modeling with Python*.  
5. Cranmer, M. (2023). *PySR: Fast & Interpretable Symbolic Regression*.  
