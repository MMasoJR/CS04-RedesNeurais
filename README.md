# Sprint Challenge 4: Previsão de Acidentes com LSTMs

Este repositório contém o projeto desenvolvido para o *Challenge Sprint 4*, focado na construção de uma solução de *deep learning* para prever padrões de acidentes em rodovias federais brasileiras.

## 1. 🎯 Objetivo do Projeto

O objetivo principal é construir um modelo de Rede Neural Recorrente (LSTM) capaz de antecipar padrões de acidentes. A solução visa apoiar decisões estratégicas em três frentes principais:

1.  **Prevenção de Riscos:** Identificar padrões que levam a acidentes.
2.  **Precificação de Seguros:** Estimar o risco com base em séries temporais.
3.  **Planejamento Logístico:** Antecipar interrupções em rodovias.

## 2. 📊 O Problema (Target Escolhido)

Dentre as várias possibilidades sugeridas, escolhemos por focar na **previsão do número total de acidentes (`n_acidentes`) por dia**, configurando um problema de **regressão de séries temporais**.

**Justificativa:** O número de acidentes diários é um indicador de risco direto e fundamental. Prever sua tendência e volume é essencial para o planejamento logístico e alocação de recursos de prevenção pela PRF e seguradoras.

## 3. 🗃️ Dataset

Foi utilizada a base de dados pública da Polícia Rodoviária Federal (PRF).

* **Fonte:** [Dados Abertos da PRF](https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf)
* **Arquivo Utilizado:** `acidentes2024_todas_causas_tipos.csv` renomeado para `acidentes2024.csv`
* **Colunas Disponíveis (Exemplos):** `id`, `data_inversa`, `dia_semana`, `horario`, `uf`, `br`, `km`, `causa_acidente`, `classificacao_acidente`, `fase_dia`, `condicao_metereologica`, `mortos`, `feridos_graves`, etc.

## 4. 🛠️ Metodologia e Pipeline

O projeto foi estruturado em um pipeline de 6 etapas principais, desde a seleção dos dados brutos até a avaliação do modelo.

### Etapa 1: Seleção de Features
Com base nos objetivos do desafio, foram selecionadas 15 colunas do dataset original para servirem como *features* ou como base para o *target*.

As colunas selecionadas foram:

```
'id' (para contagem), 'data_inversa', 'dia_semana', 'horario', 'fase_dia', 'uf', 'br', 'km', 'causa_acidente', 'tipo_acidente',
'condicao_metereologica', 'tipo_pista', 'classificacao_acidente', 'mortos', 'feridos_graves', 'feridos_leves'
```

### Etapa 2: Pré-processamento e Limpeza
Os dados brutos foram tratados para corrigir tipos e valores ausentes:
* `data_inversa` foi convertida para o formato `datetime`.
* `horario` foi convertido de string (HH:MM:SS) para um valor numérico (0 a 23).
* `km` foi convertido para numérico, tratando vírgulas como separadores decimais.
* Valores ausentes (NaN) em colunas numéricas (vítimas) foram preenchidos com `0`.
* Valores ausentes em colunas categóricas (`fase_dia`, etc.) foram preenchidos com a string `"Desconhecido"`.

### Etapa 3: Agregação Temporal
Os dados, originalmente registrados por acidente, foram agregados em uma granularidade **diária**. Isso é crucial para transformar o problema em uma série temporal.
* **Target:** `n_acidentes` (contagem de `id`).
* **Features Numéricas:** `sum` (soma) para vítimas; `mean` (média) para `horario` e `km`.
* **Features Categóricas:** `mode` (moda) para `uf`, `br`, `causa_acidente`, `condicao_metereologica`, etc.

### Etapa 4: Encoding e Normalização
Redes neurais exigem entradas numéricas e normalizadas:
1.  **Label Encoding:** Colunas categóricas (como `uf`, `br`, `causa_acidente`) foram convertidas em números.
2.  **MinMax Scaling:** Todas as 15 *features* foram normalizadas para o intervalo `[0, 1]`. Isso é vital para que o modelo não atribua pesos indevidos a features com escalas diferentes.

### Etapa 5: Criação de Sequências
Uma LSTM não olha para um dia isoladamente. Ela aprende com sequências. Os dados foram transformados usando **(`window_size = 7`)**.
* **X (Features):** Dados dos últimos 7 dias (com 15 features cada).
* **y (Target):** O número de acidentes do 8º dia.
* **Formato Final (LSTM):** `(359, 7, 15)`

### Etapa 6: Arquitetura do Modelo LSTM
A arquitetura do modelo foi ajustada para evitar o *overfitting* (identificado nos testes iniciais). A arquitetura final é:

```
Camada LSTM (units=30, activation='relu', return_sequences=True)
Dropout (0.3)
Camada LSTM (units=30, activation='relu')
Dropout (0.3)
Camada Densa (units=1) - Camada de saída para a regressão
```

* **Otimizador:** `adam`
* **Função de Perda:** `mean_squared_error` (MSE)

## 5. 📈 Resultados e Análise

A avaliação do modelo foi focada em duas frentes: a performance do treinamento e a precisão da previsão.

### Curvas de Treinamento
O gráfico de *Loss* mostra que o modelo aprendeu de forma saudável. A perda de treino (`loss`) diminuiu, e a perda de validação (`val_loss`) estabilizou, indicando que o *overfitting* foi controlado.

*<img width="859" height="555" alt="image" src="https://github.com/user-attachments/assets/6d2adca6-a503-44fe-a08e-d7cfd4b00024" />*

### Métricas de Avaliação
No conjunto de teste (separado temporalmente), as métricas de regressão foram:

* **RMSE (Root Mean Squared Error): 488.67**
* **MAE (Mean Absolute Error): 335.06**

### Análise dos Resultados
A análise conjunta das métricas e do gráfico de previsão (Passo 11) revela a principal conclusão deste projeto:

**O modelo foi bem-sucedido em aprender a tendência central (a média) dos acidentes, mas falhou em prever a alta volatilidade (os picos e vales extremos).**

*<img width="1250" height="629" alt="image" src="https://github.com/user-attachments/assets/c6b3d826-c725-4259-af57-32533ce4a4b9" />*

* **Interpretação (Linha Vermelha vs. Azul):** A linha de previsão (vermelha) é "suave" e segue o nível médio dos acidentes (entre 1500-1800). A linha real (azul) é "espinhosa", com picos que ultrapassam 4000 acidentes.
* **Interpretação (Métricas):** O RMSE (488) ser significativamente maior que o MAE (335) é a prova numérica dessa falha. O RMSE penaliza erros grandes de forma quadrática, e os "picos" que o modelo errou puxaram essa métrica para cima.
* **Hipótese:** Esses picos de acidentes provavelmente correspondem a eventos sazonais (como feriados de fim de ano), que não foram explicitamente modelados como *features*.

## 6. 🏁 Conclusão

O *baseline* do modelo LSTM é funcional e útil para estimar o **nível médio de risco** (útil para precificação de seguros).

Para evoluir o modelo para **prevenção de riscos** (que exige a previsão de picos), a recomendação principal é focar em **Engenharia de Features**, criando colunas específicas para `feriado` ou `vespera_feriado`.

## 7. 🚀 Como Executar

O projeto foi desenvolvido em um notebook Google Colab.

1.  Clone este repositório.
2.  Abra o notebook (`.ipynb`) no Google Colab.
3.  **Importante:** Faça o upload manual do arquivo `acidentes2024.csv` (baixado do [site da PRF](https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf)) para o ambiente do Colab.
4.  Execute todas as células na ordem, do "Passo 1" ao "Passo 12".

## 8. 🗂️ Arquivos no Repositório

Conforme os entregáveis do desafio:

* `CS4_Redes.ipynb`: Código completo contendo o pré-processamento e o modelo.
* `lstm_acidentes.keras`: O modelo treinado e salvo.
* `README.md`: Este relatório/documentação.
* `link_video.txt`: Arquivo contendo o link da apresentação de 5 minutos no YouTube.
