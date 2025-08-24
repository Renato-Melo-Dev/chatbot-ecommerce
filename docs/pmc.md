# Project Model Canvas — Kaggle Chatbot MVP (E-commerce)

# Contexto

As compras online se tornaram parte essencial do cotidiano.
O dataset E-commerce Public Dataset for Machine Learning contém informações sobre transações de consumidores (idade, gênero, produtos, quantidade, preço unitário, etc.).
O objetivo educacional é usar esse conjunto para analisar padrões de consumo e construir um chatbot interativo.

# Problema a ser Respondido

Quais características estão associadas a maiores gastos em e-commerce?
É possível prever a receita gerada por uma compra com base nos atributos do consumidor e da transação?

# Pergunta Norteadora

Quanto uma transação/compra gera de receita?

Quais variáveis mais impactam no gasto do consumidor?

É possível treinar um modelo simples para estimar a receita?

# Solução Proposta

Desenvolver um chatbot educacional em Streamlit que:

- Permita upload do dataset de e-commerce.

- Treine modelos de:

   Regressão linear (predição da receita).

   Árvores de decisão (comparação de desempenho).

- Mostre métricas de avaliação (RMSE, MAE, R²).

- Explique a importância das variáveis no gasto (ex.: categoria do produto, quantidade, preço unitário).

- Responda perguntas do usuário via chatbot regrado.

# Desenho de Arquitetura

O sistema será estruturado em camadas:

- Interface (app/): Streamlit como front-end para upload, treino e perguntas.

- Core (core/): módulos para dados, features, modelos, explicabilidade e chatbot.

- Dados (data/): pastas para armazenar arquivos brutos, tratados e modelos treinados.

- Documentação (docs/): PMC, arquitetura, governança e testes.

# Resultados Esperados

Modelo de regressão com boa explicabilidade e erro aceitável (RMSE baixo).

Relatório com as variáveis que mais influenciam nos gastos.

Deploy em Streamlit Cloud com documentação completa no GitHub.