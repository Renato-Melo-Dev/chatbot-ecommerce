# Project Model Canvas — E-commerce ChatBot
Contexto

O comércio eletrônico movimenta milhões de transações diariamente. Entender padrões de consumo ajuda a prever receita e melhorar decisões de negócio.

## Problema a ser Respondido

Quais fatores influenciam o gasto de um cliente em uma compra de e-commerce?

## Pergunta Norteadora

Qual será a receita gerada por uma compra de acordo com o cliente e seu país?

## Solução Proposta

Desenvolver um chatbot educacional em Streamlit que:

Permita upload de datasets de e-commerce.

Treine modelos de regressão linear para prever a receita de compras.

Mostre métricas de avaliação (RMSE, MAE, R²).

Explique a importância das variáveis no gasto do cliente (ex.: país, tipo de produto, quantidade, preço unitário).

Responda perguntas do usuário via chatbot regrado.

## Desenho de Arquitetura

O sistema é organizado em camadas:

Interface (app/): Streamlit para upload, treino, predição e perguntas.

Core (core/): módulos para dados, pré-processamento, modelos, explicabilidade e chatbot.

Dados (data/): arquivos brutos, tratados e modelos treinados.

Documentação (docs/): PMC, arquitetura, governança de dados e testes.

## Resultados Esperados

Modelo de regressão capaz de prever a receita com erro aceitável.

Relatório das variáveis mais impactantes nos gastos dos clientes.

Deploy em Streamlit Cloud com documentação completa no GitHub.