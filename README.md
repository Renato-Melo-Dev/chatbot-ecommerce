# Chatbot de Análise de Gastos em E-commerce

## Descrição do Projeto
Este projeto é um protótipo educacional que permite explorar padrões de consumo em e-commerce e prever o valor de transações.  
Através de um chatbot interativo, o usuário pode consultar insights sobre gastos de clientes e fatores que influenciam o ticket médio.

## Problema Central
Empresas querem entender quais características de clientes e produtos impactam diretamente no valor gasto.  
Nosso objetivo é responder perguntas como:  
- Qual categoria de produto gera mais receita?  
- Qual é o gasto médio por cliente?  
- É possível prever o valor de uma compra?

## Dataset
O projeto utiliza o **E-commerce Public Dataset for Machine Learning**, que contém:  
- Informações do cliente (idade, gênero, localização)  
- Detalhes das transações (produto, categoria, quantidade, preço unitário)  
- Histórico de compras  

## Objetivo
- Explorar padrões de consumo e identificar fatores que aumentam o ticket médio.  
- Construir modelos simples de regressão e árvore de decisão para estimar a receita.  
- Criar um chatbot que responda perguntas sobre gastos e comportamento dos clientes.

## Como Rodar
1. Clonar o repositório:  
```bash
git clone https://github.com/Renato-Melo-Dev/chatbot-ecommerce.git
```
2. Abrir o notebook em notebooks/ para exploração inicial. 
3. Rodar o app (opcional): 
```bash
streamlit run main_app.py
