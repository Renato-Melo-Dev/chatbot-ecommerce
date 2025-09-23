📊 Modelagem de Dados

Este documento descreve a modelagem de dados em três camadas: System of Record (SOR), System of Truth (SOT) e Specification (SPEC).

## 1. System of Record (SOR)

Tabela: SOR

Representa os dados brutos, exatamente como chegam do arquivo .csv. É a primeira camada de armazenamento, garantindo que tenhamos uma cópia fiel dos dados originais.

Propósito: Ingestão e arquivamento dos dados brutos.
Estrutura: Colunas correspondem ao dataset original, sem limpeza ou transformação.
| Coluna      | Tipo de Dado (SQL) | Descrição                  |
| ----------- | ------------------ | -------------------------- |
| InvoiceNo   | TEXT               | Número da fatura/compra.   |
| StockCode   | TEXT               | Código único do produto.   |
| Description | TEXT               | Descrição do produto.      |
| Quantity    | REAL               | Quantidade comprada.       |
| InvoiceDate | TEXT               | Data da compra.            |
| UnitPrice   | REAL               | Preço unitário do produto. |
| CustomerID  | TEXT               | Identificador do cliente.  |
| Country     | TEXT               | País do cliente.           |

## 2. System of Truth (SOT) Tabela: SOT

Representa a "versão única da verdade". Os dados da SOR são limpos, padronizados e enriquecidos. É a base confiável para análises e modelagem.

Propósito: Fornecer dados limpos e consistentes.
Transformações Aplicadas:

Remoção de registros com Quantity <= 0 ou UnitPrice <= 0.

Preenchimento de valores nulos em CustomerID e Description.

Conversão de colunas para tipos consistentes.

Criação da coluna TotalPrice = Quantity * UnitPrice.

| Coluna      | Tipo de Dado (SQL) | Descrição                                      |
| ----------- | ------------------ | ---------------------------------------------- |
| InvoiceNo   | TEXT               | Número da fatura.                              |
| StockCode   | TEXT               | Código do produto.                             |
| Description | TEXT               | Produto padronizado.                           |
| Quantity    | REAL               | Quantidade numérica.                           |
| UnitPrice   | REAL               | Preço unitário.                                |
| CustomerID  | TEXT               | Cliente (ou “Desconhecido”).                   |
| Country     | TEXT               | País do cliente.                               |
| TotalPrice  | REAL               | Valor total da compra (Quantity \* UnitPrice). |


## 3. Specification (SPEC) Tabela: spec_sales

Camada final, pronta para ser consumida em modelos de machine learning. Contém as variáveis independentes (features) e a variável alvo (TotalPrice).

Propósito: Fornecer dataset já limpo e pronto para modelagem.
Estrutura: Geralmente uma agregação da SOT por cliente, país, produto e descrição.

| Coluna      | Tipo de Dado (SQL) | Descrição                                       |
| ----------- | ------------------ | ----------------------------------------------- |
| CustomerID  | TEXT               | Identificador do cliente.                       |
| Country     | TEXT               | País do cliente.                                |
| StockCode   | TEXT               | Código do produto.                              |
| Description | TEXT               | Produto.                                        |
| Quantity    | REAL               | Quantidade agregada.                            |
| UnitPrice   | REAL               | Preço médio do produto.                         |
| TotalPrice  | REAL               | Valor total agregado da compra (variável alvo). |
