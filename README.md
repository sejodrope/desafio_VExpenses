
# 📊 Análise de Dados Netflix - Desafio Vexpenses

Este repositório contém minha solução para o **desafio final** do processo seletivo de estágio em análise de dados da **Vexpenses**. O desafio consiste em realizar uma análise de um dataset da Netflix, respondendo a perguntas e extraindo insights interessantes a partir dos dados. 

## 🚀 Sobre o Desafio

No desafio, foi solicitado que realizássemos as seguintes tarefas:

1. **Identificar as colunas presentes no dataset.**
2. **Contar quantos filmes estão disponíveis na Netflix.**
3. **Listar os 5 diretores com mais filmes e séries.**
4. **Identificar quais diretores também atuaram em suas próprias produções.**
5. **Explorar e compartilhar um insight adicional interessante sobre o dataset.**

O código foi desenvolvido em Python, utilizando principalmente a biblioteca `pandas` para manipulação e análise de dados, e as bibliotecas `matplotlib` e `seaborn` para visualizações gráficas.

---

## 🛠️ Ferramentas Utilizadas

- **Python** 3.9
- **pandas** 1.3.3
- **numpy** 1.21.2
- **matplotlib** 3.4.3
- **seaborn** 0.11.2

---

## 📋 Estrutura do Projeto

- **`netflix_analysis.ipynb`**: Notebook com o código completo da análise, incluindo a leitura do dataset, tratamento de dados, visualizações e respostas para o desafio.
- **`netflix_titles.csv`**: Arquivo CSV com os dados da Netflix utilizados na análise.
- **`README.md`**: Este arquivo, descrevendo o projeto e seu propósito.

---

## 🔍 Resumo da Análise

Aqui estão as principais análises realizadas com o dataset da Netflix:

1. **Colunas Presentes**: O dataset contém informações como títulos, diretores, elenco, países, data de adição, e tipo de conteúdo (filme ou série).
2. **Contagem de Filmes**: Existem **4.818 filmes** no catálogo analisado da Netflix.
3. **Top 5 Diretores**: 
   - Rajiv Chilaka
   - Raul Campos
   - Suhas Kadav
   - Marcus Raboy
   - Jay Karas
4. **Diretores que Também Atuaram**: Foram encontrados **diretores que atuaram em 41 produções**.
5. **Insights Adicionais**: Descobri que a maioria dos títulos foi lançada nos últimos 10 anos, com uma grande concentração de lançamentos após 2015.

---

## 📈 Visualizações

Durante a análise, várias visualizações foram geradas para ajudar a interpretar os dados. Abaixo estão alguns exemplos:

- **Distribuição de Conteúdos na Netflix** (Filmes vs. Séries)

![Distribuição de Conteúdos](https://github.com/sejodrope/desafio_VExpenses/blob/main/imagens/distribuicao_conteudos.png)

- **Top 10 Diretores com Mais Produções**

![Top 10 Diretores com Mais Produções](https://github.com/sejodrope/desafio_VExpenses/blob/main/imagens/top10_diretores.png)

- **Evolução dos Diretores que Também Atuam ao Longo dos Anos**

![Evolução dos Diretores que Também Atuam ao Longo dos Anos](https://github.com/sejodrope/desafio_VExpenses/blob/main/imagens/evolucao_diretores.png)

- **Duração Média dos Filmes por Ano de Lançamento**

![Duração Média dos Filmes por Ano de Lançamento](https://github.com/sejodrope/desafio_VExpenses/blob/main/imagens/duracao_media.png)

---

## 🤖 Como Executar

Para executar este projeto na sua máquina local, siga os passos abaixo:

1. Clone o repositório:
   ```bash
   git clone https://github.com/sejodrope/desafio_VExpenses.git
   ```
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute o código Python ou abra o notebook Jupyter para visualizar a análise:
   ```bash
   jupyter notebook desafio_VExpenses.ipynb
   ```

---

## 💡 Insights Finais

Além das respostas solicitadas no desafio, explorei o dataset para gerar insights adicionais sobre classificações, duração de filmes e o crescimento do catálogo ao longo dos anos. 

---

### 📫 Contato

Caso tenha alguma dúvida ou sugestão, fique à vontade para me contatar:

- **E-mail**: vieirasilvajosepedro@gmail.com
- **LinkedIn**: [José Pedro](https://www.linkedin.com/in/josepedrovieira/)
