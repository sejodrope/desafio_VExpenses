"""
Análise de Dados Netflix - Desafio Vexpenses
Com visualizações aprimoradas usando Seaborn
"""

# Importando bibliotecas necessárias
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações iniciais para a exibição de dados
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)

# Configurando o estilo das visualizações
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['figure.dpi'] = 100

def verificar_dependencias():
    """
    Verifica se todas as dependências necessárias estão instaladas.
    Caso falte alguma, informa os pacotes a serem instalados.
    """
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn'
    }
    
    missing_packages = []
    for package, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Alguns pacotes necessários não estão instalados:")
        print("Execute o seguinte comando para instalar:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    return True

def carregar_dados(caminho_arquivo):
    """
    Carrega e prepara o dataset para análise.
    
    Parâmetros:
    - caminho_arquivo: Caminho do arquivo CSV a ser carregado
    
    Retorna:
    - DataFrame contendo os dados tratados
    """
    try:
        print("Carregando dataset...")
        df = pd.read_csv(caminho_arquivo)
        
        # Tratamento de valores nulos em colunas relevantes
        df = df.assign(
            director=df['director'].fillna('No Director'),
            cast=df['cast'].fillna('No Cast'),
            country=df['country'].fillna('Country Not Listed'),
            rating=df['rating'].fillna('Not Rated')
        )
        
        # Convertendo a coluna 'date_added' para o tipo datetime
        df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
        
        print(f"Dataset carregado com sucesso! Dimensões: {df.shape}")
        return df
    
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em {caminho_arquivo}")
        return None
    except Exception as e:
        print(f"Erro ao carregar dados: {str(e)}")
        return None

def mostrar_colunas(df):
    """
    Exibe informações e estatísticas descritivas sobre as colunas do dataset.
    """
    print("\n=== 1. ANÁLISE DAS COLUNAS DO DATASET ===")
    
    # Exibindo informações básicas do dataset
    print("\nInformações sobre o dataset:")
    print(df.info())
    
    # Estatísticas descritivas das colunas
    print("\nEstatísticas descritivas:")
    print(df.describe(include='all'))

def contar_filmes(df):
    """
    Analisa a distribuição de filmes e séries no dataset.
    Exibe uma visualização gráfica para essa distribuição.
    """
    print("\n=== 2. ANÁLISE DE FILMES E SÉRIES ===")
    
    # Contagem por tipo de conteúdo (Filmes ou Séries)
    type_counts = df['type'].value_counts()
    
    # Gráfico de barras com a distribuição de conteúdo
    plt.figure(figsize=(10, 6))
    sns.barplot(x=type_counts.index, y=type_counts.values, palette="Blues_d")
    plt.title('Distribuição de Conteúdo na Netflix', fontsize=14)
    plt.xlabel('Tipo de Conteúdo', fontsize=12)
    plt.ylabel('Quantidade', fontsize=12)
    plt.show()
    
    # Exibindo a contagem de cada tipo no console
    print("\nQuantidade por tipo de conteúdo:")
    for tipo, contagem in type_counts.items():
        print(f"{tipo}: {contagem} títulos ({contagem/len(df)*100:.1f}%)")

def top_diretores(df):
    """
    Exibe os 10 diretores com mais produções no catálogo da Netflix.
    Inclui uma visualização gráfica para os principais diretores.
    """
    print("\n=== 3. ANÁLISE DOS DIRETORES ===")
    
    # Analisando os diretores e contando suas produções
    all_directors = df[df['director'] != 'No Director']['director'].str.split(', ').explode()
    top_directors = all_directors.value_counts().head(10)
    
    # Gráfico de barras com os 10 diretores principais
    plt.figure(figsize=(14, 7))
    sns.barplot(x=top_directors.values, y=top_directors.index, palette="viridis")
    plt.title('Top 10 Diretores com Mais Produções', fontsize=14)
    plt.xlabel('Número de Produções', fontsize=12)
    plt.ylabel('Diretor', fontsize=12)
    plt.show()
    
    # Exibindo os 5 principais diretores no console
    print("\nTop 5 diretores com mais produções:")
    for i, (director, count) in enumerate(top_directors.head().items(), 1):
        filmes = df[df['director'].str.contains(director, na=False)]['title'].tolist()
        print(f"{i}. {director}: {count} produções")
        print(f"   Alguns títulos: {', '.join(filmes[:3])}")

def diretores_atores(df):
    """
    Identifica diretores que também atuaram em suas produções.
    Inclui uma visualização da evolução dessas produções ao longo dos anos.
    """
    print("\n=== 4. DIRETORES QUE TAMBÉM ATUARAM ===")
    
    def find_director_actor(row):
        if row['director'] == 'No Director' or row['cast'] == 'No Cast':
            return []
        directors = set(row['director'].split(', '))
        cast = set(row['cast'].split(', '))
        return list(directors.intersection(cast))
    
    # Criando uma nova coluna para identificar diretores que atuaram
    df['director_actor'] = df.apply(find_director_actor, axis=1)
    director_actor_df = df[df['director_actor'].apply(len) > 0]
    
    # Gráfico mostrando a evolução de diretores-atores por ano
    director_actor_by_year = director_actor_df['release_year'].value_counts().sort_index()
    plt.figure(figsize=(15, 6))
    sns.lineplot(x=director_actor_by_year.index, y=director_actor_by_year.values, marker='o')
    plt.title('Evolução de Diretores-Atores ao Longo dos Anos', fontsize=14)
    plt.xlabel('Ano', fontsize=12)
    plt.ylabel('Número de Produções', fontsize=12)
    plt.show()
    
    # Exibindo exemplos no console
    print(f"\nForam encontrados {len(director_actor_df)} títulos onde o diretor também atuou.")
    
    if not director_actor_df.empty:
        print("\nAlguns exemplos:")
        for _, row in director_actor_df.head().iterrows():
            print(f"Título: {row['title']}")
            print(f"Diretor(es)/Ator(es): {', '.join(row['director_actor'])}")
            print(f"Ano: {row['release_year']}")
            print("---")

def encontrar_insights(df):
    """
    Explora o dataset em busca de insights adicionais interessantes.
    Inclui visualizações gráficas para insights sobre ano de lançamento, classificações e duração de filmes.
    """
    print("\n=== 5. INSIGHTS ADICIONAIS ===")
    
    # Insight 1: Distribuição de conteúdo por ano
    plt.figure(figsize=(15, 6))
    sns.histplot(data=df, x='release_year', hue='type', multiple="stack", palette="coolwarm")
    plt.title('Distribuição de Conteúdo por Ano de Lançamento', fontsize=14)
    plt.xlabel('Ano', fontsize=12)
    plt.ylabel('Número de Títulos', fontsize=12)
    plt.show()
    
    # Insight 2: Ratings mais comuns
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, y='rating', order=df['rating'].value_counts().index[:10], palette="magma")
    plt.title('Top 10 Classificações mais Comuns', fontsize=14)
    plt.xlabel('Número de Títulos', fontsize=12)
    plt.ylabel('Classificação', fontsize=12)
    plt.show()
    
    # Insight 3: Duração média dos filmes por ano
    movies_df = df[df['type'] == 'Movie'].copy()
    movies_df['duration'] = movies_df['duration'].str.extract('(\d+)').astype(float)
    
    plt.figure(figsize=(15, 6))
    sns.boxplot(data=movies_df, x='release_year', y='duration', color='skyblue')
    plt.title('Duração dos Filmes por Ano', fontsize=14)
    plt.xlabel('Ano', fontsize=12)
    plt.ylabel('Duração (minutos)', fontsize=12)
    plt.xticks(rotation=90)
    plt.show()

# Exemplo de fluxo completo de análise
if verificar_dependencias():
    df_netflix = carregar_dados('netflix_titles.csv')
    
    if df_netflix is not None:
        mostrar_colunas(df_netflix)
        contar_filmes(df_netflix)
        top_diretores(df_netflix)
        diretores_atores(df_netflix)
        encontrar_insights(df_netflix)
