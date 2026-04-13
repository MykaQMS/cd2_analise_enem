# 📊 Análise do Impacto das Características Escolares no Desempenho no ENEM

## 🎯 Objetivo

Este projeto tem como objetivo investigar quais fatores escolares e socioeconômicos influenciam o desempenho dos alunos no ENEM, utilizando dados reais e aplicando técnicas de estatística descritiva e inferencial.

O trabalho é desenvolvido como parte da disciplina **Ciência de Dados II**, com foco em análise estatística aplicada.

---

## 🧠 Problema de Pesquisa

Quais características das escolas e dos alunos estão mais associadas ao desempenho no ENEM?

---

## 📚 Bases de Dados

- Microdados do ENEM 2024  
- Microdados do Censo Escolar 2025  

> ⚠️ Os dados não estão versionados neste repositório devido ao tamanho.

### 📥 Como obter os dados

1. Baixar os microdados no site do INEP
2. Extrair os arquivos
3. Salvar em:


data/raw/


---

## 🛠️ Tecnologias Utilizadas

- Python
- Pandas
- NumPy
- Matplotlib / Seaborn
- Scikit-learn
- Statsmodels
- Jupyter Notebook

---

## 📂 Estrutura do Projeto


cd2_analise_enem/
│
├── data/
│ ├── raw/ # Dados brutos (não versionados)
│ └── processed/ # Dados tratados
│
├── notebooks/ # Análises exploratórias
│
├── src/ # Scripts de processamento e análise
│
├── outputs/
│ ├── figures/ # Gráficos gerados
│ └── tables/ # Tabelas finais
│
├── requirements.txt
├── .gitignore
└── README.md


---

## ⚙️ Como Executar o Projeto

### 1. Clonar o repositório

```bash
git clone <url-do-repositorio>
cd cd2_analise_enem

```

2. Criar ambiente virtual

``` bash

python -m venv venv

```


Ativar:

Windows:
``` bash

venv\Scripts\activate

```

3. Instalar dependências

``` bash
pip install -r requirements.txt

```

4. Executar análises

Abra os notebooks:

notebooks/
📊 Metodologia

O projeto segue três etapas principais:

🔹 1. Análise Exploratória
Estatística descritiva
Distribuições
Visualizações
🔹 2. Análise Inferencial
Testes de hipótese
Comparação entre grupos
Correlações
🔹 3. Modelagem
Regressão múltipla
Análise de fatores explicativos
(Opcional) Classificação e agrupamento

📈 Resultados Esperados
Identificação de fatores que impactam o desempenho no ENEM
Comparação entre diferentes perfis de escolas
Insights sobre desigualdade educacional

🚀 Possíveis Extensões
Clusterização de escolas
Modelos preditivos de desempenho
Dashboard em Power BI

👤 Autor
Mykael Querido

📝 Observações
Este projeto tem caráter educacional e foi desenvolvido para fins acadêmicos.