# CECIERJ – IA e ML para Soluções Práticas

Projeto GitHub alinhado à **ementa oficial** do curso _"Inteligência artificial e aprendizado de máquina para soluções práticas"_
do eixo **Gestão de dados e tecnologia da informação** (Fundação Cecierj – Edital 2025.1). O foco é **aplicar algoritmos com Scikit-learn (Python)** em problemas reais, seguindo as etapas da ementa: introdução, fundamentos, pré-processamento, aprendizado supervisionado, avaliação, não supervisionado (incluindo PCA), introdução a reforço e ética em IA.

> Referência da ementa (Edital 2025.1 – Diretoria de Extensão/Fundação Cecierj): ver seção **Referências**.

---

## Estrutura

```text
cecierj-ia-ml-solucoes-praticas/
├── env/
│   └── requirements.txt
├── notebooks/
│   ├── 01_intro_ia_ml.ipynb
│   ├── 02_python_scikitlearn_setup.ipynb
│   ├── 03_preprocessamento_dados.ipynb
│   ├── 04_supervisionado_classificacao.ipynb
│   ├── 05_supervisionado_regressao.ipynb
│   ├── 06_avaliacao_modelos.ipynb
│   ├── 07_nao_supervisionado_kmeans_pca.ipynb
│   ├── 08_reforco_introducao.ipynb
│   └── 09_etica_futuro_ia.ipynb
├── reports/
│   └── relatorio_final.md
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   └── evaluation.py
├── data/  # (vazia – usamos datasets do scikit-learn)
├── LICENSE
├── .gitignore
└── README.md
```

## Como executar

1. Crie o ambiente (Windows PowerShell):
   ```pwsh
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r env/requirements.txt
   python -m ipykernel install --user --name cecierj-ia-ml
   jupyter lab
   ```
   (Linux/macOS: `source .venv/bin/activate`)

2. Abra os notebooks na pasta `notebooks/` na ordem sugerida.

## O que há em cada capítulo (mapeado à ementa)

1. **Introdução à IA e ML** – Conceitos básicos; tipos de aprendizado (supervisionado, não supervisionado e por reforço); exemplos práticos.
2. **Fundamentos de Python & Scikit‑learn** – Estruturas de dados, `pip/venv`; instalação e _setup_ do Scikit‑learn.
3. **Pré‑processamento de dados** – Limpeza, _encoding_, normalização/padronização, divisão _train/test_.
4. **Supervisionado (Classificação)** – _Workflows_ com `LogisticRegression`, `KNN`, `DecisionTree` (datasets: `load_iris`, `load_wine`).
5. **Supervisionado (Regressão)** – `LinearRegression`, `RandomForestRegressor` (dataset: `load_diabetes`).
6. **Avaliação de modelos** – _Holdout_, _cross‑validation_, métricas (`accuracy`, `precision/recall`, `f1`, `ROC-AUC`, `MAE/MSE/RMSE`).
7. **Não supervisionado** – `KMeans`, `DBSCAN` (noções), **PCA** para redução de dimensionalidade e visualização.
8. **Aprendizado por Reforço (introdução)** – Conceitos, terminologia (agente/ambiente/recompensa), exemplos conceituais.
9. **Ética & Futuro da IA** – Uso responsável, viés, transparência, segurança de dados e tendências.

## Dados
Usamos **datasets embutidos** do Scikit‑learn (ex.: Iris, Wine, Breast Cancer, Diabetes). Não é necessário baixar arquivos.

## Referências
- Edital 2025.1 – Cursos de Qualificação Profissional – Fundação Cecierj (Eixo: Gestão de dados e TI – _Inteligência artificial e aprendizado de máquina para soluções práticas_). PDF: https://extensao.cecierj.edu.br/cursos/qualificacao-profissional/2025-1/Edital-QP-2025-1.pdf

---

## Licença
MIT – veja `LICENSE`.
