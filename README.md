Tweet Virality Classification & Explainability

Classifying the virality of tweets (by user group) and explaining why using multiple NLP pipelines: TF-IDF + FNN/XGBoost, BERT Encoder + LSTM, and NER + XGBoost, plus explorations in GANs, reinforcement-style optimization, and prompt engineering. Each approach is benchmarked against simple baselines, with visual explainability throughout. 
GitHub

🔍 Project Goals

Predict tweet virality with respect to user groups.

Provide visual explanations of features contributing to virality (e.g., tokens, entities).

Explore generation/optimization ideas (GANs, RL-style loops, prompt engineering) to improve tweets. 
GitHub

📦 Data

CSV files used across experiments (place in / or a data/ folder to suit your workflow):

File	Purpose
dataset.csv	Main training/evaluation dataset
dataset_score.csv	Scores/labels for evaluation
dataset_viralscore.csv	Supplemental viral-score features
dataset_viralscore_2.csv	Additional viral-score features

(See repo root for the exact filenames.) 
GitHub

🗂️ Repository Contents

EDA.ipynb — Exploratory data analysis

EDACont_WordFeatureImportance.ipynb — Extended EDA + word feature importance

TFIDF_LSTM_LIME.ipynb — TF-IDF with LSTM + LIME explainability (file appears as ITIDF_LSTM_LIME.ipynb in repo)

NER and Explainable AI.ipynb — NER + XGBoost with explainable AI (typo in repo filename: “Eplainable”)

generator-discriminator-final.ipynb — GAN exploration (generator/discriminator)

prompt-engineering final.ipynb — Prompt engineering experiments

sentiment_analysis_results.csv — Sentiment results

feature_impact.csv — Feature impact summaries

Refer to the repo file list for exact names/spacing. 
GitHub

⚙️ Quickstart
# (1) Create & activate a virtual environment (recommended)
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (Powershell):
.venv\Scripts\Activate.ps1

# (2) Install core dependencies
pip install numpy pandas scikit-learn xgboost torch transformers jupyterlab lime spacy matplotlib

# (3) (Optional) Download spaCy model if NER pipeline needs it
python -m spacy download en_core_web_sm


Run notebooks in this order (suggested):

EDA.ipynb → schema & distribution checks

TFIDF_LSTM_LIME.ipynb and NER and Explainable AI.ipynb → training + explainability

generator-discriminator-final.ipynb, prompt-engineering final.ipynb → augmentation/optimization

Artifacts (plots/CSVs) are saved alongside notebooks or to the repo root (per notebook settings). 
GitHub

🧪 Methods (at a glance)

Classical + DL baselines: TF-IDF → FNN/XGBoost; BERT encoder → LSTM

Information extraction: spaCy-style NER, features into XGBoost

Explainability: LIME + feature impact CSVs for transparency

Advanced ideas: GAN data augmentation; reinforcement-style optimization; prompt strategies

Each model is evaluated against baselines for incremental gains. 
GitHub

📊 Results

Comparative metrics across TF-IDF/FNN/XGBoost, BERT-LSTM, and NER-XGBoost

Local & global explanations (LIME plots, feature impact tables)

Observations from prompt-engineering and GAN/RL-style experiments

Add concrete scores/figures here after final runs; notebooks currently contain detailed outputs. 
GitHub

🛣️ Roadmap / Future Work

Broader hyperparameter sweeps & CV

Domain adaptation, debiasing

Stronger NER models & sentence-level features

More robust RL-style optimization loops
