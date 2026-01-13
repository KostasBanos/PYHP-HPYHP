## POWER-LAW NON-PARAMETRIC TEMPORAL MODELS FOR CONTINUOUS-TIME DOCUMENT STREAMS (Master’s Thesis)

Implementation of Sequential Monte Carlo (SMC) inference for the Pitman–Yor Hawkes Process (PYHP), used to discover and track temporal clusters of news events by combining Hawkes process dynamics (time) with Dirichlet-multinomial modeling (text). This repository contains the core implementation, data preprocessing and analysis notebooks, and visualization utilities. This work was conducted at McMaster University (2026).

This codebase is an application of an extension of the Dirichlet–Hawkes Process (DHP) developed during my master’s research. The extension introduces Pitman–Yor discounting to the Hawkes intensities (reducing to DHP when d = 0) and supports online inference via SMC. A manuscript based on this research is under preparation and will include additional results such as simulation studies and generalization to more datasets. This paper is expected to be released in the next months.

### Thesis
POWER-LAW NON-PARAMETRIC TEMPORAL MODELS FOR CONTINUOUS-TIME DOCUMENT STREAMS  
McMaster University, 2026 — supervised by Distinguished University Professor Narayanaswamy Balakrishnan.

### Features
- SMC particle filtering for online clustering
- Pitman–Yor discounting over Hawkes intensities (reduces to DHP when d=0)
- RBF-basis triggering kernels and Bayesian parameter updates
- Text modeling via Dirichlet–multinomial likelihoods
- Cluster word-frequency export and circular word clouds

### Background (brief)
- Hawkes component: captures temporal self-/mutual-excitation via a mixture of RBF kernels.
- Text component: compares documents and clusters via a Dirichlet–multinomial likelihood.
- Pitman–Yor extension: applies discounting to existing-cluster intensities while boosting the mass for a new cluster, yielding richer power-law cluster size behavior; when d → 0, it recovers the Dirichlet–Hawkes Process.

### Repository structure
- `SMC_sampling.py`: Main PYHP + SMC algorithm; entry point to run experiments
- `utils.py`: Data structures, kernels, inference utilities, and visualization helpers
- `data/`: Example vocab maps and news JSON (see “Data format”)
- `results/`: Example outputs and wordclouds
- `*.ipynb`: Preprocessing, power-law analysis, and result analysis notebooks

### Installation
Requirements: Python ≥ 3.8

```bash
pip install -r requirements.txt
```

If you plan to run the preprocessing notebook with stemming/lemmatization, download the NLTK resources:

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### Dataset
- Download the news dataset from Kaggle: [All the News (Kaggle)](https://www.kaggle.com/datasets/cymerunaslam/allthenews).
- After downloading, place the files in the `data/` folder of this repository.
- If you download CSVs (`articles1.csv`, `articles2.csv`, `articles3.csv`), run `data_preprocessing.py` to produce `data/all_the_news_2017.json` which the scripts expect.

### Data format
The main script expects `data/all_the_news_2017.json` to be a JSON list of items, each as:

```json
[doc_id, unix_timestamp_seconds, [word_id, count], total_word_count]
```

Notes:
- Timestamps are internally converted to hours.
- `word_id` is an integer index into the vocabulary.
- Optional: `data/id2word.json` and `data/word2id.json` can be used with the analysis utilities to map IDs ↔ tokens.

### Quick start
Run the SMC inference over the provided JSON:

```bash
python SMC_sampling.py
```

Outputs:
- `results/particles.pkl`: Final particle set (cluster assignments, weights, parameters)
- Word-frequency summaries and word clouds (see Result analysis below)

### Result analysis and visualization
Use the notebooks for end-to-end analysis:
- `data_preprocessing.ipynb`: Build/clean vocabulary and datasets.
- `power_law_of_PYHP_over_DHP.ipynb`: Analyze PYHP vs DHP (d = 0) behavior.
- `result_analysis.ipynb`: Inspect clusters and render word clouds.

You can also programmatically extract top words per cluster:

```python
from utils import ClusterWordAnalyzer
import json, pickle

# Load a particle (e.g., from results/particles.pkl)
with open('results/particles.pkl', 'rb') as f:
    particles = pickle.load(f)
particle = particles[0]  # pick a representative particle

# Load id2word mapping
with open('data/id2word.json', 'r') as f:
    id2word = json.load(f)

analyzer = ClusterWordAnalyzer(particle, id2word)
word_freq = analyzer.get_word_freq(cluster_idx=0, top_n=200, min_count=1.0)
analyzer.save_word_freq(word_freq, 'results/cluster_0_wordfreq.json', 'results/cluster_0_wordfreq.csv')
analyzer.make_wordcloud(cluster_idx=0, out_path='results/cluster_0_wordcloud_circular.png', circular=True)
```

### Notes on performance and reproducibility
- SMC involves stochastic sampling; set seeds if you need partially reproducible behavior (some randomness is inherent in resampling).
- Runtime depends on number of particles, vocabulary size, and document count. Start with fewer particles to validate the pipeline, then scale up.

### Reproducibility tips
- Set NumPy random seeds where appropriate if you need deterministic runs.
- Particle filtering and multinomial sampling are stochastic by design.

### Research status
- This repository demonstrates the application of a Pitman–Yor extension to the Dirichlet–Hawkes Process within an online SMC framework.
- A manuscript with expanded results (simulation studies, ablations, and additional datasets) is in preparation and expected to be released in the coming months.
- Alongside the paper, a general-purpose software package will be released for broader use of these models (Pitman–Yor Hawkes Process and its hierarchical extension).

### How to cite
If you use this code, please cite the master’s thesis:

Banos.K (2026). “POWER-LAW NON-PARAMETRIC TEMPORAL MODELS FOR CONTINUOUS-TIME DOCUMENT STREAMS.” Master’s Thesis, McMaster University. Supervised by Distinguished University Professor Narayanaswamy Balakrishnan.

See `CITATION.cff` for structured metadata once the thesis is finalized.

### Acknowledgements
I am deeply grateful to my supervisor, Distinguished University Professor Narayanaswamy Balakrishnan, for his guidance and support. I also thank the Department of Mathematics & Statistics at McMaster University for providing a stimulating research environment, and acknowledge the authors and maintainers of the open-source libraries and datasets used in this project.

### License
Released under the MIT License (see `LICENSE`).


