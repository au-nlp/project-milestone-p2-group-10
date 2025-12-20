[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/hgNAtOO3)

# Mapping the Narrative Journey: A Structural Analysis of Podcast Conversations

## Abstract

This project explores how podcast conversations evolve over time by modeling their narrative journey. Rather than identifying static topics, we aim to capture the dynamic flow of discussion, that is, how speakers transition between ideas, and themes throughout an episode. The project builds on established natural language processing techniques, combining semantic segmentation, topic labeling, and graph-based modeling to map how podcast conversations unfold. Using NLP methods, we will segment transcripts into coherent topical units, assign interpretable topic labels, and represent their sequence as a directed graph. Visualizing this structure will reveal patterns of conversational movement, such as recurrent loops, digressions, or shifts in focus. Our analysis seeks to uncover how different podcast genres (e.g., interviews vs. storytelling formats) construct their narrative arcs, contributing both methodological insights and intuitive visual tools for studying the narrative architecture, semantic velocity, and temporal drift of long-form dialogue.

## Contributions

The contribution of this project is primarily analytical. We aim to provide new insights into conversational structures, comparing the narrative journeys across different podcast genres such as interviews, discussions, and scripted narratives. The project will include a methodological framework and a set of visualizations for examining the narrative architecture of long-form dialogue. Our approach goes beyond static topic modeling by focusing on transitions, capturing how meaning develops across turns in conversation. Through the combination of segmentation, topic modeling, and narrative graphing, we expect to uncover new patterns in how conversations flow and evolve, contributing to both computational methods and interpretive understanding. The work will result in a reproducible NLP pipeline that can be applied to other forms of long-form dialogue, and it will offer visual tools that make complex narrative dynamics intuitively understandable.

## Datasets

We use the **SPoRC (Spotify Podcast Corpus)** dataset provided in the course. It contains a large collection of podcast transcripts across multiple genres and formats, making it suitable for studying topic transitions and conversational structures. The dataset includes thousands of episodes with diverse lengths and metadata such as podcast title, genre, and episode duration. No additional datasets are used.

## Methods

Our analytical pipeline consists of six main stages: (1) semantic segmentation, (2) embedding-based
topic labeling, (3) topic-transition graph modeling, (4) visualization of the narrative flow, (5) semantic
velocity or pacing, and (6) temporal drift

In the **segmentation phase**, podcast transcripts will be divided into coherent topical segments based on similarity between two *blocks* of sentences (e.g., $`k=5`$ sentences *before* a gap vs. $`k=5`$ sentences *after*). It will find gaps where the "topic" of the preceding block is most different from the "topic" of the following block, then finds the deepest "valleys" in *that* graph. This is what we call `neuralTextTiling` algorithm. For the purpose of segmenting text into sentences or other semantic units, we used a transformer based model called SaT(Segment Any Text) from wtpsplit library.  Sentence embeddings are obtained using Sentence-BERT (“all-mpnet-base-v2”), and cosine similarity between consecutive embeddings will be used to detect topical boundaries. Three segmentation strategies will be explored: fixed threshold segmentation, where a similarity cutoff determines when a new topic begins, and adaptive segmentation, which detects boundaries using local minima in smoothed similarity curves. And finally our `neuralTextTiling` method which we decided to use for the project. 

In the **topic modeling phase**, each segment will be assigned a topic label using BERTopic, which clusters embeddings and generates interpretable topic names. We will also experiment with alternatives such as FASTopic, which has been shown to improve topic coherence and computational efficiency. If time allows, we may also explore using large language models (LLMs) to refine or evaluate topic labels for interpretability and alignment with human perception.

In the **graph construction phase**, we will represent the flow of conversation as a directed graph using NetworkX (or similar). Each node corresponds to a topic, and each edge represents a transition between topics within a given episode. Edge weights will indicate the frequency of transitions, allowing us to capture both dominant and peripheral topic flows. The graph will serve as a structured representation of the conversation’s narrative journey.

In the **visualization phase**, we will use Plotly to create interpretable topic-flow visualizations. These will include topic transition graphs and Sankey-style diagrams illustrating the progression and recurrence of topics. Graph metrics such as degree centrality and clustering will provide quantitative measures of narrative complexity. By comparing these patterns across podcast genres, we will identify stylistic and structural differences, for example, storytelling podcasts may follow more linear paths, while interview podcasts may display cyclical or branching topic flows.

And finally, the last two phases contribute directly to the statistical analysis of narrative dynamics.

In **semantic velocity and pacing analysis**, we measure the speed and rhythm of information within a conversation or podcast. Instead of just looking at what is being said, it looks at how fast the topics are changing.

In **temporal drift analysis**, we track how a genre evolves over several years by identifying its "center of gravity" for each year. By measuring the distance between these yearly centers, researchers can quantify how much a category has reinvented itself over time. Using visual mapping, they can identify "pivot years"—like the 2020 pandemic—where the focus of the content shifted drastically and stayed different.


### Feasibility and Data Handling

The SPoRC dataset includes thousands of podcast episodes, each with transcripts ranging from a few hundred to several thousand tokens. Preliminary inspection confirms that the data can be processed efficiently using standard hardware. Preprocessing steps, such as tokenization, sentence segmentation, and embedding generation, will be implemented with <code>wtpsplit</code> and <code>sentence-transformers</code>. Missing or incomplete transcripts will be excluded, and intermediate representations (e.g., sentence embeddings) will be cached to optimize runtime and ensure reproducibility. These design choices make the project computationally feasible given the dataset’s size and structure.

### Alternatives Considered

We initially considered fine-tuning a transformer-based model (e.g., DistilBERT) for supervised topic classification. However, the lack of labeled data and the increased computational cost make this approach impractical for the current scope. An unsupervised strategy, combining segmentation and topic modeling, offers greater scalability and generalization across podcast genres while remaining interpretable.

## Updates Since Milestone P2

- Added two new analysis methods to our pipeline to analyze the narrative dynamics of long-form conversation. 

## Proposed timeline

**Week 43**: Data exploration and initial segmentation experiments.

**Week 44**: Segmentation refinement and topic modeling setup.

**Week 45**: Graph construction, visualization, and documentation.

By the P2 deadline, the segmentation will be complete and documented in the main notebook. The following phase will focus on topic labeling, in-depth analysis, visualization, and interpretation of results.

## Organization within the team

**Internal milestones**

1. Document segmentation will be complete by Milestone P2 (November 7).
2. Topic modeling and labeling by November 20
3. Graph analysis and genre comparison by November 27
4. Final project report and repository ready for Milestone P3 (December 19).

**Organization within the team**

**Sadik**: Implemented the analytical pipeline, including segmentation, topic modeling, graph construction, visualization, and statistical analysis. Contributed to writing the report.

**Enok**: Contributed to topic modeling experiments, model refinement, coding tasks, and writing the report.

**Naja**: Conducted initial data exploration, segmentation and visualization experiments, documentation, and writing the report.

## Appendix

**Question for the TA**

**Repository Organization**
* <code>README.md</code> - Project description, contributions, and updates.
* <code>main.ipynb</code> - Main analysis notebook containing the end-to-end pipeline.
* <code>requirements.txt</code> - Python dependencies.

## Script Descriptions

This project is organized into three main phases: **Data Processing**, **Modeling**, and **Analysis & Visualization**. Below is a detailed breakdown of each script's role in the pipeline.

### 1. Data Processing & Segmentation
* **`harvester.py`** The main ETL driver for the project. It streams a large JSONL dataset, cleans the transcripts, and applies the hybrid segmentation algorithm to chunk episodes into coherent topics. It saves the output (embeddings, text snippets, and metadata) as `.npy` files for efficient loading.
* **`Hybrid_TextTiling_Segmenter.py`** Implements the custom segmentation logic. It combines `wtpsplit` for robust sentence boundary detection with `SentenceTransformers` for semantic similarity, calculating "depth scores" to identify where topics shift within an episode.
* **`preprocess_episode.py`** A cleaning utility that normalizes raw transcripts. It removes metadata tags (e.g., `[MUSIC]`, `(laughing)`), cleans up whitespace, and filters out non-speech artifacts before segmentation.
* **`coherence_eval.py`** A quality control metric used to evaluate segmentation performance. It calculates a score based on **intra-segment similarity** (cohesion within a topic) minus **inter-segment similarity** (distinctiveness between topics), with a penalty for over-fragmentation.
* **`group_by_cat.py`** A file management script that organizes processed `.npy` files into categorical folders (e.g., `grouped_health`, `grouped_education`) based on keywords found in the source dataset.

### 2. Topic Modeling & Graph Construction
* **`marge_data.py`** *(Note: typo in filename, intended as `merge_data.py`)* Aggregates the individual `.npy` segment files into a single global dataset. It loads vectors, text snippets, and timestamps to prepare the data for the global BERTopic model.
* **`topic_model.py`** Configures and trains the **BERTopic** pipeline. It utilizes UMAP for dimensionality reduction, HDBSCAN for clustering, and a CountVectorizer to generate topic representations.
* **`custom_tokenizer.py`** A helper class utilizing `spacy` to improve topic labels. It extracts meaningful syntactic bigrams (e.g., "Adjective + Noun") rather than simple unigrams, resulting in more descriptive topic names.
* **`zero_shot.py`** Defines a custom representation model for BERTopic using an LLM (via OpenAI API or local server). It generates human-readable, zero-shot titles for topics based on their keywords.
* **`graph_model.py`** Constructs a **NetworkX Directed Graph** from the topic timeline. It models the narrative flow by creating edges between sequential topics (e.g., how often "Topic A" transitions to "Topic B") and filters out noise.
* **`cycle_breaker.py`** A graph refinement algorithm that converts the cyclic topic graph into a linear Directed Acyclic Graph (DAG). It iteratively detects cycles and breaks them by removing the weakest transition edge to reveal the dominant narrative flow.

### 3. Analysis, Diagnostics & Visualization
* **`semantic_velocity.py`** Analyzes the "pacing" of episodes. It calculates the semantic distance between consecutive segments to determine how fast the conversation moves, visualizing episodes on a "Speed vs. Stability" archetype map.
* **`temp_drift.py`** Performs **Temporal Drift Analysis**. It computes the centroid of all topics for each year and uses PCA to visualize how the center of conversation has shifted in semantic space over time.
* **`drift_diagnostics.py`** Provides the "why" behind temporal drift. It uses TF-IDF with custom stopwords to identify the specific keywords that were most distinctive for each year (e.g., identifying unique vocabulary drivers per era).
* **`sankey_vis.py`** Generates a **Sankey Diagram** using `plotly`. This visualizes the structural flow of topics, showing the weight and direction of transitions between different conversation states.
