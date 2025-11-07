[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/hgNAtOO3)

# Mapping the Narrative Journey: A Structural Analysis of Podcast Conversations

## Abstract

This project explores how podcast conversations evolve over time by modeling their narrative journey. Rather than identifying static topics, we aim to capture the dynamic flow of discussion, that is, how speakers transition between ideas, themes, and emotional tones throughout an episode. The project builds on established natural language processing techniques, combining semantic segmentation, topic labeling, and graph-based modeling to map how podcast conversations unfold. Using NLP methods, we will segment transcripts into coherent topical units, assign interpretable topic labels, and represent their sequence as a directed graph. Visualizing this structure will reveal patterns of conversational movement, such as recurrent loops, digressions, or shifts in focus. Our analysis seeks to uncover how different podcast genres (e.g., interviews vs. storytelling formats) construct their narrative arcs, contributing both methodological insights and intuitive visual tools for studying long-form dialogue.

## Contributions

The contribution of this project is primarily analytical. We aim to provide new insights into conversational structures, comparing the narrative journeys across different podcast genres such as interviews, discussions, and scripted narratives. The project will include a methodological framework and a set of visualizations for examining the narrative architecture of long-form dialogue. Our approach goes beyond static topic modeling by focusing on transitions, capturing how meaning develops across turns in conversation. Through the combination of segmentation, topic modeling, and narrative graphing, we expect to uncover new patterns in how conversations flow and evolve, contributing to both computational methods and interpretive understanding. The work will result in a reproducible NLP pipeline that can be applied to other forms of long-form dialogue, and it will offer visual tools that make complex narrative dynamics intuitively understandable.

## Datasets

We will use the **SPoRC (Spotify Podcast Corpus)** dataset provided in the course. It contains a large collection of podcast transcripts across multiple genres and formats, making it suitable for studying topic transitions and conversational structures. No additional datasets are planned at this stage.

## Methods

Our analytical pipeline consists of four main stages: segmentation, topic modeling, graph construction, and visualization.

In the **segmentation phase**, podcast transcripts will be divided into coherent topical segments based on semantic similarity between adjacent sentences. Sentence embeddings will be obtained using Sentence-BERT (“all-MiniLM-L6-v2”), and cosine similarity between consecutive embeddings will be used to detect topical boundaries. Two segmentation strategies will be explored: fixed threshold segmentation, where a similarity cutoff determines when a new topic begins, and adaptive segmentation, which detects boundaries using local minima in smoothed similarity curves. This process generates a set of coherent segments for each episode that reflect natural shifts in the conversation.

In the **topic modeling phase**, each segment will be assigned a topic label using BERTopic, which clusters embeddings and generates interpretable topic names. We will also experiment with alternatives such as FASTopic, which has been shown to improve topic coherence and computational efficiency. If time allows, we may also explore using large language models (LLMs) to refine or evaluate topic labels for interpretability and alignment with human perception.

In the **graph construction phase**, we will represent the flow of conversation as a directed graph. Each node corresponds to a topic, and each edge represents a transition between topics within a given episode. Edge weights will indicate the frequency of transitions, allowing us to capture both dominant and peripheral topic flows. The graph will serve as a structured representation of the conversation’s narrative journey.

Finally, in the **visualization phase**, we will use NetworkX and Plotly to create interpretable topic-flow visualizations. These will include topic transition graphs and Sankey-style diagrams illustrating the progression and recurrence of topics. Graph metrics such as degree centrality and clustering will provide quantitative measures of narrative complexity. By comparing these patterns across podcast genres, we will identify stylistic and structural differences, for example, storytelling podcasts may follow more linear paths, while interview podcasts may display cyclical or branching topic flows.

## Proposed timeline

**Week 43**: Data exploration and initial segmentation experiments.

**Week 44**: Segmentation.

**Week 45**: Topic modeling, graph construction, visualization, and documentation.

By the P2 deadline, the segmentation and topic modeling components will be complete and documented in the main notebook. The following phase will focus on in-depth analysis, visualization, and interpretation of results.

## Organization within the team

**Internal milestones**

1. Adaptive segmentation and coherence evaluation completed by Milestone P2 (November 7).
2. Topic modeling and labeling completed.
3. Graph analysis and genre comparison completed.
4. Final project report and repository ready for Milestone P3 (December 19).

## Appendix

**Repository Organization**
* README.md - Project description and proposal.
* main.ipynb - Main analysis notebook containing the end-to-end pipeline.
* requirements.txt - Requirements text-file.
