# BERTGradGraph

A web-based tool for visualizing attention and gradient sensitivity in transformer models.
URL: [yifan0sun.github.io/BERTGradGraph/](https://yifan0sun.github.io/BERTGradGraph/)

---

## 🌟 Overview

**BERTGradGraph** provides an intuitive interface for exploring how transformer-based language models (like BERT, RoBERTa, and DistilBERT) process natural language. By visualizing both attention and input gradient flow across layers, the tool helps users understand which tokens a model attends to — and which ones truly influence its behavior.

This tool supports three standard NLP tasks:
- **MLM** (Masked Language Modeling)
- **SST** (Sentiment Classification)
- **MNLI** (Premise-Hypothesis Inference)

Users can interactively choose a model and task, mask a word (in MLM), and view the resulting attention maps and gradient norms per token.

---

## 📊 Visualizations

### 1. **Attention Map**

A directed bipartite graph shows how much attention each token gives to others in a selected transformer layer. Node colors and edge thickness indicate attention strength.

### 2. **Gradient Norm Map**

Using input embedding gradients, the tool computes how much each token influences the attention map — providing a saliency-like signal across the sequence.

### 3. **Top Predictions**

The tool displays model predictions for the selected task:
- **MLM**: shows the model's top predicted tokens for the masked position.
- **SST**: shows sentiment classification scores (e.g., positive vs. negative).
- **MNLI**: shows the model’s confidence scores for entailment, neutrality, or contradiction between the premise and hypothesis.

All predictions are visualized using an interactive bar chart.

---

## 📎 Technical Notes

- Backend is written in **FastAPI** and deployed on **HuggingFace**.
- Frontend is built with **React + Vite** and hosted via **GitHub Pages** (`docs/` folder).
- Uses **Hugging Face Transformers** for all model inference.
- Visualizations are built using **Plotly.js**.

✅ Backend is stateless and re-initializes models per request, with caching to reduce load time.

✅ No dependencies required to use the tool — it runs entirely in-browser and via public API endpoints.

---

## 🧪 Prototype Disclaimer

This tool is an early prototype.

- Some UI features and visual scaling choices may evolve.
- Feedback and suggestions for improvement are welcome!

---

## 📬 Contact

Built by **Yifan Sun**  
Email: `yifan dot sun at stonybrook dot edu`  
Website: [optimalvisualizer.com](http://optimalvisualizer.com)

---

