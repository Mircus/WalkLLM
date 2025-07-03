# ♟ WalkGPT

> Bridging Knowledge Graphs and Large Language Models through stochastic semantic exploration.

📖 **Related article**: [WalkGPT: Random Walks Meet Language Models](https://medium.com/p/45ed4e69d166)

---

## 🤖 What is WalkGPT?

**WalkGPT** is an experimental framework that connects **Knowledge Graphs (KGs)** and **LLMs** by using **random walks** to drive context-aware prompting and semantic exploration.

Each step in the walk guides the language model to generate or query new information, effectively simulating a conversational traversal of a structured space.

---

## 🧠 Key Features

* 🔁 Perform random walks over any RDF-style or NetworkX-style graph
* 💬 Use walk paths to construct evolving prompts for LLMs
* 🧩 Mix deterministic queries with generative text
* 🕸 Semantic drift control via walk temperature and history
* ⚙️ Pluggable backends for LLM inference (OpenAI, Hugging Face, etc.)

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/Mircus/WalkGPT.git
cd WalkGPT
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run a walk + prompt loop

```bash
python walkgpt.py --config config.json
```

---

## ⚙️ How It Works

1. A **graph** is loaded (e.g. social graph, ontology, KG)
2. A **random walk** selects a sequence of nodes/edges
3. A **prompt template** integrates this walk as context
4. An **LLM** generates or completes based on the prompt
5. The output can be used to guide further walks or exploration

> This creates a loop between symbolic structure and linguistic creativity.

---

## 🧰 Tech Stack

* Python
* NetworkX / RDFLib (for graph walks)
* OpenAI / Hugging Face Transformers
* JSON-based config system

---

## 📄 License

MIT License — see `LICENSE` file.

---

## 🧭 Research Context

This work explores the intersection between **neuro-symbolic AI**, **semantic traversal**, and **generative reasoning**.
It's part of the **Holomathics** project.

If you use WalkGPT in your research or creative work, please cite or link the Medium article.

> *Built to walk ideas into words.*
