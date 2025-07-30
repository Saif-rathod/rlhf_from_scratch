#  Reinforcement Learning from Human Feedback (RLHF) from Scratch!

This repository demonstrates how to align a pretrained language model (GPT-2) using **Reinforcement Learning from Human Feedback (RLHF)**, implemented through a series of Jupyter notebooks.

---

##  Overview

**Reinforcement Learning from Human Feedback (RLHF)** is a technique for training large language models (LLMs) like GPT-2 or GPT-3 to better align with human preferences. Instead of using direct feedback from an environment, RLHF involves:

1. **Supervised Fine-Tuning (SFT)**  
2. **Reward Model Training**  
3. **Reinforcement Learning via Proximal Policy Optimization (PPO)**  

A **reward model** is trained to imitate human preferences (e.g., ranking model outputs), and then this model is used to guide a policy through reinforcement learning.

---

##  Example Use Case

Suppose you're building a chatbot:

1. Collect a dataset of question-answer pairs.
2. Have human annotators rank the quality of answers.
3. Apply the RLHF pipeline:
   - **SFT**: Fine-tune the model to generate appropriate answers.
   - **Reward Model**: Train a model to predict rankings from annotators.
   - **PPO**: Improve the fine-tuned model to generate higher-quality responses as judged by the reward model.

---

##  This Repository's Implementation

This project adapts RLHF to train GPT-2 to generate **positively sentimented sentences**, using the [`stanfordnlp/sst2`](https://huggingface.co/datasets/stanfordnlp/sst2) dataset — a collection of movie review sentences labeled as positive or negative.

###  Objective

Use RLHF to train GPT-2 to generate **only** positive sentiment text.

---

##  Notebooks Breakdown

### 1. `SFT.ipynb` – Supervised Fine-Tuning

- Fine-tune GPT-2 using SST-2 dataset (positive/negative labeled sentences).
- Objective: Train GPT-2 to generate realistic sentences similar to the dataset.
- **Output**: Saved `SFT` model.

---

### 2. `RM.ipynb` – Reward Model Training

- Add a reward (classification) head to GPT-2.
- Train the model to classify sentiment (positive vs. negative).
- **Output**: Trained `Reward Model (GPT-2 + reward head)`.

---

### 3. `RLHF.ipynb` – PPO Reinforcement Learning

- **Sampling**: Use the SFT model to generate sentences.
- **Scoring**: Use the reward model to assign sentiment scores.
- **Optimization**: Use PPO to improve the model toward generating **higher-reward (positive)** sentences.
- **Output**: Final RLHF-optimized GPT-2 model.

---

## 📈 Final Outcome

After completing all three stages, GPT-2 is optimized to generate sentences that consistently reflect **positive sentiment**, guided by the learned reward model and reinforcement learning.
