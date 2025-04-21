
# Confidence Estimation for Error Detection in Text-to-SQL Systems

This repository contains the code, analysis, predictions, and extended version of the paper **"Confidence Estimation for Error Detection in Text-to-SQL Systems"**. The work focuses on training and evaluating T5 and LLama models for Text-to-SQL tasks, along with confidence estimation techniques for error detection.

---

## Table of Contents
1. [Extended Version of the Paper](#extended-version-of-the-paper)
2. [Data](#data)
3. [Code](#code)
4. [Evaluation](#evaluation)
5. [Citation](#citation)
6. [Contact](#contact)

---

## Extended Version of the Paper

The extended version of the paper, including appendices, is available on **arXiv**:  
[Confidence Estimation for Error Detection in Text-to-SQL Systems](https://arxiv.org/abs/2501.09527)

---

## Setup

To set up the environment, ensure you have **Python 3.10** installed. It is recommended to use **Miniconda** for managing the environment.

1. **Install Required Libraries**:  
   Install the dependencies listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

---


## Data

The dataset splits for training and evaluation are available from the following sources:

### Datasets
1. **Original PAUQ XSP**  
   - Repository: [PAUQ XSP](https://github.com/ai-spiderweb/pauq)  
   - Contains the database and table information for PAUQ XSP.

2. **Compositional PAUQ Template SSP and PAUQ Test Long SSP**  
   - Google Drive: [Compositional Splits](https://drive.google.com/drive/folders/12cBewVCrBObBb1qgEg1nXHoqq3hHTT7K?usp=sharing)  
   - Code for preparing compositional splits: [Splitting Strategies](https://github.com/runnerup96/splitting-strategies)  

3. **EHRSQL**  
   - Repository: [EHRSQL](https://github.com/glee4810/ehrsql-2024)  

---

## Code

The code for this project is divided into three parts, each corresponding to a specific dataset and model:

1. **Training and Inference of T5 Models for PAUQ**  
   - Repository: [T5-fine-tuning-for-text-to-SQL](https://github.com/runnerup96/T5-fine-tuning-for-text-to-SQL)  
   - Follow the setup and usage instructions in the repository.

2. **Training and Inference of T5 Models for EHRSQL**  
   - Repository: [EHRSQL-text2sql-solution](https://github.com/runnerup96/EHRSQL-text2sql-solution)  
   - Follow the setup and usage instructions in the repository.

3. **Training and Inference of LLama Models**  
   - Repository: [Text-to-SQL-LLama](https://github.com/runnerup96/Text-to-SQL-LLama)  
   - Follow the setup and usage instructions in the repository.

4. **Analysis of Results**  
   - The analysis scripts and results are available in the source code of this repository as Jupyter Notebooks.

---

## Evaluation

The evaluation of Text-to-SQL systems is conducted using the following tools:

1. **SPIDER Test Suite**  
   - Used for execution match evaluation on the PAUQ dataset.  
   - Repository: [SPIDER Test Suite](https://github.com/taoyds/test-suite-sql-eval)  
   - Add the repository to your system path using:
     ```python
     import sys
     sys.path.append("path/to/test-suite-sql-eval")
     ```

2. **EHRSQL Evaluation**  
   - Used for evaluation on the EHRSQL dataset.  
   - The evaluation code is already copy-pasted into `support_functions.py`
   - Repository: [EHRSQL 2024](https://github.com/glee4810/ehrsql-2024)  

---

## Citation

If you use this work in your research, please cite the following paper:

```bibtex
@article{somov2025, 
title={Confidence Estimation for Error Detection in Text-to-SQL Systems}, 
volume={39}, 
url={https://arxiv.org/abs/2501.09527}, 
DOI={10.1609/aaai.v39i23.34699}, 
abstractNote={Text-to-SQL enables users to interact with databases through natural language, simplifying the retrieval and synthesis of information. Despite the success of large language models (LLMs) in converting natural language questions into SQL queries, their broader adoption is limited by two main challenges: achieving robust generalization across diverse queries and ensuring interpretative confidence in their predictions. To tackle these issues, our research investigates the integration of selective classifiers into Text-to-SQL systems. We analyse the trade-off between coverage and risk using entropy based confidence estimation with selective classifiers and assess its impact on the overall performance of Text-to-SQL models. Additionally, we explore the models’ initial calibration and improve it with calibration techniques for better model alignment between confidence and accuracy. Our experimental results show that encoder-decoder T5 is better calibrated than in-context-learning GPT 4 and decoder-only Llama 3, thus the designated external entropy-based selective classifier has better performance. The study also reveal that, in terms of error detection, selective classifier with a higher probability detects errors associated with irrelevant questions rather than incorrect query generations.}, 
number={23}, 
journal={Proceedings of the AAAI Conference on Artificial Intelligence},
author={Somov, Oleg and Tutubalina, Elena}, 
year={2025}, 
month={Apr.}, 
pages={25137-25145}
}
```

---

## Contact

For questions, feedback, or collaboration opportunities, feel free to reach out:

- **Telegram**: [@olg_smv](https://t.me/olg_smv)  
- **Email**: [somov.ol.dm@gmail.com](mailto:somov.ol.dm@gmail.com)  

---
