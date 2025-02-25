
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
@inproceedings{
anonymous2024confidence,
title={Confidence Estimation for Error Detection in Text-to-{SQL} Systems},
author={Oleg Somov, Elena Tutubalina},
booktitle={The 39th Annual AAAI Conference on Artificial Intelligence},
year={2025},
url={https://openreview.net/forum?id=W6g7kK9kQW}
}
```

---

## Contact

For questions, feedback, or collaboration opportunities, feel free to reach out:

- **Telegram**: [@olg_smv](https://t.me/olg_smv)  
- **Email**: [somov.ol.dm@gmail.com](mailto:somov.ol.dm@gmail.com)  

---