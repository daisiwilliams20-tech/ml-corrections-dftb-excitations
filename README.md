# ML Corrections for DFTB Excitation Energies  
## Δ-Learning Framework

---

## 📌 Overview

This repository implements a **machine-learning Δ-learning framework** to correct **DFTB-calculated electronic excitation energies** toward **CC2 reference accuracy** using **Coulomb Matrix molecular descriptors**.

By learning the systematic error between **TD-DFTB** and **CC2**, the workflow enables **near–ab-initio accuracy at tight-binding computational cost**, making it suitable for **large-scale excited-state screening**.

---

## 🔬 Scientific Background & Motivation

Density Functional Tight Binding (DFTB) methods offer significant computational efficiency for excited-state calculations but often exhibit **systematic deviations** from higher-level quantum-chemical methods such as **CC2**.

This project applies **Δ-learning**, where a supervised machine-learning model learns the correction:

\[
\Delta E = E_{\mathrm{CC2}} - E_{\mathrm{DFTB}}
\]

The learned correction is then added to new DFTB predictions to obtain CC2-level accuracy.

---

## 🎯 Objectives

- Quantify systematic DFTB excitation-energy errors  
- Learn DFTB → CC2 corrections using machine learning  
- Evaluate performance using MAE and error distributions  
- Enable scalable excited-state predictions for large molecular datasets  

---

## 📁 Repository Structure

ml-corrections-dftb-excitations/
│
├── data/
│ ├── cleaned_merged_with_CM_5dp.csv
│ ├── qm8_CC2_only.csv
│ └── cm_vectors_molml.csv
│
├── cm_vectors/
│ ├── cm.npy
│ ├── ids.npy
│ ├── max_atoms.npy
│ └── upper_only.npy
│
├── scripts/
│ ├── extract_cc2.py
│ ├── featuring.py
│ ├── make_cm_table.py
│ └── merge_dftb_qm8_simple.py
│
├── notebooks/
│ └── DFTB_to_CC2_Delta_Learning_with_Coulomb_Matrix.ipynb
│
└── README.md

---

## 🚀 How to Run the Project

### ⚠️ Required Files (Must Be Used Together)

The main workflow **requires both files below**:

- `notebooks/DFTB_to_CC2_Delta_Learning_with_Coulomb_Matrix.ipynb`
- `data/cleaned_merged_with_CM_5dp.csv`

The notebook **will not run correctly without** the dataset.

---

### Step 1: Verify dataset availability

Ensure the following file exists:


This file contains:
- DFTB excitation energies  
- CC2 reference excitation energies  
- Coulomb Matrix features (5-decimal precision)  

---

### Step 2: Run the Δ-learning notebook

Open and execute:


Inside the notebook:
- The dataset is loaded  
- ΔE targets are constructed  
- ML models are trained  
- Performance metrics (e.g., MAE) are evaluated  

⚠️ **Important:**  
The notebook and dataset must remain in the same relative directory structure, or file paths must be updated.

---

## 📊 Outputs

- Predicted Δ-corrections  
- ML-corrected excitation energies  
- MAE comparison (DFTB vs corrected vs CC2)  
- Error distributions and parity plots  

---

## 🛠 Requirements

- Python ≥ 3.8  
- NumPy  
- Pandas  
- scikit-learn  
- molml  
- matplotlib / seaborn  
- Jupyter Notebook  

---

## 📚 References

1. Ramakrishnan et al., *Machine Learning of Quantum Chemical Properties*, **Phys. Rev. Lett. (2015)**  
2. QM8 Dataset — Quantum Machine Learning Benchmark  
3. Elstner et al., *Self-Consistent-Charge Density-Functional Tight-Binding Method*  

---

## 👤 Author

**Daisi Williams**  
Computational Physics • Machine Learning for Quantum Chemistry  
GitHub: https://github.com/daisiwilliams20-tech  

---

## 📄 License

Released for academic and research use.  
Please cite appropriately if used in publications.
