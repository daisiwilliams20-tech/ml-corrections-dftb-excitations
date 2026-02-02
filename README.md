# ML Corrections for DFTB Excitation Energies

This repository contains datasets, feature representations, and scripts for **machine-learning (Δ-learning) corrections of DFTB-calculated electronic excitation energies** toward higher-level quantum chemistry reference data (CC2).

The project focuses on improving the accuracy of **TD-DFTB excitation energies** by learning the systematic error between DFTB and CC2 using **Coulomb Matrix–based molecular descriptors** and supervised ML models.

---

## 🔬 Scientific Motivation

Density Functional Tight Binding (DFTB) methods provide fast excited-state calculations but often suffer from systematic deviations when compared to higher-level methods such as **CC2**.

This project applies **Δ-learning**, where a machine-learning model is trained to predict:

\[
\Delta E = E_{\mathrm{CC2}} - E_{\mathrm{DFTB}}
\]

The learned correction can then be added to new DFTB predictions to obtain **near-CC2 accuracy at DFTB cost**.

---

## 📁 Repository Structure

