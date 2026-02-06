# INSY 695 Final Project: Enterprise AI in the Global Coffee Supply Chain

## 📋 Project Overview
This project applies enterprise-grade Machine Learning to the **Coffee Industry**, focusing on optimizing production and supply chain efficiency. By leveraging the end-to-end data science lifecycle, we aim to provide actionable insights into coffee variety performance, regional production trends, and market dynamics using SOTA supervised learning and causal inference.

## 🛠 Enterprise Tech Stack
In accordance with the INSY 695 syllabus, our project utilizes:
* **Cloud Infrastructure:** Microsoft Azure & Databricks (Apache Spark)
* **Containerization:** Docker & Kubernetes for production deployment
* **Modeling & SOTA ML:** * Supervised Learning (XGBoost/LightGBM) for production forecasting
    * Deep Learning (TensorFlow/Keras) for quality classification
    * AutoML (H2O.ai) for baseline benchmarking
* **Explainability & Governance:** SHAP and LIME to interpret model drivers
* **Causal Analytics:** EconML to evaluate the impact of environmental/economic factors on yield

## 🏗 Project Architecture & Modules
* **`/data`**: Raw coffee production, climate, and pricing datasets.
* **`/notebooks`**: Comprehensive Databricks notebooks covering EDA, Feature Engineering (PCA/Dimensionality Reduction), and Model Training.
* **`/src`**: Modular Python code for preprocessing and testing (`pytest`, `black`).
* **`/docs`**: Analysis of coffee varieties (Arabica vs. Robusta) and regional infographics.



## 🚀 Reproducibility & Best Practices
* **Standardized Environment:** Fully reproducible via Docker containers.
* **Data Integrity:** All transformations are documented within the provided Databricks `.dbc` files.
* **Ethics & Transparency:** Implementation of SHAP values to ensure transparency in automated decision-making.

## 👥 Group Members
* Yuyang Chen
* Zhaihan Gong
* Yujia Sun
* Zihan Xu
* Ruihe Zhang

## ⚖️ Evaluation Framework
* **Final Presentation (15%):** Business impact and technical architecture.
* **Final Submission (30%):** Code quality, reproducibility, and enterprise-grade documentation.

---
*This project is submitted to the Desautels Faculty of Management in fulfillment of the requirements for INSY 695.*
