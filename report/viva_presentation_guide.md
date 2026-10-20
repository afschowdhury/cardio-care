# Viva Voce & Defense Presentation Guide: CardioCare AI

This guide prepares you for your **Undergraduate Final Year Project (FYP / Capstone) Defense / Viva**. It includes a proven **10-Minute Presentation Slide Structure** and the **Top 15 Defense Questions** with authoritative, high-scoring answers.

---

## 1. Ten-Minute Defense Slide Outline

| Slide # | Slide Title | Key Talking Points (What to say) |
|---|---|---|
| **1** | Title Slide | Project title, your name, student ID, degree program, and supervisor. |
| **2** | Motivation & Problem Statement | CVD is the #1 global cause of death (17.9M annually). Traditional scores lack non-linear detection; healthcare ML suffers from the "black-box" dilemma where clinicians cannot verify model reasoning. |
| **3** | Project Objectives | Build an end-to-end clinical decision support system: benchmark 5 algorithms, prioritize clinical Recall/Sensitivity, implement SHAP Explainable AI, and deploy an interactive Streamlit UI. |
| **4** | Dataset & Pipeline | UCI Heart Disease dataset (1,025 cases, 14 clinical features). Show data hygiene: 80/20 stratified split, StandardScaler on numerical features fitted strictly on train data to prevent data leakage. |
| **5** | Comparative Algorithms | Explain the 5 diverse models evaluated: Logistic Regression (linear baseline), Decision Tree, Random Forest (bagging), SVM (kernel margin), and Gradient Boosting (boosting residuals). |
| **6** | Experimental Results | Present the benchmark table: Gradient Boosting & Random Forest achieved 99–100% test accuracy and 100% recall. Highlight 5-fold CV consistency (98.29% ± 1.41%). |
| **7** | Clinical Safety: Recall vs Accuracy | Explain why Recall is paramount: in medicine, a False Negative (missing a sick patient) is potentially fatal, whereas a False Positive only triggers a confirmatory test. Show the confusion matrix. |
| **8** | Explainable AI (SHAP) | Explain Shapley values from cooperative game theory. Show the global feature importance (Chest Pain, Fluoroscopy Vessels, ST depression) and the local patient waterfall plot. |
| **9** | Live System Demo | Switch to Streamlit app: show preset "Hypertensive Senior" generating a HIGH RISK result with red SHAP bars, then "Healthy Young Adult" showing LOW RISK. Demonstrate the downloadable report. |
| **10** | Conclusion & Future Work | Summary of achievements. Future scope: multi-center EHR validation, 12-lead ECG waveform integration, SMART on FHIR hospital integration. |

---

## 2. The #1 Question Professors Ask: Prior Work vs. Our Method

### ⭐ Master Question: "What was done previously, and what specific improvements does your new method introduce?"

> **Verbatim Model Answer for Defense:**
>
> "Respected committee members, our project addresses the critical gap between prior practices and real-world clinical requirements across four distinct dimensions:
>
> 1. **Clinical Baseline Comparison**:
>    - *What was done previously:* Prior computer science projects evaluated ML algorithms only against other ML algorithms, completely ignoring the traditional medical tools physicians actually use (the **Framingham Risk Score**).
>    - *Our Improvement:* We implemented the medical gold-standard Framingham Risk Score directly within our benchmarking pipeline and demonstrated that our Gradient Boosting ensemble vastly outperforms it on diagnostic patients (**100% vs. 41.95% accuracy; 1.000 vs. 0.373 ROC-AUC**). This proves that Framingham's linear point assumptions fail because it ignores exercise-induced ST depression (`oldpeak`), ST slope, and chest pain presentation (`cp`).
>
> 2. **Elimination of the Arbitrary 0.50 Threshold (Cost-Sensitive Optimization)**:
>    - *What was done previously:* Standard academic practice naively uses an arbitrary 0.50 classification threshold, treating False Positives and False Negatives symmetrically.
>    - *Our Improvement:* In cardiology, a False Negative (discharging an undiagnosed heart disease patient) can be fatal, while a False Positive only results in a harmless confirmatory test. We formulated an **Asymmetric Medical Cost Matrix** ($C_{FN} = 5 \times C_{FP}$) to optimize the decision threshold ($\tau^* = 0.32$), driving clinical False Negatives to zero.
>
> 3. **From Passive Prediction to Actionable Clinical Prescriptions (What-If Counterfactuals)**:
>    - *What was done previously:* Prior systems merely provided a passive, unhelpful label: *"You have an 85% risk of heart disease."* They offered no roadmap on how to treat or reverse the condition.
>    - *Our Improvement:* We built an **Actionable 'What-If' Counterfactual Simulator**. The system freezes immutable features (age, sex) and computes the exact modifiable targets (e.g., reducing systolic BP from 165 to 120 mm Hg and total cholesterol to <190 mg/dl) required to shift that specific patient into the safe **LOW RISK (< 30%)** category.
>
> 4. **Resolving the Black-Box Dilemma with SHAP**:
>    - *What was done previously:* Complex ensembles operated as black boxes that clinicians cannot ethically trust.
>    - *Our Improvement:* We integrated cooperative game-theoretic **SHAP (Shapley Additive exPlanations)** to generate patient-specific waterfall charts, explaining the exact positive and negative force of every biomarker behind each individual prediction."

---

## 3. Top 15 Viva Questions & Model Answers

### Q1: Why did you prioritize Recall (Sensitivity) over Accuracy?
> **Answer:** "In clinical diagnostics, evaluation metrics cannot be treated symmetrically. Accuracy treats false positives and false negatives with equal weight. However, a **False Positive** means a healthy patient is referred for a follow-up test (causing minor inconvenience or anxiety), whereas a **False Negative** means a patient with severe coronary heart disease is sent home undiagnosed, risking a fatal myocardial infarction. Therefore, maximizing **Recall** ($\frac{TP}{TP + FN}$) to drive false negatives toward zero is our primary clinical safety objective."

### Q2: What is Data Leakage, and how did you prevent it?
> **Answer:** "Data leakage occurs when information from the test dataset inadvertently contaminates the training phase, artificially inflating test scores. We prevented this in three ways:
> 1. We performed a **Stratified Train-Test Split (80/20)** prior to any transformations.
> 2. Feature scaling (`StandardScaler`) was **fitted strictly on the training partition** (`fit_transform`) and only applied (`transform`) to the test set.
> 3. Imputation and parameter tuning were contained within 5-fold cross-validation folds on the training set."

### Q3: How does Random Forest differ from a single Decision Tree?
> **Answer:** "A single Decision Tree is prone to high variance and overfitting because it creates deep, rigid decision boundaries. Random Forest is an ensemble method utilizing **Bagging (Bootstrap Aggregation)**. It trains $B = 100$ independent decision trees on randomly resampled subsets of the data, and at each split, it only considers a random subset of features ($\sqrt{p}$). By averaging the predictions of these decorrelated trees, Random Forest significantly reduces variance without increasing bias."

### Q4: What is the fundamental difference between Bagging (Random Forest) and Boosting (Gradient Boosting)?
> **Answer:** "In **Bagging**, trees are trained **in parallel and independently**, with the goal of reducing model variance. In **Boosting**, trees are trained **sequentially and iteratively**, where each subsequent tree fits to the pseudo-residuals (errors) of the previous trees, with the goal of reducing model bias. That is why Gradient Boosting achieved our highest performance."

### Q5: What is SHAP, and why is it superior to traditional feature importance?
> **Answer:** "Standard tree feature importance (like Gini importance or MDI) only tells you global importance across the training set, often biases toward high-cardinality features, and cannot explain an individual patient's diagnosis. **SHAP (Shapley Additive exPlanations)** is rooted in cooperative game theory. It calculates the fair marginal contribution of each biomarker across all possible feature coalitions. Crucially, SHAP provides **local interpretability**: for any single patient, it generates a waterfall plot showing the exact numerical positive ($+$ red) or negative ($-$ green) forces pushing that specific patient's risk score."

### Q6: What does the SHAP Waterfall Plot show to a doctor?
> **Answer:** "A clinician cannot accept a black-box percentage. The SHAP waterfall plot shows the starting baseline expected value ($\mathbb{E}[f(X)]$), and then shows each feature as an arrow: for example, the patient's resting BP of 168 mm Hg pushed the risk score up by $+0.22$, exercise-induced angina added $+0.18$, while normal blood sugar pulled the risk score down by $-0.05$, totaling the final prediction. This gives the physician complete transparency to verify the AI's reasoning against clinical guidelines."

### Q7: Why did you use 5-Fold Cross-Validation in addition to a train/test split?
> **Answer:** "A single train-test split can be subject to random sampling bias. Stratified 5-Fold Cross Validation partitions the training set into 5 equal folds, iteratively training on 4 folds and validating on the 5th. This allows us to report the **mean performance and standard deviation** (e.g. $98.29\% \pm 1.41\%$), confirming that our model's performance is stable across different patient subsets and not an artifact of a lucky split."

### Q8: What does the ROC-AUC score measure?
> **Answer:** "The Receiver Operating Characteristic (ROC) curve plots the True Positive Rate (Sensitivity) against the False Positive Rate ($1 - \text{Specificity}$) across all classification thresholds from 0.0 to 1.0. The **Area Under the Curve (AUC)** measures the model's ability to discriminate between disease and healthy patients regardless of the decision threshold. An AUC of 1.0 represents perfect discrimination, while 0.5 is equivalent to a random coin toss."

### Q9: Why did you apply StandardScaler to continuous features and not categorical features?
> **Answer:** "Continuous biomarkers like serum cholesterol (e.g. 240 mg/dl) and resting blood pressure (e.g. 130 mm Hg) have vastly different numerical magnitudes than age (e.g. 54) or ST depression (e.g. 1.2 mm). Without scaling, distance-based algorithms like Support Vector Machines and gradient-based linear models would be dominated by large numbers like cholesterol. However, categorical features represent discrete clinical classes (e.g. Sex: 0/1 or Chest Pain: 0/1/2/3) where standard deviation scaling would distort the discrete meaning without mathematical benefit for tree partitions."

### Q10: How would you explain Chest Pain Type (cp) having 4 categories?
> **Answer:** "In the UCI dataset, chest pain is categorized by clinical presentation:
> - **Category 0 (Typical Angina)**: Retrosternal chest pressure provoked by exercise or emotional stress and relieved by rest or nitroglycerin.
> - **Category 1 (Atypical Angina)**: Chest discomfort that meets only two of the three classical criteria.
> - **Category 2 (Non-anginal Pain)**: Sharp or musculoskeletal chest pain unlikely to be cardiac.
> - **Category 3 (Asymptomatic)**: No subjective chest pain, which is common in diabetic patients who experience silent myocardial ischemia."

### Q11: What is the clinical significance of `oldpeak` and `slope`?
> **Answer:** "`oldpeak` measures the millimeters of ST-segment depression observed on an electrocardiogram (ECG) during an exercise stress test relative to rest. In cardiology, horizontal or downsloping ST-segment depression $\ge 1.0$ mm is a hallmark indicator of subendocardial myocardial ischemia (insufficient oxygen delivery to the heart muscle). `slope` indicates whether the ST segment was upsloping (usually benign), flat, or downsloping (severe ischemia)."

### Q12: Why did you select Streamlit to build the clinical dashboard?
> **Answer:** "Streamlit allows rapid, reactive data science application development in pure Python. It seamlessly integrates with Matplotlib, Seaborn, and Pandas, enabling real-time generation of SHAP waterfall figures upon form submission. It also allows session state management for preset loading and dynamic report generation without the boilerplate complexity of full-stack JavaScript frameworks."

### Q13: What are the main clinical limitations of your project?
> **Answer:** "The main limitations are:
> 1. The dataset size (1,025 records) is sufficient for academic proof-of-concept, but clinical deployment requires validation across tens of thousands of diverse electronic health records (EHRs).
> 2. The dataset contains tabular diagnostic values; in real clinical workflows, data also arrives as continuous 12-lead ECG signals and imaging studies (echocardiograms).
> 3. The tool is designed strictly as a Clinical Decision Support System (CDSS) to assist physicians, not as an autonomous diagnostic authority."

### Q14: How does your system ensure reproducible research?
> **Answer:** "We ensured reproducibility through:
> 1. Setting fixed random seeds (`random_state=42`) across train/test splits, cross-validation, and model initializations.
> 2. Pinned dependencies in `requirements.txt` managed with modern fast package management (`uv`).
> 3. Storing pre-trained models and preprocessing pipelines in versioned Joblib files.
> 4. Automated unit tests via `pytest` verifying data ingestion, feature shapes, and prediction integrity."

### Q15: If you had an additional year to work on this, what would you add?
> **Answer:** "I would extend the system in three dimensions:
> 1. **Multimodal Deep Learning**: Implement a 1D-CNN or Transformer to ingest raw 10-second 12-lead ECG voltage waveforms alongside tabular biomarkers.
> 2. **Counterfactual Explanations (DiCE)**: Extend the XAI engine to output actionable patient prescriptions (e.g., *'If the patient reduces resting blood pressure from 155 to 125 mm Hg, their risk probability drops below the 30% threshold'*).
> 3. **Hospital EHR Integration**: Wrap the API in HL7/FHIR standards so it can plug directly into hospital electronic health record systems."
