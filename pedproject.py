# ============================================================
# INDOFLOODS - Flood vs Severe Flood  |  Preprocessing + 6 ML models
# Models: KNN, SVM, Naive Bayes, Neural Network, Bagged Trees, Boosted Trees
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve, classification_report)
 
OUT = "model_outputs"                 # graphs/results is folder me save honge
os.makedirs(OUT, exist_ok=True)
 
# =====================================================================
#                  PART A : DATA PREPROCESSING
# =====================================================================
# ---------- STEP 1: Load ----------
df = pd.read_csv(r"C:\Users\ahmad\Downloads\ProjectData.csv")
 
# ---------- STEP 2: Data checking ----------
print("Shape (rows, columns):", df.shape)
print("\nPehli 5 rows:\n", df.head())
print("\nColumn types:\n", df.dtypes)
print("\nStatistics:\n", df.describe())
print("\nDuplicate rows:", df.duplicated().sum())
 
# ---------- STEP 3: Null check ----------
print("\nNull values per column:\n", df.isnull().sum())
print("\nNull %:\n", (df.isnull().mean() * 100).round(2))
 
# ---------- STEP 4: Duplicates hatao ----------
df = df.drop_duplicates()
 
# ---------- STEP 5: Null flag columns (fill baad me, split ke baad) ----------
for col in ["Peak Discharge Q (cumec)", "Flood Volume (cumec)"]:
    df[col + " was_null"] = df[col].isnull().astype(int)
df = df.drop(columns=["Peak Discharge Date"])               # sirf date, model ko zaroorat nahi
 
# ---------- STEP 6: Dates -> numbers ----------
df["Start Date"] = pd.to_datetime(df["Start Date"], format="%d-%m-%Y")
df["Year"] = df["Start Date"].dt.year
df["Month"] = df["Start Date"].dt.month
 
# ---------- STEP 7: Target text -> number ----------
df["Flood Type"] = df["Flood Type"].map({"Flood": 0, "Severe Flood": 1})   # 0=Flood, 1=Severe
 
# ---------- STEP 8: Unnecessary columns hatao ----------
df = df.drop(columns=["EventID", "Start Date", "End Date", "Peak FL Date"])
 
# ---------- STEP 9: Skewed data / outliers -> log transform ----------
for col in ["Peak Discharge Q (cumec)", "Flood Volume (cumec)", "Event Duration (days)",
            "Time to Peak (days)", "Recession Time (day)", "Num Peak FL"]:
    df[col] = np.log1p(df[col])
 
X = df.drop(columns=["Flood Type"])
y = df["Flood Type"]
 
# ---------- STEP 10: Train-Test split (80/20) ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
 
# ---------- STEP 11: Null fill (median of TRAIN data only) ----------
medians = X_train.median()
X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)
print("\nNull values after handling:", X_train.isnull().sum().sum() + X_test.isnull().sum().sum())
 
# ---------- STEP 12: Scaling (fit on TRAIN, apply on TEST) ----------
# FIX 2: pehle code me scaled data (X_scaled) use hi nahi hua tha, isliye SVM/NN/KNN fail hue
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)
 
# Clean dataset save
pd.concat([X_train.assign(**{"Flood Type": y_train}),
           X_test.assign(**{"Flood Type": y_test})]).sort_index().to_csv(
    os.path.join(OUT, "cleaned_data.csv"), index=False)
print("Train:", X_train.shape, " Test:", X_test.shape)
 
# =====================================================================
#                  PART B : MACHINE LEARNING MODELS
# =====================================================================
models = {
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "SVM": SVC(kernel="rbf", C=10, random_state=42),
    "Naive Bayes": GaussianNB(),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=600, random_state=42),
    "Bagged Trees": BaggingClassifier(n_estimators=100, random_state=42),
    "Boosted Trees": GradientBoostingClassifier(n_estimators=200, random_state=42),
}
 
def get_score(model, X):
    """Probability/score for ROC curve (SVM ke liye decision_function)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)
 
results, cms, rocs = [], {}, {}
pred_table = pd.DataFrame({"Actual": y_test.map({0: "Flood", 1: "Severe Flood"}).values})
 
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = get_score(model, X_test)
 
    results.append({
        "Model": name,
        "Train Accuracy": accuracy_score(y_train, model.predict(X_train)),
        "Test Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1-score": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, score),
        "Error Rate": 1 - accuracy_score(y_test, pred),
        "Wrong Predictions": int((pred != y_test).sum()),
    })
    cms[name] = confusion_matrix(y_test, pred)
    rocs[name] = roc_curve(y_test, score)
    pred_table[name + " Prediction"] = pd.Series(pred).map({0: "Flood", 1: "Severe Flood"}).values
 
    print("\n=====", name, "=====")
    print("Confusion Matrix:\n", cms[name])
    print(classification_report(y_test, pred, target_names=["Flood", "Severe Flood"], zero_division=0))
 
# ---------- Comparison ----------
res = pd.DataFrame(results).sort_values("Test Accuracy", ascending=False).reset_index(drop=True)
print("\n================ MODEL COMPARISON ================")
print(res.round(4).to_string(index=False))
print("\nBEST MODEL:", res.loc[0, "Model"], "| Accuracy =", round(res.loc[0, "Test Accuracy"] * 100, 2), "%")
res.to_csv(os.path.join(OUT, "model_comparison.csv"), index=False)
pred_table.to_csv(os.path.join(OUT, "predictions.csv"), index=False)
 
# =====================================================================
#                  PART C : GRAPHS  (save + show)
# =====================================================================
def finish(fname):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, fname), dpi=120)
    plt.show()
 
 
# =====================================================================
#            PART C1 : GRAPHS OF EACH MODEL SEPARATELY
# =====================================================================
colors = {"KNN": "#4C78A8", "SVM": "#F58518", "Naive Bayes": "#54A24B",
          "Neural Network": "#E45756", "Bagged Trees": "#72B7B2", "Boosted Trees": "#B279A2"}
R = res.set_index("Model")
 
for i, name in enumerate(models, start=1):
    model = models[name]
    pred = model.predict(X_test)
    fpr, tpr, _ = rocs[name]
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"{name}  -  Test Accuracy = {R.loc[name, 'Test Accuracy']*100:.2f}%", fontsize=15)
 
    # (a) Confusion matrix
    sns.heatmap(cms[name], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax[0, 0],
                xticklabels=["Flood", "Severe"], yticklabels=["Flood", "Severe"])
    ax[0, 0].set_title("Confusion Matrix"); ax[0, 0].set_xlabel("Predicted"); ax[0, 0].set_ylabel("Actual")
 
    # (b) ROC curve
    ax[0, 1].plot(fpr, tpr, color=colors[name], lw=2, label=f"AUC = {R.loc[name, 'ROC-AUC']:.3f}")
    ax[0, 1].plot([0, 1], [0, 1], "k--")
    ax[0, 1].set_title("ROC Curve"); ax[0, 1].set_xlabel("False Positive Rate")
    ax[0, 1].set_ylabel("True Positive Rate"); ax[0, 1].legend(loc="lower right")
 
    # (c) Metrics
    vals = [R.loc[name, k] for k in ["Test Accuracy", "Precision", "Recall", "F1-score"]]
    bars = ax[1, 0].bar(["Accuracy", "Precision", "Recall", "F1"], vals, color=colors[name])
    for b, v in zip(bars, vals):
        ax[1, 0].text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center")
    ax[1, 0].set_ylim(0, 1.1); ax[1, 0].set_title("Performance Metrics")
 
    # (d) Actual vs predicted counts
    cnt = pd.DataFrame({"Actual": y_test.value_counts().sort_index().values,
                        "Predicted": pd.Series(pred).value_counts().reindex([0, 1], fill_value=0).values},
                       index=["Flood", "Severe Flood"])
    cnt.plot.bar(ax=ax[1, 1], rot=0, color=["#999999", colors[name]])
    for c in ax[1, 1].containers:
        ax[1, 1].bar_label(c)
    ax[1, 1].set_title(f"Actual vs Predicted counts  (wrong = {R.loc[name, 'Wrong Predictions']})")
    finish(f"model_{i}_{name.replace(' ', '_')}.png")
 
# =====================================================================
#            PART C2 : COMPARISON GRAPHS (ALL MODELS)
# =====================================================================
# 1. Confusion matrices (all models)
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, (name, cm) in zip(axes.ravel(), cms.items()):
    acc = res.set_index("Model").loc[name, "Test Accuracy"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Flood", "Severe"], yticklabels=["Flood", "Severe"])
    ax.set_title(f"{name}  (Acc = {acc*100:.1f}%)"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.suptitle("Confusion Matrices")
finish("1_confusion_matrices.png")
 
# 2. Accuracy comparison
plt.figure(figsize=(9, 5))
ax = sns.barplot(data=res, x="Model", y="Test Accuracy", hue="Model", palette="viridis", legend=False)
for p in ax.patches:
    ax.annotate(f"{p.get_height()*100:.1f}%", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom")
plt.ylim(0, 1.05); plt.xticks(rotation=15); plt.title("Test Accuracy Comparison")
finish("2_accuracy_comparison.png")
 
# 3. Accuracy / Precision / Recall / F1
m = res.melt(id_vars="Model", value_vars=["Test Accuracy", "Precision", "Recall", "F1-score"])
plt.figure(figsize=(11, 5))
sns.barplot(data=m, x="Model", y="value", hue="variable")
plt.ylim(0, 1.05); plt.xticks(rotation=15); plt.title("Accuracy / Precision / Recall / F1")
finish("3_all_metrics.png")
 
# 4. ROC curves
plt.figure(figsize=(7, 6))
for name, (fpr, tpr, _) in rocs.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC={res.set_index('Model').loc[name, 'ROC-AUC']:.3f})")
plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves"); plt.legend()
finish("4_roc_curves.png")
 
# 5. Train vs Test error
e = res.melt(id_vars="Model", value_vars=["Train Accuracy", "Test Accuracy"])
e["Error"] = 1 - e["value"]
plt.figure(figsize=(10, 5))
sns.barplot(data=e, x="Model", y="Error", hue="variable", palette="rocket")
plt.xticks(rotation=15); plt.title("Error Rate: Train vs Test")
finish("5_error_train_vs_test.png")
 
# 6. One-page comparison dashboard
fig, ax = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("MODEL COMPARISON  (best: " + res.loc[0, "Model"] + ")", fontsize=16)
order = list(res["Model"])
sns.barplot(data=res, x="Model", y="Test Accuracy", hue="Model", palette=colors, legend=False, ax=ax[0, 0])
for p in ax[0, 0].patches:
    ax[0, 0].annotate(f"{p.get_height()*100:.1f}%", (p.get_x() + p.get_width()/2, p.get_height()), ha="center", va="bottom")
ax[0, 0].set_ylim(0, 1.05); ax[0, 0].set_title("Test Accuracy"); ax[0, 0].tick_params(axis="x", rotation=20)
sns.barplot(data=m, x="Model", y="value", hue="variable", ax=ax[0, 1])
ax[0, 1].set_ylim(0, 1.05); ax[0, 1].set_title("Accuracy / Precision / Recall / F1"); ax[0, 1].tick_params(axis="x", rotation=20)
ax[0, 1].set_ylabel("")
for name, (fpr, tpr, _) in rocs.items():
    ax[1, 0].plot(fpr, tpr, color=colors[name], label=f"{name} ({R.loc[name, 'ROC-AUC']:.3f})")
ax[1, 0].plot([0, 1], [0, 1], "k--"); ax[1, 0].set_title("ROC Curves"); ax[1, 0].legend(loc="lower right")
ax[1, 0].set_xlabel("False Positive Rate"); ax[1, 0].set_ylabel("True Positive Rate")
sns.barplot(data=res, x="Model", y="Wrong Predictions", hue="Model", palette=colors, legend=False, ax=ax[1, 1])
for p in ax[1, 1].patches:
    ax[1, 1].annotate(int(p.get_height()), (p.get_x() + p.get_width()/2, p.get_height()), ha="center", va="bottom")
ax[1, 1].set_title("Wrong Predictions (lower is better)"); ax[1, 1].tick_params(axis="x", rotation=20)
finish("6_comparison_dashboard.png")
 
print("\nAll files saved in:", OUT)
 
