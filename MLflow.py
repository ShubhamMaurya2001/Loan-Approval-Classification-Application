import mlflow 
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report, accuracy_score, f1_score, recall_score, precision_score, roc_curve, auc
from mlflow.models.signature import infer_signature
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# Set MLflow experiment name
mlflow.set_experiment("Loan Approval XGBoost Model")

# Get script directory
script_dir = Path(__file__).parent.absolute()

# Load Saved Model and Test Data
try:
    xgboost_model = joblib.load(script_dir / 'xgboost.pkl')
    x_test = joblib.load(script_dir / 'x_test.pkl')
    y_test = joblib.load(script_dir / 'y_test.pkl')
    print("✅ Models and test data loaded successfully")
except FileNotFoundError as e:
    print(f"❌ Error loading files: {e}")
    exit(1)

# Start MLflow Run
with mlflow.start_run() as run:
    print(f"🔍 MLflow Run ID: {run.info.run_id}")
    
    # Model Prediction
    y_predict = xgboost_model.predict(x_test)
    y_predict_proba = xgboost_model.predict_proba(x_test)[:, 1]
    
    # Calculate Metrics
    model_score = xgboost_model.score(x_test, y_test)
    accuracy = accuracy_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict)
    recall = recall_score(y_test, y_predict)
    f1 = f1_score(y_test, y_predict)
    
    # Calculate ROC AUC
    fpr, tpr, _ = roc_curve(y_test, y_predict_proba)
    roc_auc = auc(fpr, tpr)
    
    # Log Model Parameters
    params = xgboost_model.get_params()
    filtered_params = {k: v for k, v in params.items() if v is not None and not isinstance(v, (dict, list))}
    mlflow.log_params(filtered_params)
    
    # Log Metrics
    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "model_score": round(model_score, 4)
    }
    mlflow.log_metrics(metrics)
    
    # Log Classification Report as Artifact
    report = classification_report(y_test, y_predict)
    report_path = script_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(str(report_path), artifact_path="reports")
    
    # Log Confusion Matrix
    plt.figure(figsize=(10, 6))
    sns.heatmap(confusion_matrix(y_test, y_predict), annot=True, cbar=False, fmt='g', cmap='Blues')
    plt.title('XGBoost Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    mlflow.log_figure(plt.gcf(), "plots/confusion_matrix.png")
    plt.close()
    
    # Log ROC Curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Loan Approval Model')
    plt.legend(loc="lower right")
    plt.tight_layout()
    mlflow.log_figure(plt.gcf(), "plots/roc_curve.png")
    plt.close()
    
    # Log Model Signature and Input Example
    input_example = x_test.iloc[:1]
    predicted_example = xgboost_model.predict(input_example)
    signature = infer_signature(input_example, predicted_example)
    
    # Log Model
    mlflow.sklearn.log_model(
        sk_model=xgboost_model,
        artifact_path="model",
        signature=signature,
        input_example=input_example
    )
    
    # Log PKL Model as Artifact
    mlflow.log_artifact(str(script_dir / "xgboost.pkl"), artifact_path="pkl_model")
    
    # Log Tags for better organization
    mlflow.set_tags({
        "model_type": "XGBoost",
        "task": "loan_approval_classification",
        "dataset": "loan_applicants",
        "status": "production"
    })
    
    print("✅ Metrics logged:")
    for metric, value in metrics.items():
        print(f"   {metric}: {value}")
    print(f"✅ Artifacts saved to MLflow")
    print(f"🌐 View results: mlflow ui (run http://localhost:5000)")