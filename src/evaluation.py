import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, matthews_corrcoef
from pathlib import Path

def evaluate_model(model, X_test, y_test, save_dir=None, dataset_name="Test Set"):
    """
    Đánh giá model với đầy đủ metrics: Accuracy, F1, Recall, Precision, MCC.
    Lưu report và confusion matrix vào folder experiment (nếu có save_dir).
    """
    print(f"\n📊 Evaluating on {dataset_name}...")
    
    # 1. Dự đoán
    y_pred = model.predict(X_test)
    
    # 2. Tính Metrics
    acc = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred) 
    report_dict = classification_report(y_test, y_pred, target_names=["Normal", "Attack"], output_dict=True)
    report_str = classification_report(y_test, y_pred, target_names=["Normal", "Attack"], digits=4)
    
    print(f"   ✅ Accuracy: {acc:.4f}")
    print(f"   ⭐ MCC:      {mcc:.4f}")
    
    # 3. Lưu kết quả nếu có đường dẫn
    if save_dir:
        save_dir = Path(save_dir)
        
        # A. Lưu Report Text
        report_path = save_dir.parent / f"report_{dataset_name.replace(' ', '_')}.txt"
        with open(report_path, "w") as f:
            f.write(f"=== EVALUATION REPORT: {dataset_name} ===\n")
            f.write(f"Accuracy: {acc:.6f}\n")
            f.write(f"MCC:      {mcc:.6f}\n\n")
            f.write("--- Detailed Classification Report ---\n")
            f.write(report_str)
        print(f"   📝 Report saved to: {report_path.name}")
            
        # B. Vẽ & Lưu Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
        plt.title(f"CM - {dataset_name}\nAcc: {acc:.4f} | MCC: {mcc:.4f}")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        cm_path = save_dir / f"cm_{dataset_name.replace(' ', '_')}.png"
        plt.savefig(cm_path)
        plt.close()
        print(f"   🖼️ Confusion Matrix saved to: {cm_path.name}")

    return {
        "accuracy": acc,
        "mcc": mcc,
        "report": report_dict
    }





########################################################################################################################
#
#  1st 
#
########################################################################################################################
# # src/evaluation.py
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import classification_report, confusion_matrix
# import pandas as pd
# from . import config

# def evaluate_model(model, X_test, y_test, label_names=None):
#     print("\n[Evaluation] Predicting...")
#     y_pred = model.predict(X_test)
    
#     # 1. Text Report
#     print("\n=== CLASSIFICATION REPORT ===")
#     report = classification_report(y_test, y_pred, target_names=label_names, digits=4)
#     print(report)
    
#     # Lưu report ra file
#     with open(config.RESULTS_DIR / "classification_report.txt", "w") as f:
#         f.write(report)

#     # 2. Confusion Matrix Plot
#     cm = confusion_matrix(y_test, y_pred)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#                 xticklabels=label_names, yticklabels=label_names)
#     plt.title('Confusion Matrix (Deep AE + RF)')
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
    
#     save_path = config.FIGURES_DIR / "confusion_matrix.png"
#     plt.savefig(save_path)
#     print(f"[Evaluation] Confusion Matrix saved to {save_path}")