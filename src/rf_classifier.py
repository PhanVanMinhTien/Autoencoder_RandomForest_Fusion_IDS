# src/rf_classifier.py
from sklearn.ensemble import RandomForestClassifier
import joblib
from . import config

def train_rf(X_train, y_train , save_path=None):
    print("[RandomForest] Training classifier...")
    
    rf = RandomForestClassifier(
        n_estimators = config.RF_ESTIMATORS,
        max_depth = config.RF_MAX_DEPTH,
        max_features = 'sqrt',
        criterion = 'gini', # or 'entropy'
        class_weight = 'balanced', # handling Data Imbalance
        n_jobs = 8, # 14 cores total # -2 means all but one core # -1 means all cores # 
        random_state = config.SEED,
        verbose = 2
    )
    
    rf.fit(X_train, y_train)
    
    # Save Model
    if save_path:
        joblib.dump(rf, save_path)
        print(f"✅ RF Model saved to: {save_path}")
    else:
        default_path = config.MODELS_DIR / "classifier/rf_model.joblib"
        default_path.parent.mkdir(parents=True, exist_ok=True) # <--- DÒNG CỨU CÁNH
        
        joblib.dump(rf, default_path)
        print(f"⚠️ RF Model saved to default: {default_path}")
    
    return rf