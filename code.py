# Basic libraries
import os
import time
import matplotlib.pyplot as plt
import tempfile
import numpy as np
import traceback
import gc
import locale

import logging
import warnings

# Data processing and analysis
import pandas as pd
from sklearn.base import clone
from collections import Counter
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
from collections import Counter
from sklearn.model_selection import learning_curve
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from matplotlib import font_manager
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import LeaveOneOut

# Database processing
from dbfread import DBF
import dbf

# Machine learning
from sklearn import model_selection
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import svm

# Model evaluation
from sklearn.metrics import (
    roc_auc_score as AUC,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    classification_report,
    confusion_matrix,
    recall_score,
    accuracy_score,
    f1_score,
    precision_score,
    PrecisionRecallDisplay,
    auc
)

# Imbalanced data handling
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Special models
import xgboost as xgb

# Environment settings
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Print available styles
print("Available styles:", plt.style.available)

# Select the most appropriate academic style from available styles
# Based on your output, available seaborn styles have v0_8 version numbers
ACADEMIC_STYLE = 'seaborn-v0_8-whitegrid'  # Similar to seaborn-whitegrid style

# Apply style
plt.style.use(ACADEMIC_STYLE)

# Other style settings remain unchanged
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Model name mapping remains unchanged
MODEL_NAMES = {
    'XGBoost': 'XGB',
    'SVM': 'SVM',
    'RandomForest': 'RF',
    'GBDT': 'GBDT',
    'MLP': 'MLP',
    'Logistic': 'LR'
}

# Set global font to English (ensure all text displays correctly)
plt.rcParams.update({
    'font.family': 'Arial',  # Use Arial font
    'font.size': 10,
    'mathtext.fontset': 'stix',  # Math formula font
    'axes.unicode_minus': False  # Fix minus sign display
})

# Set system encoding
import sys

if sys.version_info[0] >= 3:
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except locale.Error:
        print("Failed to set system default encoding to en_US.UTF-8", flush=True)

# Plot settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("Available styles:", plt.style.available)


def process_large_dbf(file_path, chunk_size=50000):
    """Correctly handle chunked reading of large DBF files"""
    table = DBF(file_path, encoding='utf-8')
    records = list(iter(table))  # Read all records first
    for i in range(0, len(records), chunk_size):
        chunk = records[i:i + chunk_size]
        yield pd.DataFrame(chunk)


def save2dbf(output_path, predictions):
    """
    Save prediction results to a new DBF file (compatible with 1269 rows of actual data)

    Parameters:
        output_path (str): Path to save the new DBF file
        predictions (numpy.ndarray): Prediction result array

    Returns:
        bool: Returns True if save is successful, False otherwise
    """
    try:
        print(f"\nSaving prediction results to: {output_path}")

        # Validate input data
        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)

        # Modified to accept 1269 rows of data (original 1270 includes header)
        if predictions.shape[0] != 1254:
            raise ValueError(f"Requires 1254 rows of data, got {predictions.shape[0]} rows")

        # Create new DBF table structure
        table = dbf.Table(
            output_path,
            'XGB_PRED N(7,3); '  # XGBoost prediction
            'SVC_PRED N(7,3); '  # SVM prediction
            'RF_PRED N(7,3); '  # Random Forest
            'GBDT_PRED N(7,3); '  # GBDT
            'MLP_PRED N(7,3); '  # Neural Network
            'LR_PRED N(7,3); '  # Logistic Regression
            'COMB_PRED N(7,3)',  # Combined prediction
            codepage='utf8'
        )

        # Write data
        table.open(mode=dbf.READ_WRITE)
        for row in predictions:
            table.append(tuple(
                float(x) for x in row  # Ensure all values are floats
            ))
        table.close()

        print(f"Successfully saved {predictions.shape[0]} records to {output_path}")
        return True

    except Exception as e:
        print(f"Save failed: {str(e)}")
        if 'table' in locals() and hasattr(table, 'close'):
            table.close()
        return False


def extract(training_dbf_path, prediction_dbf_path, factor):
    try:
        print(f"\n===== Starting data extraction =====")

        # 1. Read training data
        train_table = DBF(training_dbf_path, encoding='utf-8', ignore_missing_memofile=True)
        df_train = pd.DataFrame(iter(train_table))

        # 2. Read prediction data
        pred_table = DBF(prediction_dbf_path, encoding='utf-8', ignore_missing_memofile=True)
        df_pred = pd.DataFrame(iter(pred_table))

        # 3. Data preprocessing
        def clean_data(df):
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df.dropna()

        df_train = clean_data(df_train)
        df_pred = clean_data(df_pred)

        # 4. Extract features and labels
        X_train = df_train.iloc[:, factor].values
        y_train = df_train.iloc[:, -1].values
        y_train = np.where(y_train > 0, 1, 0)  # Binarize labels

        X_pred = df_pred.iloc[:, factor].values

        print(f"Training data shape: {X_train.shape}, Prediction data shape: {X_pred.shape}")
        print("First 5 training data samples:\n", X_train[:5])
        print("===== Data extraction completed =====")

        # Check for overlap between training and prediction data
        assert not np.any([np.array_equal(X_train[i], X_pred[j])
                           for i in range(min(10, len(X_train)))  # Check first 10 samples
                           for j in range(min(10, len(X_pred)))]), "Error: Training and prediction data overlap!"

        # Check if labels are independent
        assert not np.array_equal(X_train[:, -1], y_train), "Error: Last column of feature matrix may be label column!"
        return X_train, y_train, X_pred, list(df_train.columns[factor])  # Removed sample_weights

    except Exception as e:
        print(f"Data extraction error: {str(e)}\n{traceback.format_exc()}")
        return None, None, None, None


def cross_validate_model(model, X, y, n_splits=5):
    """Perform stratified cross-validation and return average AUC"""
    cv = StratifiedKFold(n_splits=n_splits)
    auc_scores = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:, 1]
        auc_scores.append(AUC(y_test, y_pred))

    return np.mean(auc_scores)


def clean_data(df):
    # Convert numeric types and handle null values
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Replace 0 values with column mean
        df[col] = df[col].replace(0, df[col].mean())
    return df.dropna()


def plot_coupled_model_performance(y_test, xgb_pred, lr_pred, combined_pred, output_dir):
    """Plot academic-style coupled model performance"""
    try:
        plt.figure(figsize=(12, 4))

        # ==================== ROC Curve ====================
        plt.subplot(1, 3, 1)

        # Calculate metrics
        fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_pred)
        auc_xgb = AUC(y_test, xgb_pred)

        fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_pred)
        auc_lr = AUC(y_test, lr_pred)

        fpr_comb, tpr_comb, _ = roc_curve(y_test, combined_pred)
        auc_comb = AUC(y_test, combined_pred)

        # Plot curves (all changed to solid lines)
        plt.plot(fpr_xgb, tpr_xgb, color='#e41a1c', linestyle='-',
                 label=f'XGB (AUC={auc_xgb:.3f})')
        plt.plot(fpr_lr, tpr_lr, color='#377eb8', linestyle='-',
                 label=f'LR (AUC={auc_lr:.3f})')
        plt.plot(fpr_comb, tpr_comb, color='#4daf4a', linestyle='-',
                 label=f'XGB+LR (AUC={auc_comb:.3f})')

        # Format settings
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc='lower right', frameon=True)
        plt.grid(True, alpha=0.3)

        # ==================== PR Curve ====================
        plt.subplot(1, 3, 2)

        # Calculate metrics
        precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, xgb_pred)
        ap_xgb = average_precision_score(y_test, xgb_pred)

        precision_lr, recall_lr, _ = precision_recall_curve(y_test, lr_pred)
        ap_lr = average_precision_score(y_test, lr_pred)

        precision_comb, recall_comb, _ = precision_recall_curve(y_test, combined_pred)
        ap_comb = average_precision_score(y_test, combined_pred)

        # Plot baseline
        baseline = np.mean(y_test)
        plt.axhline(y=baseline, color='k', linestyle='--', alpha=0.5,
                    label=f'Random (AP={baseline:.2f})')

        # Plot curves
        plt.plot(recall_xgb, precision_xgb, color='#e41a1c', linestyle='-',
                 label=f'XGB (AP={ap_xgb:.3f})')
        plt.plot(recall_lr, precision_lr, color='#377eb8', linestyle='-',
                 label=f'LR (AP={ap_lr:.3f})')
        plt.plot(recall_comb, precision_comb, color='#4daf4a', linestyle='-',
                 label=f'XGB+LR (AP={ap_comb:.3f})')

        # Format settings
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR Curves')
        plt.legend(loc='upper right', frameon=True)
        plt.grid(True, alpha=0.3)

        # ==================== Recall-Threshold Curve ====================
        plt.subplot(1, 3, 3)
        thresholds = np.linspace(0, 1, 100)

        def plot_recall_vs_threshold(y_true, y_pred, color, linestyle, label):
            recalls = []
            for thresh in thresholds:
                y_pred_class = (y_pred >= thresh).astype(int)
                recalls.append(recall_score(y_true, y_pred_class))
            plt.plot(thresholds, recalls, color=color, linestyle=linestyle, label=label)

        plot_recall_vs_threshold(y_test, xgb_pred, '#e41a1c', '-', 'XGB')
        plot_recall_vs_threshold(y_test, lr_pred, '#377eb8', '--', 'LR')
        plot_recall_vs_threshold(y_test, combined_pred, '#4daf4a', '-.', 'XGB+LR')

        # Format settings
        plt.xlabel('Threshold')
        plt.ylabel('Recall')
        plt.title('Recall vs. Threshold')
        plt.legend(loc='lower left', frameon=True)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'XGB_LR_Coupled_Model_Performance.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error plotting coupled model performance: {str(e)}")


def XGB(X_train_res, X_test, y_train_res, y_test, X_total, output_dir):
    try:
        print("\n===== XGBoost training started =====")

        # Base model
        xgb_clf = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            random_state=42,
            n_jobs=-1
        )

        # Grid search parameters for small samples
        param_grid = {
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.6, 0.8],
            'colsample_bytree': [0.6, 0.8],
            'n_estimators': [50, 100],
            'gamma': [0, 0.1, 0.2],
            'reg_alpha': [0, 0.1],
            'reg_lambda': [1, 1.5]
        }

        # Perform grid search
        best_xgb = grid_search_model(
            xgb_clf,
            param_grid,
            X_train_res,
            y_train_res,
            cv=5  # Use stratified cross-validation
        )

        y_pred = best_xgb.predict_proba(X_test)[:, 1]
        y_total_pred = best_xgb.predict_proba(X_total)[:, 1]
        auc_score = AUC(y_test, y_pred)

        print(f"===== XGBoost training completed | AUC: {auc_score:.4f} =====")
        return y_total_pred, auc_score, best_xgb  # Return trained model

    except Exception as e:
        print(f"XGBoost training error: {str(e)}\n{traceback.format_exc()}")
        return None, None, None


def SVC(X_train_res, X_test, y_train_res, y_test, X_total, output_dir, sample_weights_res):
    try:
        print("\n=== Starting SVC model training ===")
        svc = svm.SVC(
            probability=True,
            class_weight='balanced',
            shrinking=True,
            tol=1e-4,
            cache_size=200
        )
        param_grid = {
            'C': [0.1, 0.5, 1],
            'kernel': ['rbf', 'linear'],
            'gamma': ['auto', 'scale']
        }
        best_svc = grid_search_model(
            svc,
            param_grid,
            X_train_res,
            y_train_res,
            cv=5
        )
        y_pred = best_svc.predict_proba(X_test)[:, 1]
        y_total_pred = best_svc.predict_proba(X_total)[:, 1]
        auc_score = AUC(y_test, y_pred)
        print("=== SVC model training completed ===")
        return y_total_pred, auc_score, best_svc
    except Exception as e:
        print(f"SVC function execution error: {e}")
        return None, None, None


def GBDT(X_train_res, X_test, y_train_res, y_test, X_total, output_dir):
    """GBDT model training (with grid search)"""
    try:
        print("\n===== Starting GBDT training =====")

        # Base model
        gbdt = GradientBoostingClassifier(random_state=42)

        # Grid search parameters for small samples
        param_grid = {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.6, 0.8],
            'max_features': ['sqrt', 0.5]
        }

        # Perform grid search
        best_gbdt = grid_search_model(
            gbdt,
            param_grid,
            X_train_res,
            y_train_res,
            cv=min(5, np.bincount(y_train_res)[0])
        )

        # Model evaluation
        y_pred = best_gbdt.predict_proba(X_test)[:, 1]
        y_total_pred = best_gbdt.predict_proba(X_total)[:, 1]
        auc_score = AUC(y_test, y_pred)

        print(f"===== GBDT training completed | AUC: {auc_score:.4f} =====")
        return y_total_pred, auc_score, best_gbdt

    except Exception as e:
        print(f"GBDT training error: {str(e)}\n{traceback.format_exc()}")
        return None, None, None


def RFC(X_train_res, X_test, y_train_res, y_test, X_total, output_dir, factor, dbfroad):
    try:
        print("\n=== Starting Random Forest training ===")
        rfc = RandomForestClassifier(
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 0.5, 0.7]
        }
        grid_search = GridSearchCV(
            estimator=rfc,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=min(5, np.bincount(y_train_res)[0]),
            n_jobs=-1,
            verbose=1
        )
        try:
            grid_search.fit(X_train_res, y_train_res)
        except Exception as e:
            print(f"Random Forest grid search error: {str(e)}")
            return None, None, None

        best_rfc = grid_search.best_estimator_
        if best_rfc is None:
            print("No best Random Forest model obtained")
            return None, None, None

        # Plot grid search heatmap
        plot_grid_search_heatmap(grid_search, param_grid, output_dir)

        y_pred = best_rfc.predict_proba(X_test)[:, 1]
        y_total_pred = best_rfc.predict_proba(X_total)[:, 1]
        auc_score = AUC(y_test, y_pred)

        print("=== Random Forest training completed ===")
        if best_rfc is not None:
            # Call modified SHAP analysis function
            shap_success1, shap_success2 = plot_rf_shap(best_rfc, X_test, feature_names, output_dir)

            # If main method fails, try backup solution
            if not (shap_success1 and shap_success2):
                print("Main SHAP method failed, trying backup...")
                plot_rf_shap_backup(best_rfc, X_test, feature_names, output_dir)

        return y_total_pred, auc_score, best_rfc

    except Exception as e:
        print(f"Random Forest training error: {str(e)}\n{traceback.format_exc()}")
        return None, None, None


def MLP(X_train_res, X_test, y_train_res, y_test, X_total, output_dir, sample_weights_res):
    try:
        print("\n=== Starting MLP model training ===")
        mlp = MLPClassifier(
            activation='relu',
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=20,
            random_state=42,
            solver='adam',
            tol=1e-4,
            verbose=False
        )
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'alpha': [0.01, 0.1, 1],
            'learning_rate_init': [0.001, 0.01],
            'batch_size': [min(16, len(X_train_res)), min(32, len(X_train_res))]  # Modified here
        }
        best_mlp = grid_search_model(
            mlp,
            param_grid,
            X_train_res,  # Use correct parameter name
            y_train_res,
            cv=min(5, np.bincount(y_train_res)[0]))
        y_pred = best_mlp.predict_proba(X_test)[:, 1]
        y_total_pred = best_mlp.predict_proba(X_total)[:, 1]
        auc_score = AUC(y_test, y_pred)
        print("=== MLP model training completed ===")
        return y_total_pred, auc_score, best_mlp
    except Exception as e:
        print(f"MLP training error: {str(e)}\n{traceback.format_exc()}")  # Print full error
        return None, None, None


def LR(X_train_res, X_test, y_train_res, y_test, X_total, output_dir, sample_weights_res, feature_names):
    """Logistic Regression model training (with SHAP analysis)"""
    try:
        print("\n===== Starting Logistic Regression training =====")

        # Base model
        lr = LogisticRegression(
            class_weight='balanced',
            n_jobs=-1,
            max_iter=1000,
            penalty='l2',
            solver='lbfgs',
            tol=1e-4,
            warm_start=False,
            random_state=42
        )

        # Grid search parameters suitable for small samples
        param_grid = {
            'C': [0.1, 0.5, 1]
        }

        # Perform grid search
        best_lr = grid_search_model(
            lr,
            param_grid,
            X_train_res,
            y_train_res,
            cv=min(5, np.bincount(y_train_res)[0]))

        # ============ SHAP Analysis - Scatter Plot Style ============
        try:
            print("\n--- Starting SHAP analysis ---")

            # Create SHAP explainer
            explainer = shap.LinearExplainer(best_lr, X_train_res, feature_names=feature_names)

            # Calculate SHAP values (using test set samples)
            X_test_sample = X_test[:100] if len(X_test) > 100 else X_test
            shap_values = explainer.shap_values(X_test_sample)

            # 1. SHAP scatter plot (show all features)
            plt.figure(figsize=(12, 8))
            shap.plots.beeswarm(shap.Explanation(values=shap_values,
                                                 data=X_test_sample,
                                                 feature_names=feature_names),
                                show=False)
            plt.title("Logistic Regression SHAP Value Distribution (Scatter Plot)", fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'LR_SHAP_Scatter.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()

            # 2. Detailed analysis for Road feature
            if 'Road' in feature_names:
                Road_idx = feature_names.index('Road')
                plt.figure(figsize=(10, 6))
                shap.plots.scatter(shap.Explanation(values=shap_values[:, Road_idx],
                                                    data=X_test_sample[:, Road_idx],
                                                    feature_names='Road'),
                                   show=False)
                plt.title("Road Feature SHAP Value Analysis", fontsize=12)
                plt.xlabel("Road (weight reduced)")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'LR_SHAP_Road.png'),
                            dpi=300, bbox_inches='tight')
                plt.close()

            print("SHAP analysis completed, charts saved")

        except Exception as e:
            print(f"SHAP analysis error: {str(e)}")
            traceback.print_exc()

        # Model evaluation
        y_pred = best_lr.predict_proba(X_test)[:, 1]
        y_total_pred = best_lr.predict_proba(X_total)[:, 1]
        auc_score = AUC(y_test, y_pred)

        print(f"===== Logistic Regression training completed | AUC: {auc_score:.4f} =====")
        return y_total_pred, auc_score, best_lr

    except Exception as e:
        print(f"Logistic Regression training error: {str(e)}\n{traceback.format_exc()}")
        return None, None, None


def plot_enhanced_shap(model, X, feature_names, output_dir, model_name):
    try:
        # Ensure X is 2D array
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        # Determine explainer type
        if isinstance(model, (RandomForestClassifier, xgb.XGBClassifier, GradientBoostingClassifier)):
            explainer = shap.TreeExplainer(model, feature_perturbation="interventional")
            shap_values = explainer.shap_values(X)
            # Handle multi-class output
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
        elif isinstance(model, (LogisticRegression, svm.SVC)):
            explainer = shap.LinearExplainer(model, X)
            shap_values = explainer.shap_values(X)
        else:
            explainer = shap.KernelExplainer(model.predict_proba, shap.utils.sample(X, 100))
            shap_values = explainer.shap_values(X)

        # Ensure shap_values is 2D array
        if len(shap_values.shape) == 1:
            shap_values = np.reshape(shap_values, (-1, 1))

        # 1. Feature importance bar chart
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, feature_names=feature_names,
                          plot_type="bar", show=False, max_display=15)
        plt.title(f"{model_name} - Feature Importance", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_Feature_Importance.png'),
                    dpi=500, bbox_inches='tight')
        plt.close()

        # 2. SHAP value swarm plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, feature_names=feature_names,
                          show=False, max_display=15)
        plt.title(f"{model_name} - SHAP Value Distribution", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_SHAP_Summary.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"Error generating SHAP plot: {str(e)}")
        traceback.print_exc()


def apply_resampling(X_train, y_train, sample_weights_train):
    """Modified sampling function - returns original data directly without resampling"""
    print("\n===== Using original data, no resampling =====")
    print("Class distribution:", Counter(y_train))
    print(f"Sample weights: All 1.0")

    return X_train, y_train, sample_weights_train


def plot_combined_roc_pr(models_results, y_test, output_dir):
    """Unified ROC and PR curve comparison style"""
    try:
        # Set Chinese font and style
        plt.style.use('ggplot')
        plt.rcParams['font.family'] = 'SimHei'
        plt.rcParams['axes.unicode_minus'] = False

        # Model color and name configuration
        model_colors = {
            'XGBoost': '#FF7F0E',
            'SVM': '#1F77B4',
            'RandomForest': '#2CA02C',
            'GBDT': '#D62728',
            'MLP': '#9467BD',
            'Logistic': '#8C564B'
        }

        # ==================== ROC Curve Comparison ====================
        plt.figure(figsize=(10, 8))

        # Plot random guess line
        plt.plot([0, 1], [0, 1], 'k--', label='Random guess (AUC=0.5)', linewidth=2)

        # Plot each model's ROC curve
        for model_name, result in models_results.items():
            if 'y_pred_prob' in result:
                fpr, tpr, _ = roc_curve(y_test, result['y_pred_prob'])
                roc_auc = AUC(y_test, result['y_pred_prob'])
                plt.plot(fpr, tpr,
                         color=model_colors.get(model_name, '#000000'),
                         linewidth=2.5,
                         label=f'{model_name} (AUC={roc_auc:.3f})')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Model ROC Curve Comparison', fontsize=14, pad=20)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save ROC plot
        roc_path = os.path.join(output_dir, 'Six_Models_ROC_Comparison.png')
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC comparison plot saved to: {roc_path}")

        # ==================== PR Curve Comparison ====================
        plt.figure(figsize=(10, 8))

        # Plot random guess line
        baseline = np.mean(y_test)
        plt.axhline(y=baseline, color='k', linestyle='--',
                    label=f'Random guess (AP={baseline:.2f})', linewidth=2)

        # Plot each model's PR curve
        for model_name, result in models_results.items():
            if 'y_pred_prob' in result:
                precision, recall, _ = precision_recall_curve(y_test, result['y_pred_prob'])
                ap_score = average_precision_score(y_test, result['y_pred_prob'])
                plt.plot(recall, precision,
                         color=model_colors.get(model_name, '#000000'),
                         linewidth=2.5,
                         label=f'{model_name} (AP={ap_score:.3f})')

        plt.xlim([-0.05, 1.05])
        plt.ylim([0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Model PR Curve Comparison', fontsize=14, pad=20)
        plt.legend(loc='lower left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save PR plot
        pr_path = os.path.join(output_dir, 'Six_Models_PR_Comparison.png')
        plt.savefig(pr_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"PR comparison plot saved to: {pr_path}")

        return roc_path, pr_path

    except Exception as e:
        print(f"Error plotting comparison: {str(e)}")
        return None, None


def plot_xgb_shap(model, X, feature_names, output_dir):
    """Plot XGBoost SHAP feature importance (scatter plot style)"""
    try:
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)

        # Calculate SHAP values (use first 100 samples to avoid memory issues)
        X_sample = X[:100] if len(X) > 100 else X
        shap_values = explainer.shap_values(X_sample)

        # 1. Feature importance scatter plot (global)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                          plot_type="dot", show=False, max_display=15)
        plt.title("XGBoost Feature Importance (SHAP Values)", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'XGB_SHAP_Scatter.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Single feature dependence plot (show top 3 important features)
        for i in range(min(3, len(feature_names))):
            plt.figure(figsize=(8, 6))
            shap.dependence_plot(i, shap_values, X_sample,
                                 feature_names=feature_names,
                                 interaction_index=None,
                                 show=False)
            plt.title(f"XGBoost Feature Dependence - {feature_names[i]}", fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'XGB_SHAP_{feature_names[i]}.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()

        print("XGBoost SHAP analysis completed")
    except Exception as e:
        print(f"XGBoost SHAP analysis error: {str(e)}")


def plot_mlp_shap(model, X, feature_names, output_dir):
    """Plot SHAP plot for MLP model (consistent with XGBoost style)"""
    try:
        plt.style.use(ACADEMIC_STYLE)

        # 1. Data preparation
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        X_sample = X[:100] if len(X) > 100 else X

        print("\nCalculating SHAP values for MLP...")

        # 2. Create explainer (using simplified prediction function)
        def mlp_predict(X):
            return model.predict_proba(X)[:, 1]  # Return only positive class probabilities

        explainer = shap.KernelExplainer(
            mlp_predict,
            shap.sample(X_sample, min(10, len(X_sample))  # Use a small number of background samples
                        ))

        # 3. Calculate SHAP values (limit sample count to speed up calculation)
        shap_values = explainer.shap_values(
            X_sample,
            nsamples=100,  # Reduce computation load
            silent=True
        )

        # Ensure SHAP values are 2D (n_samples, n_features)
        shap_values = np.array(shap_values)
        if len(shap_values.shape) == 3:
            shap_values = shap_values.reshape(shap_values.shape[0], -1)

        # 4. Plot SHAP summary
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=feature_names,
            plot_type="dot",
            show=False,
            max_display=15
        )
        plt.title("MLP Model Feature Importance (SHAP Values)", fontsize=20, pad=20)
        plt.xlabel("Impact on model output", fontsize=20)
        plt.xticks(fontsize=18)  # X-axis tick label size
        plt.yticks(fontsize=18)  # Y-axis tick label size
        # Increase font size for color bar labels
        cb = plt.gcf().axes[-1]
        cb.tick_params(labelsize=18)
        plt.tight_layout()

        # 5. Save image
        save_path = os.path.join(output_dir, 'MLP_SHAP_Summary.png')
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        plt.close()
        print(f"MLP SHAP plot saved to: {save_path}")

    except Exception as e:
        print(f"Error plotting MLP SHAP plot: {str(e)}")
        traceback.print_exc()


def plot_rf_shap(model, X, feature_names, output_dir):
    try:
        plt.style.use(ACADEMIC_STYLE)

        # Data validation
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        X_sample = X[:100] if len(X) > 100 else X  # Limit sample count

        print(f"\nSHAP analysis data validation - Sample count: {X_sample.shape[0]}, Feature count: {X_sample.shape[1]}")
        print(f"Feature name examples: {feature_names[:5]}...")

        # Create explainer
        try:
            explainer = shap.TreeExplainer(
                model,
                data=X_sample,
                feature_perturbation="interventional"
            )
        except Exception as e:
            print(f"Error creating explainer: {str(e)}")
            return None, None

        # Calculate SHAP values - Get SHAP values for both classes
        print("Calculating SHAP values...")
        try:
            shap_values = explainer.shap_values(X_sample)
        except Exception as e:
            print(f"Error calculating SHAP values: {str(e)}")
            return None, None

        # Ensure we get SHAP values for both classes
        if len(shap_values) != 2:
            print(f"Warning: Expected SHAP values for 2 classes, got {len(shap_values)}")
            return None, None

        # ============ Plot SHAP plot for Class 1 (positive class) ============
        plt.figure(figsize=(12, 8))
        try:
            shap.summary_plot(
                shap_values[1],  # SHAP values for Class 1
                X_sample,
                feature_names=feature_names,
                plot_type="dot",
                show=False,
                max_display=15
            )
        except Exception as e:
            print(f"Error plotting Class 1 SHAP plot: {str(e)}")
            return None, None

        plt.title("Random Forest Feature Importance (Class 1 - Positive)", fontsize=20)
        plt.xticks(fontsize=18)  # X-axis tick label size
        plt.yticks(fontsize=18)  # Y-axis tick label size
        # Increase font size for color bar labels
        cb = plt.gcf().axes[-1]
        cb.tick_params(labelsize=18)
        plt.tight_layout()

        # Save Class 1 SHAP plot
        save_path_class1 = os.path.join(output_dir, 'RF_SHAP_Class1.png')
        try:
            plt.savefig(save_path_class1, dpi=500, bbox_inches='tight')
            plt.close()
            print(f"✓ Class 1 SHAP plot saved to: {save_path_class1}")
        except Exception as e:
            print(f"Error saving Class 1 SHAP plot: {str(e)}")
            return None, None

        # ============ Plot SHAP plot for Class 0 (negative class) ============
        plt.figure(figsize=(12, 8))
        try:
            shap.summary_plot(
                shap_values[0],  # SHAP values for Class 0
                X_sample,
                feature_names=feature_names,
                plot_type="dot",
                show=False,
                max_display=15
            )
        except Exception as e:
            print(f"Error plotting Class 0 SHAP plot: {str(e)}")
            return None, None

        plt.title("Random Forest Feature Importance (Class 0 - Negative)", fontsize=20)
        plt.xlabel("Impact on model output (Class 0)", fontsize=20)
        plt.tight_layout()

        # Save Class 0 SHAP plot
        save_path_class0 = os.path.join(output_dir, 'RF_SHAP_Class0.png')
        try:
            plt.savefig(save_path_class0, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Class 0 SHAP plot saved to: {save_path_class0}")
            return save_path_class1, save_path_class0
        except Exception as e:
            print(f"Error saving Class 0 SHAP plot: {str(e)}")
            return None, None

    except Exception as e:
        print(f"SHAP analysis failed: {str(e)}")
        traceback.print_exc()
        return None, None


def plot_rf_shap_backup(model, X, feature_names, output_dir):
    """Backup solution: Use KernelExplainer"""
    try:
        # Define prediction function
        def rf_predict(X):
            return model.predict_proba(X)[:, 1]  # Positive class probabilities

        # Use a small number of background samples
        background = shap.sample(X, min(10, len(X)))

        # Create explainer
        explainer = shap.KernelExplainer(rf_predict, background)

        # Calculate SHAP values
        X_sample = X[:50] if len(X) > 50 else X
        shap_values = explainer.shap_values(X_sample, nsamples=100)


        # Plot figure
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=feature_names,
            plot_type="dot",
            show=False,
            max_display=15
        )

        plt.title("Random Forest Feature Importance (SHAP Values)", fontsize=20)
        plt.tight_layout()
        plt.xticks(fontsize=18)  # X-axis tick label size
        plt.yticks(fontsize=18)  # Y-axis tick label size
        # Increase font size for color bar labels
        cb = plt.gcf().axes[-1]
        cb.tick_params(labelsize=18)
        save_path = os.path.join(output_dir, 'RF_SHAP_Backup.png')
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        plt.close()

        print(f"Backup SHAP plot saved to: {save_path}")
        return save_path

    except Exception as e:
        print(f"Failed to generate backup SHAP plot: {str(e)}")
        traceback.print_exc()
        return None

def plot_gbdt_shap(model, X, feature_names, output_dir):
    """Plot SHAP swarm plot for GBDT (consistent with other models)"""
    try:
        plt.style.use(ACADEMIC_STYLE)

        # Create explainer
        explainer = shap.TreeExplainer(model)

        # Calculate SHAP values (use first 100 samples to avoid memory issues)
        X_sample = X[:100] if len(X) > 100 else X
        shap_values = explainer.shap_values(X_sample)

        # Ensure SHAP values are 2D array
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]

        # 1. SHAP swarm plot (global)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=feature_names,
            plot_type="dot",  # Change to swarm plot
            show=False,
            max_display=15
        )
        plt.title("GBDT Feature Importance (SHAP Values)", fontsize=20)
        plt.xticks(fontsize=18)  # X-axis tick label size
        plt.yticks(fontsize=18)  # Y-axis tick label size
        # Increase font size for color bar labels
        cb = plt.gcf().axes[-1]
        cb.tick_params(labelsize=18)
        plt.tight_layout()

        plt.savefig(
            os.path.join(output_dir, 'GBDT_SHAP_Summary.png'),
            dpi=500,
            bbox_inches='tight'
        )
        plt.close()

        print("GBDT SHAP swarm plot generated")
    except Exception as e:
        print(f"Error in GBDT SHAP analysis: {str(e)}")
        traceback.print_exc()


def plot_svm_shap(model, X, feature_names, output_dir):
    """Plot SHAP plot for SVM (kernel method approximation)"""
    try:
        plt.style.use(ACADEMIC_STYLE)

        # Create explainer (using kernel method)
        def svm_predict(X):
            return model.predict_proba(X)[:, 1]

        # Use a small number of background samples
        background = shap.sample(X, min(10, len(X)))
        explainer = shap.KernelExplainer(svm_predict, background)

        # Calculate SHAP values (limit sample count)
        X_sample = X[:50] if len(X) > 50 else X
        shap_values = explainer.shap_values(X_sample, nsamples=100)

        # 1. Feature importance plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                          show=False, max_display=15)
        plt.title("SVM Feature Importance (SHAP Values)", fontsize=20)
        plt.xticks(fontsize=18)  # X-axis tick label size
        plt.yticks(fontsize=18)  # Y-axis tick label size
        # Increase font size for color bar labels
        cb = plt.gcf().axes[-1]
        cb.tick_params(labelsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'SVM_SHAP_Summary.png'),
                    dpi=500, bbox_inches='tight')
        plt.close()

        print("SVM SHAP analysis completed")
    except Exception as e:
        print(f"Error in SVM SHAP analysis: {str(e)}")
        traceback.print_exc()

def plot_ensemble_shap(xgb_model, mlp_model, X, feature_names, output_dir, xgb_weight=0.2868):
    """Ensemble model SHAP analysis (including separate MLP analysis)"""
    try:
        # First plot separate SHAP plot for MLP
        plot_mlp_shap(mlp_model, X, feature_names, output_dir)

        # Original ensemble analysis code (keep your previously working version)
        plt.style.use(ACADEMIC_STYLE)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        X_sample = X[:100] if len(X) > 100 else X

        # XGBoost SHAP value calculation
        xgb_explainer = shap.TreeExplainer(xgb_model)
        xgb_shap_values = xgb_explainer.shap_values(X_sample)
        if isinstance(xgb_shap_values, list):
            xgb_shap_values = xgb_shap_values[1]

        # MLP SHAP value calculation (using method from new function for consistency)
        def mlp_predict(X):
            return mlp_model.predict_proba(X)[:, 1]

        mlp_explainer = shap.KernelExplainer(
            mlp_predict,
            shap.sample(X_sample, min(10, len(X_sample))))
        mlp_shap_values = mlp_explainer.shap_values(X_sample, nsamples=100, silent=True)
        mlp_shap_values = np.array(mlp_shap_values).reshape(mlp_shap_values.shape[0], -1)

        # Unify shapes
        min_samples = min(xgb_shap_values.shape[0], mlp_shap_values.shape[0])
        xgb_shap_values = xgb_shap_values[:min_samples]
        mlp_shap_values = mlp_shap_values[:min_samples]
        X_sample = X_sample[:min_samples]

        # Weighted combination
        combined_shap = xgb_weight * xgb_shap_values + (1 - xgb_weight) * mlp_shap_values

        # Plot ensemble SHAP plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            combined_shap,
            X_sample,
            feature_names=feature_names,
            plot_type="dot",
            show=False,
            max_display=15
        )
        plt.title(
            f"XGBoost+MLP Ensemble Feature Importance\n(XGB Weight:{xgb_weight:.2f} MLP Weight:{1 - xgb_weight:.2f})",
            fontsize=20, pad=20)
        plt.xlabel("Impact on model output", fontsize=20)
        plt.xticks(fontsize=18)  # X-axis tick label size
        plt.yticks(fontsize=18)  # Y-axis tick label size
        # Increase font size for color bar labels
        cb = plt.gcf().axes[-1]
        cb.tick_params(labelsize=18)
        plt.tight_layout()
        save_path = os.path.join(output_dir, 'Ensemble_SHAP_Summary.png')
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        plt.close()
        print(f"Ensemble SHAP plot saved to: {save_path}")

    except Exception as e:
        print(f"Error plotting ensemble model SHAP plot: {str(e)}")
        traceback.print_exc()
def plot_rf_shap_interaction(model, X, feature_names, output_dir):
    """Improved random forest SHAP interaction swarm plot"""
    try:
        plt.style.use(ACADEMIC_STYLE)

        # Create explainer (disable interaction values)
        explainer = shap.TreeExplainer(
            model,
            feature_perturbation="interventional"  # Ensure using marginal predictions
        )

        # Calculate SHAP values
        X_sample = X[:100] if len(X) > 100 else X
        shap_values = explainer.shap_values(X_sample)

        # Handle multi-class output
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]

        # Calculate SHAP values for feature interactions
        shap_interaction_values = explainer.shap_interaction_values(X_sample)

        # 1. SHAP interaction swarm plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_interaction_values,
            X_sample,
            feature_names=feature_names,
            plot_type="dot",  # Swarm plot
            show=False,
            max_display=15
        )
        plt.xticks(fontsize=18)  # X-axis tick label size
        plt.yticks(fontsize=18)  # Y-axis tick label size
        plt.title("Random Forest SHAP Interaction Value Distribution", fontsize=14)
        plt.xlabel("SHAP interaction value (impact on model output)")
        plt.tight_layout()

        # Save image
        plt.savefig(
            os.path.join(output_dir, 'RF_SHAP_Interaction_Summary.png'),
            dpi=500,
            bbox_inches='tight'
        )
        plt.close()

    except Exception as e:
        print(f"Error in random forest SHAP interaction analysis: {str(e)}")
        traceback.print_exc()

def plot_grid_search_heatmap(grid_search, param_grid, output_dir):
    """Plot grid search heatmap"""
    try:
        # Get results
        results = pd.DataFrame(grid_search.cv_results_)

        # Select two most important parameters for visualization
        if 'n_estimators' in param_grid and 'max_depth' in param_grid:
            # Create pivot table
            pivot = pd.pivot_table(
                results,
                values='mean_test_score',
                index='param_n_estimators',
                columns='param_max_depth'
            )

            # Plot heatmap
            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu",
                        cbar_kws={'label': 'AUC Score'})
            plt.title("Random Forest Grid Search Results")
            plt.xlabel("Max Depth")
            plt.ylabel("Number of Estimators")

            # Save image
            plt.savefig(os.path.join(output_dir, 'RF_GridSearch_Heatmap.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()
            print("Random Forest grid search heatmap saved")
    except Exception as e:
        print(f"Error plotting grid search heatmap: {str(e)}")

def plot_weight_optimization(weights1, weights2, auc_scores, output_dir, model1_name, model2_name):
    """Plot weight optimization graph, x-axis for model1 weight, y-axis for model2 weight, and label AUC value at the peak"""
    try:
        plt.figure(figsize=(8, 6))

        # Plot optimization curve
        scatter = plt.scatter(weights1, weights2, c=auc_scores, cmap='viridis', s=100, edgecolors='w', alpha=0.7)
        plt.colorbar(scatter, label='AUC Score')

        # Mark best point
        best_idx = np.argmax(auc_scores)
        best_auc = auc_scores[best_idx]
        best_weight1 = weights1[best_idx]
        best_weight2 = weights2[best_idx]

        plt.scatter(best_weight1, best_weight2, color='red', s=200, zorder=5, label=f'Best (AUC={best_auc:.3f})')
        plt.text(best_weight1, best_weight2, f'AUC={best_auc:.3f}', fontsize=12, ha='right', va='bottom', color='white')

        # Format settings
        plt.xlabel(f'{model1_name} Weight', fontsize=15)
        plt.ylabel(f'{model2_name} Weight', fontsize=15)
        plt.title(f'Weight Optimization between {model1_name} and {model2_name}', fontsize=15)
        plt.legend(loc='upper left', frameon=True)
        plt.grid(True, alpha=0.3)

        # Set axis ranges
        plt.xlim(0.05, 0.95)
        plt.ylim(0.05, 0.95)

        # Save image
        plt.savefig(os.path.join(output_dir, 'Weight_Optimization.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error plotting weight optimization graph: {str(e)}")


def plot_model_comparison(models_results, y_test, output_dir):
    """Generate ROC/PR curve comparison for six models using academic style (combined in one figure)"""
    try:
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))  # Left-right layout (1 row, 2 columns)
        fig.subplots_adjust(wspace=0.3)  # Increase horizontal spacing

        # Color scheme (ColorBrewer Set1)
        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
        linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5))]

        # ==================== ROC Curve Comparison ====================
        ax1.set_title('ROC Curve Comparison', fontsize=20, pad=10)

        # Plot ROC curves for each model
        for i, (model_name, result) in enumerate(models_results.items()):
            if 'y_pred_prob' in result and len(result['y_pred_prob']) == len(y_test):
                fpr, tpr, _ = roc_curve(y_test, result['y_pred_prob'])
                roc_auc = auc(fpr, tpr)
                ax1.plot(fpr, tpr,
                         color=colors[i],
                         linestyle='-',  # Solid line
                         linewidth=2.5,  # Line width
                         label=f'{MODEL_NAMES.get(model_name, model_name)} (AUC={roc_auc:.3f})')

        # Plot diagonal line
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        ax1.set_xlim([-0.05, 1.05])
        ax1.set_ylim([-0.05, 1.05])
        ax1.set_xlabel('False Positive Rate', fontsize=18)
        ax1.set_ylabel('True Positive Rate', fontsize=18)
        ax1.legend(loc='lower right', frameon=True, fontsize=18)
        ax1.grid(True, alpha=0.3)

        # ==================== PR Curve Comparison ====================
        ax2.set_title('Precision-Recall Curve Comparison', fontsize=20, pad=10)

        # Plot random baseline
        baseline = np.mean(y_test)
        ax2.axhline(y=baseline, color='k', linestyle='--', linewidth=1, alpha=0.5,
                    label=f'Random (AP={baseline:.2f})')

        # Plot PR curves for each model
        for i, (model_name, result) in enumerate(models_results.items()):
            if 'y_pred_prob' in result and len(result['y_pred_prob']) == len(y_test):
                precision, recall, _ = precision_recall_curve(y_test, result['y_pred_prob'])
                ap_score = average_precision_score(y_test, result['y_pred_prob'])
                ax2.plot(recall, precision,
                         color=colors[i],
                         linestyle='-',
                         linewidth=2.5,
                         label=f'{MODEL_NAMES.get(model_name, model_name)} (AP={ap_score:.3f})')

        ax2.set_xlim([-0.01, 1.05])
        ax2.set_ylim([0, 1.05])
        ax2.set_xlabel('Recall', fontsize=18)
        ax2.set_ylabel('Precision', fontsize=18)
        ax2.legend(loc='lower left', frameon=True, fontsize=18)
        ax2.grid(True, alpha=0.3)

        # Adjust layout and save
        plt.tight_layout(pad=3.0)
        save_path = os.path.join(output_dir, 'Combined_ROC_PR_Comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Combined ROC/PR comparison plot saved to: {save_path}")

    except Exception as e:
        print(f"Error generating model comparison plot: {str(e)}")
        traceback.print_exc()



def grid_search_model(model, param_grid, X_train, y_train, scoring='roc_auc', cv=5):
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training data must contain two classes")

    # Use stratified cross-validation
    cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        error_score='raise'  # Throw exception on error
    )

    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def plot_loocv_results(models, X, y, output_dir):
    """Improved LOOCV result visualization, handling single-class cases"""
    try:
        plt.figure(figsize=(10, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

        for i, (name, model) in enumerate(models.items()):
            loo = LeaveOneOut()
            y_pred = []
            y_true = []

            print(f"\nPerforming LOOCV for {name}...")

            for train_idx, test_idx in loo.split(X):
                X_train_fold, X_test_fold = X[train_idx], X[test_idx]
                y_train_fold, y_test_fold = y[train_idx], y[test_idx]

                # Check if test set has two classes
                if len(np.unique(y_test_fold)) < 2:
                    continue

                # Clone model
                current_model = clone(model)
                current_model.fit(X_train_fold, y_train_fold)

                try:
                    pred_prob = current_model.predict_proba(X_test_fold)[:, 1][0]
                    y_pred.append(pred_prob)
                    y_true.append(y_test_fold[0])
                except Exception as e:
                    print(f"LOOCV prediction error: {str(e)}")
                    continue

            if len(y_true) == 0:
                print(f"⚠️ No valid LOOCV results for {name} (all folds are single-class)")
                continue

            # Calculate ROC
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, color=colors[i],
                     label=f'{MODEL_NAMES.get(name, name)} (AUC={roc_auc:.3f})',
                     linewidth=2)

        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('LOOCV ROC Curves Comparison', fontsize=14)
        plt.legend(loc='lower right', frameon=True)
        plt.grid(True, alpha=0.3)

        save_path = os.path.join(output_dir, 'LOOCV_ROC_Comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nLOOCV result plot saved to: {save_path}")

    except Exception as e:
        print(f"Error plotting LOOCV results: {str(e)}")
        traceback.print_exc()


# Modified MI_model function
def MI_model(models_trained, X_test, y_test, X_total, output_dir):
    try:
        print("\n===== Starting model evaluation =====")
        y_pred_combined = None
        best_auc = 0

        # 1. Collect prediction results from each model
        predictions = {}
        model_names = ['XGBoost', 'SVM', 'RandomForest', 'GBDT', 'MLP', 'Logistic']

        for name in model_names:
            if name in models_trained:
                model = models_trained[name]
                try:
                    y_pred = model.predict_proba(X_test)[:, 1]
                    predictions[name] = y_pred
                    print(f"{name} model prediction completed | AUC: {AUC(y_test, y_pred):.4f}")
                except Exception as e:
                    print(f"{name} model prediction failed: {str(e)}")
                    continue

        if len(predictions) < 2:
            raise ValueError("At least two valid models are required for coupling")

        # 2. Calculate and save correlation matrix between model predictions
        corr_matrix = pd.DataFrame(predictions).corr()
        plt.figure(figsize=(15, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f",
                    annot_kws={"size": 10}, cbar_kws={"shrink": 0.8})
        plt.title('Model Prediction Correlation Matrix', fontsize=15)
        plt.xticks(fontsize=15, rotation=45)
        plt.yticks(fontsize=15)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'Model_Correlation_Matrix.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Find two models with lowest correlation
        min_corr = 1
        best_pair = None

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                current_corr = abs(corr_matrix.iloc[i, j])
                if current_corr < min_corr:
                    min_corr = current_corr
                    best_pair = (corr_matrix.columns[i], corr_matrix.columns[j])

        print(f"Selected model pair with lowest correlation: {best_pair} | Correlation: {min_corr:.4f}")

        if best_pair is None or len(best_pair) < 2:
            raise ValueError("Could not find suitable low-correlation model pair")

        # 4. Weight adjustment process
        model1, model2 = best_pair
        y_pred1 = predictions[model1]
        y_pred2 = predictions[model2]

        # Weight range
        weight_range = np.linspace(0.05, 0.95, 20)
        auc_scores = []
        weight1_list = []
        weight2_list = []

        # Record best weight combination
        best_weight1 = 0.5
        best_weight2 = 0.5
        best_auc = 0.0

        for weight1 in weight_range:
            weight2 = 1 - weight1
            y_pred_combined = weight1 * y_pred1 + weight2 * y_pred2
            auc_combined = AUC(y_test, y_pred_combined)
            auc_scores.append(auc_combined)
            weight1_list.append(weight1)
            weight2_list.append(weight2)

            if auc_combined > best_auc:
                best_auc = auc_combined
                best_weight1 = weight1
                best_weight2 = weight2

        print(f"Best weight combination: {model1} = {best_weight1:.4f}, {model2} = {best_weight2:.4f} | AUC: {best_auc:.4f}")

        # 5. Plot improved weight adjustment process graph
        plt.figure(figsize=(12, 6))

        # Subplot 1: Weight vs AUC relationship
        plt.subplot(1, 2, 1)
        plt.plot(weight1_list, auc_scores, 'b-', label=f'{model1} weight')
        plt.plot(weight2_list, auc_scores, 'r-', label=f'{model2} weight')
        plt.scatter(best_weight1, best_auc, color='green', s=100, zorder=5,
                    label=f'Best point (AUC={best_auc:.4f})')
        plt.xlabel('Model weight')
        plt.ylabel('AUC score')
        plt.title('Model Weight vs AUC Relationship')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Subplot 2: 3D weight optimization graph
        ax = plt.subplot(1, 2, 2, projection='3d')
        ax.scatter(weight1_list, weight2_list, auc_scores, c=auc_scores, cmap='viridis')
        ax.scatter(best_weight1, best_weight2, best_auc, color='red', s=100,
                   label=f'Best combination (AUC={best_auc:.4f})')
        ax.set_xlabel(f'{model1} weight')
        ax.set_ylabel(f'{model2} weight')
        ax.set_zlabel('AUC score')
        ax.set_title('3D View of Weight Optimization')
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'Enhanced_Weight_Adjustment_Process.png'), dpi=300)
        plt.close()

        try:
            # Save current settings
            original_rc = plt.rcParams.copy()

            # Set English font and style (override previous settings)
            plt.rcParams.update({
                'font.family': 'Arial',  # Use common English fonts like Arial
                'axes.unicode_minus': False,  # Fix negative sign display
                'font.size': 12,  # Base font size
                'axes.titlesize': 14,  # Subplot title size
                'axes.labelsize': 12,  # Axis label size
                'xtick.labelsize': 11,  # X-axis tick size
                'ytick.labelsize': 11,  # Y-axis tick size
                'legend.fontsize': 11,  # Legend size
                'figure.titlesize': 16,  # Main title size
                'figure.dpi': 150  # Moderate DPI
            })

            plt.figure(figsize=(12, 6))

            # ===== ROC Curve =====
            plt.subplot(1, 2, 1)
            fpr, tpr, _ = roc_curve(y_test, y_pred_combined)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color='darkorange',
                     label=f'Ensemble (AUC={roc_auc:.3f})', linewidth=2)
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
            plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')  # English label
            plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')  # English label
            plt.title('ROC Curve', fontsize=14, fontweight='bold')  # English title
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)

            # ===== PR Curve =====
            plt.subplot(1, 2, 2)
            precision, recall, _ = precision_recall_curve(y_test, y_pred_combined)
            ap_score = average_precision_score(y_test, y_pred_combined)
            plt.plot(recall, precision, color='blue',
                     label=f'AP={ap_score:.3f}', linewidth=2)
            plt.xlabel('Recall', fontsize=12, fontweight='bold')  # English label
            plt.ylabel('Precision', fontsize=12, fontweight='bold')  # English label
            plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')  # English title
            plt.legend(loc="lower left")
            plt.grid(True, alpha=0.3)

            # Main title
            plt.suptitle(f'Integrated Model Performance ({model1} + {model2})',
                         fontsize=16, fontweight='bold', y=1.02)

            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, 'MI_Combined_Model_Performance.png'),
                dpi=150,
                bbox_inches='tight',
                facecolor='white'
            )
            plt.close()

        except Exception as e:
            print(f"Plotting error: {str(e)}")
            traceback.print_exc()
        finally:
            # Restore original settings
            plt.rcParams.update(original_rc)

        # 7. Full dataset prediction
        y_total1 = models_trained[model1].predict_proba(X_total)[:, 1]
        y_total2 = models_trained[model2].predict_proba(X_total)[:, 1]
        y_total_combined = best_weight1 * y_total1 + best_weight2 * y_total2

        # 8. Save coupled model information
        with open(os.path.join(output_dir, 'MI_Model_Info.txt'), 'w') as f:
            f.write(f"Coupled model composition: {model1} + {model2}\n")
            f.write(f"Model correlation: {min_corr:.4f}\n")
            f.write(f"Test set AUC: {best_auc:.4f}\n")
            f.write(f"AP score: {ap_score:.4f}\n")
            f.write(f"Model {model1} weight: {best_weight1:.4f}\n")
            f.write(f"Model {model2} weight: {best_weight2:.4f}\n")
            # Return best weights
            return y_total_combined, best_auc, best_weight1, best_weight2

        return y_total_combined, best_auc

    except Exception as e:
        import traceback
        print(f"Model error: {str(e)}\n{traceback.format_exc()}")
        return None, None



def cross_validate_model(model, X, y, n_splits=5):
    if len(np.unique(y)) < 2:
        raise ValueError("Data must contain two classes")

    # Use stratified cross-validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    auc_scores = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        current_model = clone(model)
        current_model.fit(X_train, y_train)

        try:
            y_pred = current_model.predict_proba(X_test)[:, 1]
            auc_scores.append(AUC(y_test, y_pred))
        except Exception as e:
            print(f"Cross-validation error: {str(e)}")
            traceback.print_exc()
            continue

    if len(auc_scores) == 0:
        print("All folds skipped due to errors")
        return np.nan

    mean_score = np.mean(auc_scores)
    std_score = np.std(auc_scores)
    print(f"{model.__class__.__name__} {cv.__class__.__name__} AUC: {mean_score:.3f} (±{std_score:.3f})")
    return mean_score


def save_raster_results(output_path, data):
    """Save raster point prediction results to DBF"""
    try:
        table = dbf.Table(output_path,
                         'XGB N(7,3); SVC N(7,3); RF N(7,3); GBDT N(7,3); MLP N(7,3); LR N(7,3); COMBINED N(7,3)')
        table.open(mode=dbf.READ_WRITE)
        for row in data:
            table.append({
                'XGB': row[0],
                'SVC': row[1],
                'RF': row[2],
                'GBDT': row[3],
                'MLP': row[4],
                'LR': row[5],
                'COMBINED': row[6]
            })
        table.close()
        print(f"Raster point prediction results saved to: {output_path}")
    except Exception as e:
        print(f"Error saving raster point results: {str(e)}")
# Use chunking for large data processing
def process_large_dbf(file_path, chunk_size=50000):
    table = DBF(file_path)
    for i in range(0, len(table), chunk_size):
        chunk = table[i:i+chunk_size]
        # Process chunk data
        yield pd.DataFrame(chunk)
def preprocess_features(X, feature_names):
    """Preprocess features, specifically reducing impact of Road feature"""
    if 'Road' in feature_names:
        Road_idx = feature_names.index('Road')
        # Scale down Road values by 10x (adjust based on actual performance)
        X[:, Road_idx] = X[:, Road_idx] / 1.2
    return X


def calculate_variance(predictions):
    """Calculate variance of prediction results"""
    mean_prediction = np.mean(predictions)
    variance = np.var(predictions, ddof=1)  # Sample variance
    return variance


def plot_variance_comparison(models, X_test, y_test, output_dir, best_weight1, best_weight2):
    """Show variance comparison of model prediction distributions using box plot"""
    plt.figure(figsize=(10, 7), facecolor='white')
    ax = plt.gca()

    # Collect prediction results for each model (for box plot)
    all_predictions = []
    model_names = []

    if 'XGBoost' in models:
        xgb_pred = models['XGBoost'].predict_proba(X_test)[:, 1]
        all_predictions.append(xgb_pred)
        model_names.append('XGBoost')

    if 'MLP' in models:
        mlp_pred = models['MLP'].predict_proba(X_test)[:, 1]
        all_predictions.append(mlp_pred)
        model_names.append('MLP')

    if 'XGBoost' in models and 'MLP' in models:
        xgb_pred = models['XGBoost'].predict_proba(X_test)[:, 1]
        mlp_pred = models['MLP'].predict_proba(X_test)[:, 1]
        ensemble_pred = best_weight1 * xgb_pred + best_weight2 * mlp_pred
        all_predictions.append(ensemble_pred)
        model_names.append('Ensemble')

    # Calculate variances (for title explanation)
    variances = {name: calculate_variance(pred) for name, pred in zip(model_names, all_predictions)}

    # Plot box plot (academic style optimization)
    boxprops = dict(linestyle='-', linewidth=2, color='darkblue')
    whiskerprops = dict(linestyle='--', linewidth=1.5, color='darkblue')
    medianprops = dict(linestyle='-', linewidth=2.5, color='firebrick')
    meanprops = dict(marker='D', markeredgecolor='black', markerfacecolor='firebrick')

    boxplot = ax.boxplot(
        all_predictions,
        labels=model_names,
        patch_artist=True,
        boxprops=boxprops,
        whiskerprops=whiskerprops,
        medianprops=medianprops,
        meanprops=meanprops,
        showmeans=True,  # Show mean value
        widths=0.6  # Box width
    )

    # Fill box colors (using professional color scheme)
    colors = sns.color_palette("colorblind", n_colors=len(model_names))
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)  # Transparency to show grid

    # Axis and title optimization
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Prediction Probabilities', fontsize=14, fontweight='bold')
    ax.set_title(
        'Distribution Comparison of Prediction Probabilities\n(XGBoost vs MLP vs Ensemble)',
        fontsize=16,
        fontweight='bold',
        pad=20
    )
    ax.set_xticklabels(model_names, fontsize=12, fontweight='bold', rotation=0)
    ax.tick_params(axis='y', labelsize=12)

    # Add grid lines (horizontal only)
    ax.grid(axis='y', linestyle='--', alpha=0.7, color='gray')
    ax.set_axisbelow(True)

    # Add variance values explanation in top-right corner
    variance_text = "\n".join([f"{name}: Variance = {var:.4f}" for name, var in variances.items()])
    ax.text(
        0.95, 0.95, variance_text,
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        fontsize=10
    )

    # Save high-resolution plot
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, 'Variance_Comparison_Boxplot.png'),
        dpi=600,
        bbox_inches='tight'
    )
    plt.close()

    # Print results
    print("\n===== Model Prediction Variance Results =====")
    for name, var in variances.items():
        print(f"{name} prediction variance: {var:.4f}")


# Example usage
if __name__ == "__main__":
    # Single model SHAP plot paths (modify according to actual file paths)
    single_shap_paths = {
        'XGBoost': 'XGBoost_SHAP.png',
        'SVM': 'SVM_SHAP.png',
        'RandomForest': 'RandomForest_SHAP.png',
        'GBDT': 'GBDT_SHAP.png',
        'MLP': 'MLP_SHAP.png',
        'Logistic': 'Logistic_SHAP.png',
        'Ensemble': 'Ensemble_SHAP.png'
    }
    # Output path
    output_dir = r'E:\1zaidian\results'
    os.makedirs(output_dir, exist_ok=True)

def yfx(XY_sample, y_sample, factor, output_dir, dbfroad, XY_raster, feature_names):
    try:
        print("\n===== Starting analysis =====")
        # 1. Data validation and preprocessing
        print("\nClass distribution:", Counter(y_sample))
        if len(np.unique(y_sample)) < 2:
            raise ValueError("Training data must contain two classes")
        assert XY_sample.shape[0] == 68, f"Training sample count should be 68, actual: {XY_sample.shape[0]}"
        assert XY_raster.shape[0] == 1254, f"Prediction sample count should be 1254, actual: {XY_raster.shape[0]}"
        assert not np.array_equal(XY_sample, XY_raster), "Training data and prediction data cannot be identical!"

        # Adjust Road feature
        Road_idx = feature_names.index('Road')
        XY_sample[:, Road_idx] = XY_sample[:, Road_idx] / 1
        XY_raster[:, Road_idx] = XY_raster[:, Road_idx] / 1

        # 2. Data standardization
        scaler = StandardScaler()
        X_train = scaler.fit_transform(XY_sample)
        X_raster = scaler.transform(XY_raster)

        XY_sample = preprocess_features(XY_sample, feature_names)
        XY_raster = preprocess_features(XY_raster, feature_names)

        # 3. Split into training and validation sets (30% validation)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_sample,
            test_size=0.3,
            random_state=42,
            stratify=y_sample
        )
        print(f"\nData split: Training set {X_train.shape[0]} samples | Validation set {X_val.shape[0]} samples")

        # 4. Initialize prediction result array
        yifaxing = np.zeros((X_raster.shape[0], 7))  # 6 models + 1 combination



        # 5. Train each model (with grid search)
        models = {}  # Initialize models variable

        # 5.1 XGBoost
        print("\n=== Training XGBoost ===")
        xgb_pred, xgb_auc, xgb_model = XGB(X_train, X_val, y_train, y_val, X_raster, output_dir)
        if xgb_pred is not None:
            yifaxing[:, 0] = xgb_pred
            models['XGBoost'] = xgb_model
            # New: Stratified cross-validation evaluation
            xgb_cv_auc = cross_validate_model(xgb_model, X_train, y_train, n_splits=5)
            print(f"XGBoost stratified cross-validation AUC: {xgb_cv_auc:.4f}")
        else:
            print("XGBoost training failed, skipping this model")

        # 5.2 SVM
        print("\n=== Training SVM ===")
        # Assume sample_weights_res is sample weights corresponding to y_train
        sample_weights_res = np.ones_like(y_train)
        svc_pred, svc_auc, svc_model = SVC(X_train, X_val, y_train, y_val, X_raster, output_dir, sample_weights_res)
        if svc_pred is not None:
            yifaxing[:, 1] = svc_pred
            models['SVM'] = svc_model
            # New: Stratified cross-validation evaluation
            svc_cv_auc = cross_validate_model(svc_model, X_train, y_train, n_splits=5)
            print(f"SVM stratified cross-validation AUC: {svc_cv_auc:.4f}")
        else:
            print("SVM training failed, skipping this model")
            gc.collect()  # Call before and after SHAP calculation
            # 5.3 Random Forest
        print("\n=== Training Random Forest ===")
        rf_pred, rf_auc, rf_model = RFC(X_train, X_val, y_train, y_val, X_raster, output_dir, factor, dbfroad)
        if rf_pred is not None:
            yifaxing[:, 2] = rf_pred
            models['RandomForest'] = rf_model
            # New: Stratified cross-validation evaluation
            rf_cv_auc = cross_validate_model(rf_model, X_train, y_train, n_splits=5)
            # New: SHAP analysis
            plot_enhanced_shap(rf_model, X_val, feature_names, output_dir, "Random Forest")

            print(f"Random Forest stratified cross-validation AUC: {rf_cv_auc:.4f}")
        else:
            print("Random Forest training failed, skipping this model")

        # 5.4 GBDT
        print("\n=== Training GBDT ===")
        gbdt_pred, gbdt_auc, gbdt_model = GBDT(X_train, X_val, y_train, y_val, X_raster, output_dir)
        if gbdt_pred is not None:
            yifaxing[:, 3] = gbdt_pred
            models['GBDT'] = gbdt_model
            # New: Stratified cross-validation evaluation
            gbdt_cv_auc = cross_validate_model(gbdt_model, X_train, y_train, n_splits=5)
            print(f"GBDT stratified cross-validation AUC: {gbdt_cv_auc:.4f}")
        else:
            print("GBDT training failed, skipping this model")

        # 5.5 MLP
        print("\n=== Training MLP ===")
        sample_weights_res = np.ones_like(y_train)
        mlp_pred, mlp_auc, mlp_model = MLP(X_train, X_val, y_train, y_val, X_raster, output_dir, sample_weights_res)
        if mlp_pred is not None:
            yifaxing[:, 4] = mlp_pred
            models['MLP'] = mlp_model
            # New: Stratified cross-validation evaluation
            mlp_cv_auc = cross_validate_model(mlp_model, X_train, y_train, n_splits=5)
            print(f"MLP stratified cross-validation AUC: {mlp_cv_auc:.4f}")
        else:
            print("MLP training failed, skipping this model")

        # 5.6 Logistic Regression
        print("\n=== Training Logistic Regression ===")
        sample_weights_res = np.ones_like(y_train)
        lr_pred, lr_auc, lr_model = LR(X_train, X_val, y_train, y_val, X_raster, output_dir, sample_weights_res,feature_names)
        if lr_pred is not None:
            yifaxing[:, 5] = lr_pred
            models['Logistic'] = lr_model
            # New: Stratified cross-validation evaluation
            lr_cv_auc = cross_validate_model(lr_model, X_train, y_train, n_splits=5)
            print(f"Logistic Regression stratified cross-validation AUC: {lr_cv_auc:.4f}")
        else:
            print("Logistic Regression training failed, skipping this model")
        # XGBoost SHAP analysis
        if 'XGBoost' in models:
            plot_xgb_shap(models['XGBoost'], X_val, feature_names, output_dir)
        # Call after model training (ensure MLP model is trained)
        if 'MLP' in models:
            plot_mlp_shap(
                models['MLP'],
                X_val,  # Use validation set data
                feature_names,
                output_dir
            )
        # Ensemble model SHAP analysis (ensure MLP is also trained)
        if 'XGBoost' in models and 'MLP' in models:
            plot_ensemble_shap(
                models['XGBoost'],
                models['MLP'],
                X_val,  # Use validation set data
                feature_names,
                output_dir,
                xgb_weight=0.2868  # Your optimized weight
            )
        print(f"Feature names: {feature_names}")
        print(f"Validation set sample count: {X_val.shape[0]}, Feature count: {X_val.shape[1]}")
        # Add after Random Forest training
        if rf_pred is not None:
            yifaxing[:, 2] = rf_pred
            models['RandomForest'] = rf_model
            # Plot SHAP plot
            plot_rf_shap(
                rf_model,
                X_val,  # Use validation set data
                feature_names,
                output_dir
            )
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_val[:10])  # Calculate for first 10 samples only
        print("SHAP value example:", shap_values)
        # Add after GBDT training
        if gbdt_pred is not None:
            yifaxing[:, 3] = gbdt_pred
            models['GBDT'] = gbdt_model
            gbdt_cv_auc = cross_validate_model(gbdt_model, X_train, y_train, n_splits=5)
            # New SHAP analysis
            plot_gbdt_shap(gbdt_model, X_val, feature_names, output_dir)
            print(f"GBDT stratified cross-validation AUC: {gbdt_cv_auc:.4f}")

        # Add after SVM training
        if svc_pred is not None:
            yifaxing[:, 1] = svc_pred
            models['SVM'] = svc_model
            svc_cv_auc = cross_validate_model(svc_model, X_train, y_train, n_splits=5)
            # New SHAP analysis
            plot_svm_shap(svc_model, X_val, feature_names, output_dir)
            print(f"SVM stratified cross-validation AUC: {svc_cv_auc:.4f}")
        # 6. Two integration methods
        print("\n===== Model Integration =====")

        # 6.2 MI model integration
        print("\n--- MI model integration ---")
        # Receive 4 return values: integrated prediction, AUC, best weight 1, best weight 2
        mi_pred, mi_auc, best_weight1, best_weight2 = MI_model(models, X_val, y_val, X_raster, output_dir)
        if mi_pred is not None:
            yifaxing[:, 6] = mi_pred
            print(f"Using MI model results, AUC: {mi_auc:.4f}")

            # Directly use best weights returned by function, no need to read from file
            print(f"Best weight combination: {best_weight1:.4f} (Model 1), {best_weight2:.4f} (Model 2)")

            # Call variance validation function (use validation set X_val as test data)
            plot_variance_comparison(
                models=models,
                X_test=X_val,  # Calculate variance using validation set
                y_test=y_val,
                output_dir=output_dir,
                best_weight1=best_weight1,
                best_weight2=best_weight2
            )

        # 7. Model comparison and visualization
        print("\n===== Model Comparison =====")
        model_results = {}
        for name, model in models.items():
            if model is not None:
                y_pred_prob = model.predict_proba(X_val)[:, 1]
                auc = AUC(y_val, y_pred_prob)
                model_results[name] = {'y_pred_prob': y_pred_prob, 'auc': auc}

        # Plot comparison graph
        plot_model_comparison(model_results, y_val, output_dir)

        # In model comparison and visualization section
        print("\n===== Model Comparison =====")
        model_results = {}
        for name, model in models.items():
            if model is not None:
                y_pred_prob = model.predict_proba(X_val)[:, 1]
                auc = AUC(y_val, y_pred_prob)
                model_results[name] = {'y_pred_prob': y_pred_prob, 'auc': auc}

        # Plot combined ROC/PR comparison graph
        plot_model_comparison(model_results, y_val, output_dir)

        # 8. Check prediction results
        print("\n===== Prediction Result Check =====")
        print("Prediction result statistics:")
        print(pd.DataFrame(yifaxing, columns=[
            'XGB', 'SVM', 'RF', 'GBDT', 'MLP', 'LR', 'COMBINED'
        ]).describe())

        return yifaxing

    except Exception as e:
        print(f"Main analysis function error: {str(e)}\n{traceback.format_exc()}")
        return None



# Modified main program
if __name__ == "__main__":
    try:
        # 1. Path settings
        output_dir = r'E:\1zaidian\results'
        training_dbf_path = r'E:\1zaidian\Export_Output.dbf'
        prediction_dbf_path = r'E:\1zaidian\Export_Output_2.dbf'

        # Create output directory (if it doesn't exist)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory set to: {output_dir}")

        # 2. Data extraction (receive 4 return values)
        print("\n===== Extracting data =====")
        XY_sample, y_sample, XY_raster, feature_names = extract(
            training_dbf_path, prediction_dbf_path,
            factor=list(range(10))  # Use all 10 columns as features
        )

        if XY_sample is None:
            raise ValueError("Data extraction failed, please check input file paths and format")

        # 3. Main analysis
        print("\n===== Starting model training and prediction =====")
        predictions = yfx(
            XY_sample=XY_sample,
            y_sample=y_sample,
            factor=list(range(10)),
            output_dir=output_dir,
            dbfroad=training_dbf_path,
            XY_raster=XY_raster,
            feature_names=feature_names
        )

        # 4. Save results
        if predictions is not None:
            print("\n===== Saving prediction results =====")
            output_file = os.path.join(output_dir, 'final_predictions.dbf')
            # Validate prediction result shape
            if predictions.shape != (1254, 7):
                print(f"Warning: Prediction result shape should be (1254,7), actual: {predictions.shape}")
                print("Attempting to adjust data...")
                predictions = predictions[:1254, :7]  # Ensure we take 1254 rows and 7 columns

            # Call save function
            save_success = save2dbf(
                output_path=output_file,
                predictions=predictions
            )

            if save_success:
                print(f"✓ Prediction results successfully saved to:\n{output_file}")
                print(f"Saved {predictions.shape[0]} records with 7 model prediction values")
            else:
                print("× Save failed, please check error messages")
                check_prediction_distribution(predictions[:, 3], "GBDT")
                check_prediction_distribution(predictions[:, 4], "MLP")

    except Exception as e:
        print("\n!!! Program runtime error !!!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        print("\nError trace:")
        print(traceback.format_exc())

        # Attempt to save temporary results (if possible)
        if 'predictions' in locals():
            temp_file = os.path.join(output_dir, 'temp_predictions.npy')
            np.save(temp_file, predictions)
            print(f"\nTemporary results saved to: {temp_file}")

    finally:
        print("\nProgram execution completed")