import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 
                             recall_score, f1_score, roc_curve, auc, 
                             mean_absolute_error, mean_squared_error, r2_score)
import seaborn as sns
from imblearn.over_sampling import SMOTE
import joblib

def main():
    print("--- Reading Data ---")
    df = pd.read_csv("titanic.csv")
    
    # Target: survived
    # Features: pclass, sex, age, sibsp, parch, fare, embarked
    # Drop columns with high missing rates or not useful for prediction from raw
    features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    X = df[features]
    y = df['survived']
    
    print("\n--- Task 7: Data Splitting ---")
    # Stratified split given class imbalance (more died than survived)
    print("Class balance:")
    print(y.value_counts(normalize=True))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print("Train size:", X_train.shape, "Test size:", X_test.shape)
    
    print("\n--- Task 8: Preprocessing ---")
    num_cols = ['age', 'fare', 'sibsp', 'parch']
    cat_cols = ['sex', 'embarked', 'pclass'] # treating pclass as categorical or numeric, let's do categorical for one-hot
    
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])
    
    print("Preprocessor defined with ColumnTransformer.")
    
    print("\n--- Task 9 & 10: Model Training & Evaluation ---")
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=4),
        'Random Forest': RandomForestClassifier(random_state=42)
    }
    
    results = {}
    for name, estimator in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', estimator)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1, 'AUC': roc_auc}
        
        print(f"\n{name}:")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        if name == 'Decision Tree':
            plt.figure(figsize=(20, 10))
            # get feature names after preprocessing
            cat_features = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(cat_cols)
            all_features = num_cols + list(cat_features)
            plot_tree(pipeline.named_steps['classifier'], feature_names=all_features, class_names=['Died', 'Survived'], filled=True, rounded=True)
            plt.title("Decision Tree Visualization")
            plt.savefig('decision_tree.png')
            plt.close()
            
    print("\nModel Comparison:")
    results_df = pd.DataFrame(results).T
    print(results_df)
    
    print("\n--- Task 11: Imbalance Handling ---")
    print("Baseline Class Balance:")
    print(y_train.value_counts())
    
    rf_base = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42))])
    rf_base.fit(X_train, y_train)
    y_pred_base = rf_base.predict(X_test)
    
    rf_bal = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))])
    rf_bal.fit(X_train, y_train)
    y_pred_bal = rf_bal.predict(X_test)
    
    # SMOTE
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_processed, y_train)
    
    rf_smote = RandomForestClassifier(random_state=42)
    rf_smote.fit(X_train_sm, y_train_sm)
    y_pred_sm = rf_smote.predict(X_test_processed)
    
    print("\nImbalance Handling Comparison (Random Forest):")
    print("Baseline:")
    print(f"Precision: {precision_score(y_test, y_pred_base):.4f}, Recall: {recall_score(y_test, y_pred_base):.4f}, F1: {f1_score(y_test, y_pred_base):.4f}")
    print("Class Weight Balanced:")
    print(f"Precision: {precision_score(y_test, y_pred_bal):.4f}, Recall: {recall_score(y_test, y_pred_bal):.4f}, F1: {f1_score(y_test, y_pred_bal):.4f}")
    print("SMOTE:")
    print(f"Precision: {precision_score(y_test, y_pred_sm):.4f}, Recall: {recall_score(y_test, y_pred_sm):.4f}, F1: {f1_score(y_test, y_pred_sm):.4f}")
    
    print("\n--- Task 12: Hyperparameter Tuning ---")
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 5, 10],
        'classifier__max_features': ['sqrt', 'log2']
    }
    
    grid = GridSearchCV(
        Pipeline([('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42, oob_score=True))]),
        param_grid, cv=5, scoring='f1', n_jobs=-1
    )
    grid.fit(X_train, y_train)
    
    print("Best params:", grid.best_params_)
    print("Best F1 score from CV:", grid.best_score_)
    
    best_rf = grid.best_estimator_
    oob_score = best_rf.named_steps['classifier'].oob_score_
    print(f"OOB Score of the best model: {oob_score:.4f}")
    
    print("\n--- Task 13: Regression Side-Task ---")
    # Predict 'fare'
    df_reg = df.dropna(subset=['fare']) # although in titanic none are usually missing, just in case
    X_reg = df_reg[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'embarked']]
    y_reg = df_reg['fare']
    
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    reg_preprocessor = ColumnTransformer(transformers=[
        ('num', SimpleImputer(strategy='median'), ['age', 'sibsp', 'parch']),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), ['sex', 'embarked', 'pclass', 'survived'])
    ])
    
    reg_pipeline = Pipeline([('preprocessor', reg_preprocessor), ('regressor', LinearRegression())])
    reg_pipeline.fit(Xr_train, yr_train)
    yr_pred = reg_pipeline.predict(Xr_test)
    
    mae = mean_absolute_error(yr_test, yr_pred)
    rmse = np.sqrt(mean_squared_error(yr_test, yr_pred))
    r2 = r2_score(yr_test, yr_pred)
    n = len(yr_test)
    p = Xr_train.shape[1] # number of features before encoding. using rough p
    # Actually p is the number of features after encoding
    p = reg_pipeline.named_steps['regressor'].coef_.shape[0]
    adj_r2 = 1 - (1-r2)*(n-1)/(n-p-1)
    
    print(f"\nRegression Metrics:\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR2: {r2:.2f}\nAdj R2: {adj_r2:.2f}")
    
    residuals = yr_test - yr_pred
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=yr_pred, y=residuals, alpha=0.6)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Predicted Fare')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.savefig('residual_plot.png')
    plt.close()
    
    print("\n--- Task 15: Save Pipeline ---")
    joblib.dump(best_rf, 'best_pipeline.pkl')
    print("Saved 'best_pipeline.pkl'")
    
    # Reload and test
    loaded_pipeline = joblib.load('best_pipeline.pkl')
    test_pred = loaded_pipeline.predict(X_test.head(5))
    print("Predictions on 5 rows from reloaded pipeline:", test_pred)

if __name__ == '__main__':
    main()
