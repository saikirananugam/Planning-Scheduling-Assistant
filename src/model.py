# The `DelayPredictor` class in the `model.py` file handles training, prediction, evaluation, and
# feature importance extraction for a model that predicts delivery time or delay likelihood based on
# historical data.


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score



class DelayPredictor:
    def __init__(self, features, model_type, hyperparams, use_gridsearch, gridsearch_params, train_size):
        self.features = features
        self.model_type = model_type
        self.hyperparams = hyperparams
        self.use_gridsearch = use_gridsearch
        self.gridsearch_params = gridsearch_params
        self.train_size = train_size

        # These will be set later
        self.model = None
        self.encoder = None
        self.scaler = None  # Optional, for LR
        self.trained = False

    def fit(self, df, target_col="delay_flag"):
        self._split_data(df, target_col)
        self._preprocess_features()
        self._build_and_train_model()
        self._predict_test()



    def _split_data(self, df, target_col):
        """
        Select features and split into train/test sets.
        """
        X = df[self.features].copy()
        y = df[target_col].copy()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            train_size=self.train_size,
            stratify=y,
            random_state=42
        )
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        print("Data split complete. Train shape:", self.X_train.shape, "Test shape:", self.X_test.shape)

    def _preprocess_features(self):
        """
        One-hot encode categoricals, keep numerics as-is, concatenate for final matrix.
        """
        import pandas as pd
        # from sklearn.preprocessing import OneHotEncoder

        # Define categorical and numeric features (customize as needed)
        categorical_cols = []
        numeric_cols = []
        for col in self.features:
            if col in ['Zone', 'Region Zone', 'Top Level Branch', 'Top Level Type', 'SC Rep', 'year', 'month', 'week', 'Last Next Status']:
                categorical_cols.append(col)
            elif col == 'is_urgent':
                numeric_cols.append(col)
            else:
                if pd.api.types.is_numeric_dtype(self.X_train[col]):
                    numeric_cols.append(col)
                else:
                    categorical_cols.append(col)

        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoder.fit(self.X_train[categorical_cols])

        X_train_cat = encoder.transform(self.X_train[categorical_cols])
        X_test_cat = encoder.transform(self.X_test[categorical_cols])

        encoded_feature_names = encoder.get_feature_names_out(categorical_cols)
        X_train_cat_df = pd.DataFrame(X_train_cat, columns=encoded_feature_names, index=self.X_train.index)
        X_test_cat_df = pd.DataFrame(X_test_cat, columns=encoded_feature_names, index=self.X_test.index)

        X_train_num = self.X_train[numeric_cols].reset_index(drop=True)
        X_test_num = self.X_test[numeric_cols].reset_index(drop=True)

        X_train_final = pd.concat([X_train_cat_df.reset_index(drop=True), X_train_num], axis=1)
        X_test_final = pd.concat([X_test_cat_df.reset_index(drop=True), X_test_num], axis=1)

        self.X_train_final = X_train_final
        self.X_test_final = X_test_final
        self.encoder = encoder
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols

        print("Encoded train shape:", self.X_train_final.shape)
        print("Encoded test shape:", self.X_test_final.shape)
        print("Categorical columns:", self.categorical_cols)
        print("Numeric columns:", self.numeric_cols)
        
    def _build_and_train_model(self):
        """
        Initialize and train model with or without GridSearchCV.
        """
        if self.use_gridsearch:
            if self.model_type == "Random Forest":
                base_model = RandomForestClassifier(class_weight='balanced', random_state=42)
            elif self.model_type == "Logistic Regression":
                base_model = LogisticRegression(max_iter=500, solver='lbfgs', class_weight='balanced', random_state=42)
            else:
                raise ValueError("Unsupported model type.")

            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=self.gridsearch_params,
                scoring='f1',   # or 'roc_auc'
                cv=3,
                n_jobs=-1
            )
            grid_search.fit(self.X_train_final, self.y_train)
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            print("GridSearchCV best params:", self.best_params)
        else:
            if self.model_type == "Random Forest":
                self.model = RandomForestClassifier(
                    **self.hyperparams,
                    class_weight='balanced',
                    random_state=42
                )
            elif self.model_type == "Logistic Regression":
                self.model = LogisticRegression(
                    **self.hyperparams,
                    max_iter=500,
                    solver='lbfgs',
                    class_weight='balanced',
                    random_state=42
                )
            else:
                raise ValueError("Unsupported model type.")
            self.model.fit(self.X_train_final, self.y_train)
            print(f"{self.model_type} model trained.")

        # Set flag for trained model
        self.trained = True
    
    def _predict_test(self):
        """
        Predict labels and probabilities on the test set. Store results for evaluation.
        """
        if not self.trained:
            raise RuntimeError("Model must be trained before predicting.")
        self.y_pred = self.model.predict(self.X_test_final)
        self.y_pred_proba = self.model.predict_proba(self.X_test_final)[:, 1]
        # For compatibility with later metrics methods
        self.y_test_final = self.y_test
        print("Prediction on test set complete.")
       

    def predict(self, input_dict):
        """
        Takes user input (dict), encodes with same pipeline, returns model prediction and probability.
        """
        # To be implemented
        pass

    def get_metrics(self):
        """
        Returns a dictionary of key metrics for the test set.
        """
        metrics = {
            "accuracy": accuracy_score(self.y_test_final, self.y_pred),
            "precision": precision_score(self.y_test_final, self.y_pred),
            "recall": recall_score(self.y_test_final, self.y_pred),
            "f1": f1_score(self.y_test_final, self.y_pred),
            "roc_auc": roc_auc_score(self.y_test_final, self.y_pred_proba),
            "confusion_matrix": confusion_matrix(self.y_test_final, self.y_pred).tolist(),  # List for Streamlit display
        }
        return metrics


    
    def get_feature_importance(self):
        """
        Returns a DataFrame of feature importances (RF) or coefficients (LR).
        """
        if self.model_type == "Random Forest":
            importances = self.model.feature_importances_
            feat_names = self.X_train_final.columns
            return pd.DataFrame({
                "Feature": feat_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
        elif self.model_type == "Logistic Regression":
            # Only one output (binary)
            importances = self.model.coef_[0]
            feat_names = self.X_train_final.columns
            return pd.DataFrame({
                "Feature": feat_names,
                "Coefficient": importances
            }).sort_values(by="Coefficient", key=abs, ascending=False).reset_index(drop=True)
        else:
            return None

    def get_roc_curve(self):
        """
        Returns fpr, tpr, thresholds, and auc for ROC plotting.
        """
        fpr, tpr, thresholds = roc_curve(self.y_test_final, self.y_pred_proba)
        auc = roc_auc_score(self.y_test_final, self.y_pred_proba)
        return fpr, tpr, thresholds, auc
    
    def predict_single(self, input_dict):
        """
        Predict the probability and class for a single input order.
        input_dict: dict of feature_name -> value
        Returns: (probability of delay, predicted class)
        """
        # 1. Convert input to DataFrame (single row)
        input_df = pd.DataFrame([input_dict])

        # 2. Ensure column order matches training
        # Get categorical and numeric columns in the same order as training
        cat_cols = self.categorical_cols
        num_cols = self.numeric_cols

        # 3. One-hot encode categoricals using trained encoder
        X_cat = self.encoder.transform(input_df[cat_cols])
        X_cat_df = pd.DataFrame(X_cat, columns=self.encoder.get_feature_names_out(cat_cols), index=input_df.index)

        # 4. Numeric features as DataFrame
        X_num_df = input_df[num_cols].reset_index(drop=True)

        # 5. Concatenate encoded categoricals and numerics
        X_final = pd.concat([X_cat_df.reset_index(drop=True), X_num_df], axis=1)

        # 6. Ensure the order of columns matches training set
        X_final = X_final[self.X_train_final.columns]

        # 7. Predict probability and class
        prob = self.model.predict_proba(X_final)[0, 1]
        pred = self.model.predict(X_final)[0]

        return prob, pred

