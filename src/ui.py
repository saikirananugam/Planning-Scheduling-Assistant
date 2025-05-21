# The `UIComponents` class provides methods for rendering interactive Streamlit components such as KPI
# metrics, filter widgets, and feature selection for a Planning & Scheduling Assistant application.
"""
UI helper class for the Planning & Scheduling Assistant.

This module defines UIComponents — a class responsible for rendering 
interactive Streamlit components including:
- KPI metric display
- Filter widgets (year, zone, product type, rep, branch)
- DataFrame filtering logic based on selected filters

Suggested usage:
- Instantiate UIComponents in app.py
- Use `render_and_apply_filters(df)` to get filtered DataFrame
- Use `render_kpis(kpi_dict)` to display dashboard KPIs

This abstraction helps maintain separation of concerns, keeping UI layout 
logic modular and reusable across app views.
"""

import streamlit as st

class UIComponents:
    def __init__(self):
        pass

    def render_kpis(self, kpis: dict):
        """
        Display key metrics using Streamlit.
        """
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Orders", f"{kpis['total_orders']:,}")
        col2.metric("Delay Rate", f"{kpis['delay_rate']}%")
        col3.metric("Avg. Lead Time", f"{kpis['avg_lead_time']} days")
        col4.metric("Avg. Delay Days", f"{kpis['avg_delay_days']} days")

    def render_and_apply_filters(self, df):
        """
        Render UI filter widgets and apply filters to the DataFrame.
        """
        col1, col2, col3, col4, col5 = st.columns(5)
        year = col1.selectbox("Year", ['All'] + sorted(df['year'].dropna().unique()))
        zone = col2.selectbox("Zone", ['All'] + df['Zone'].dropna().unique().tolist())
        prod_type = col3.selectbox("Product Type", ['All'] + df['Top Level Type'].dropna().unique().tolist())
        rep = col4.selectbox("Sales Rep", ['All'] + df['SC Rep'].dropna().unique().tolist())
        branch = col5.selectbox("Branch", ['All'] + df['Top Level Branch'].dropna().unique().tolist())

        if year != 'All':
            df = df[df['year'] == year]
        if zone != 'All':
            df = df[df['Zone'] == zone]
        if prod_type != 'All':
            df = df[df['Top Level Type'] == prod_type]
        if rep != 'All':
            df = df[df['SC Rep'] == rep]
        if branch != 'All':
            df = df[df['Top Level Branch'] == branch]

        return df
    
    def render_feature_selector(self, df):
        """
        Show a multiselect for user to pick features for the ML model.
        Returns a list of selected features.
        """
        # Exclude target, ID columns, and leaky features
        exclude = [
            'delay_flag', 'delay_days', 'ECD', 'Promised Delivery Date', 'ECD Notes',
            'Top Level Order', 'Top Level Line', 'Shipment Number(s)',
            'Top Level Sold To', 'Line Creation', 'Schedule Pick Date', 'Supply Item Description','Top Level Item','TL SO Alert'
        ]
        # Only keep columns that are not in exclude
        candidate_features = [col for col in df.columns if col not in exclude]

        selected_features = st.multiselect(
            "Select features to use for prediction:",
            options=candidate_features,
            default=['Zone', 'Region Zone', 'Top Level Branch', 'Top Level Type', 'Line Amount', 'SRP1','Last Next Status','SC Rep','lead_time','is_urgent','total_units']  # Or your own best defaults
        )
        return selected_features
    
    def render_model_selector(self):
        """
        Show a dropdown to select the ML model.
        Returns selected model type as a string.
        """
        model_type = st.selectbox(
            "Select Model:",
            options=["Random Forest", "Logistic Regression"]  
        )
        return model_type
    
    def render_train_test_split_selector(self):
        """
        Show a slider for user to pick train/test split ratio.
        Returns the train size (as a float between 0 and 1).
        """
        train_size = st.slider(
            "Select Training Set Percentage:",
            min_value=50, max_value=90, value=80, step=5,
            help="Choose what % of data goes to training. The rest will be used for testing."
        )
        return train_size / 100.0  # Return as fraction
    
    def render_hyperparams(self, model_type):
        """
        Show hyperparameter controls based on model selection.
        Returns: dict of hyperparameters, gridsearch flag, gridsearch params (if any)
        """
        st.markdown("#### Hyperparameter Selection")

        use_gridsearch = st.checkbox("Use GridSearchCV for hyperparameter tuning?", value=False)

        hyperparams = {}
        gridsearch_params = {}

        if model_type == "Random Forest":
            if not use_gridsearch:
                n_estimators = st.slider("n_estimators", 10, 300, 100, step=10)
                max_depth = st.selectbox("max_depth", [None, 5, 10, 15, 20], index=0)
                min_samples_split = st.slider("min_samples_split", 2, 10, 2)
                hyperparams.update({
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split
                })
            else:
                n_estimators_grid = st.text_input("n_estimators grid (comma-separated)", "100,200,300")
                max_depth_grid = st.text_input("max_depth grid (comma-separated)", "None,10,20")
                min_samples_split_grid = st.text_input("min_samples_split grid (comma-separated)", "2,5,10")
                gridsearch_params = {
                    "n_estimators": [int(x) for x in n_estimators_grid.split(",")],
                    "max_depth": [None if x.strip() == "None" else int(x) for x in max_depth_grid.split(",")],
                    "min_samples_split": [int(x) for x in min_samples_split_grid.split(",")]
                }

        elif model_type == "Logistic Regression":
            if not use_gridsearch:
                c_value = st.number_input("C (Inverse regularization strength)", 0.001, 10.0, 1.0)
                penalty = st.selectbox("Penalty", ["l2", "none"])
                hyperparams.update({
                    "C": c_value,
                    "penalty": penalty
                })
            else:
                c_grid = st.text_input("C grid (comma-separated)", "0.01,0.1,1,10")
                penalty_grid = st.text_input("Penalty grid (comma-separated)", "l2,none")
                gridsearch_params = {
                    "C": [float(x) for x in c_grid.split(",")],
                    "penalty": [x.strip() for x in penalty_grid.split(",")]
                }

        return hyperparams, use_gridsearch, gridsearch_params

    def render_prediction_inputs(self, feature_list, df):
        """
        Dynamically render input widgets for each feature in feature_list.
        Uses training data (df) to find available options for categoricals.
        Returns: dict of {feature_name: value}
        """
        st.markdown("### Enter New Order Details for Prediction")
        user_input = {}
        for feature in feature_list:
            # If the column is categorical (dtype=object or few unique values), use selectbox
            if df[feature].dtype == 'object' or df[feature].nunique() < 20:
                options = sorted(df[feature].dropna().unique())
                user_input[feature] = st.selectbox(f"{feature}", options)
            else:
                # Use reasonable min/max from data, or fallback values
                min_val = float(df[feature].min()) if not df[feature].isnull().all() else 0.0
                max_val = float(df[feature].max()) if not df[feature].isnull().all() else 10000.0
                default_val = float(df[feature].mean()) if not df[feature].isnull().all() else 0.0
                user_input[feature] = st.number_input(
                    f"{feature}", min_value=min_val, max_value=max_val, value=default_val
                )
        return user_input


