# This Python code is a Streamlit application for a Planning & Scheduling Assistant. Here is a
# breakdown of what the code does:

import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from src.ingest import DataIngestor
from src.explore import DataExplorer
from src.ui import UIComponents
from src.model import DelayPredictor
import pickle
from pathlib import Path
from introduction import IntroductionSection


if 'predictor' not in st.session_state:
    if Path("default_predictor.pkl").exists():
        with open("default_predictor.pkl", "rb") as f:
            st.session_state['predictor'] = pickle.load(f)

#st.set_page_config(page_title="Planning & Scheduling Assistant")
st.set_page_config(
     page_title="Planning & Scheduling Assistant",
     layout="wide"  # app full width
)
st.title("📦 Planning & Scheduling Assistant")

# === Step 1: Load and clean data ===
ingestor = DataIngestor('data/sales_data.csv')
df = ingestor.load_and_clean()

# === Step 2: Initialize UI ===
ui = UIComponents()

# === Step 3: Sidebar mode selection ===
mode = st.sidebar.radio("Choose a mode", ['Introduction','Explore Delay Trends', 'Predict Delivery Outcome', 'Ask the Assistant'])


# === Step 4: Route based on selected mode ===
if mode == 'Explore Delay Trends':
    st.subheader("📊 Delay Trend Insights")
    
    # === Apply Filters ===
    filtered_df = ui.render_and_apply_filters(df)

    # === Initialize Explorer with Filtered Data ===
    explorer = DataExplorer(filtered_df)

    # === Display KPIs ===
    st.markdown("### Key Metrics")
    ui.render_kpis(explorer.get_kpis())

    # === Delay Pattern Analysis with Tabs (Left) and Key Insights (Right) ===
    st.markdown("---")
    st.markdown("### Performance Analysis")
    col1, col2 = st.columns([2, 1])

    with col1:
        tabs = st.tabs(["By Zone", "By Product", "By Rep", "By Time"])
        with tabs[0]:
            st.pyplot(explorer.plot_bar_chart("Zone", "Delay Rate by Zone", "Set2"))
        with tabs[1]:
            st.pyplot(explorer.plot_bar_chart("Top Level Type", "Delay Rate by Product Type", "Set3"))
        with tabs[2]:
            st.pyplot(explorer.plot_bar_chart("SC Rep", "Delay Rate by Sales Rep", "Set1"))
        with tabs[3]:
            st.pyplot(explorer.plot_monthly_heatmap())

    with col2:
        st.markdown("### High Value vs Normal Value Orders")
        st.pyplot(explorer.plot_value_category_comparison())


    st.markdown("---")
    st.markdown("### Delay Rate by Selected Features")
    options = st.multiselect("Group by columns:", [
        'Zone', 'Region Zone', 'Top Level Branch', 'Top Level Type',
        'SC Rep', 'Last Next Status', 'is_urgent'
    ], default=['Region Zone'])
    

    if options:
        grouped_df = explorer.get_grouped_delay_rate(options)
        grouped_df.columns = [col.replace("_", " ").title() for col in grouped_df.columns]
        grouped_df["Delay Flag"] = grouped_df["Delay Flag"].round(4)
        st.caption("📊 This table shows the average delay rate (delay_flag) by your selected feature(s).")
        st.dataframe(grouped_df.reset_index(drop=True), use_container_width=True)   

    # === Delay Reasons + Correlation ===
    st.markdown("---")
    st.markdown("### Delay Reasons and Correlation Analysis")
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Delay Reasons")
        st.pyplot(explorer.plot_delay_reasons())
    with col4:
        st.subheader("Correlation Matrix")
        st.pyplot(explorer.plot_correlation_matrix())

    # === Custom Grouping Delay Rate ===
  
   
    # === Additional Visualizations ===
    st.markdown("---")
    st.markdown("### Additional Visualizations")
    st.pyplot(explorer.plot_additional_insights())

    # === Monthly Delay Trends ===
    st.markdown("---")
    st.markdown("### 📈 Monthly Delay Trends")
    # years = sorted(filtered_df['year'].dropna().unique())
    years = ["All"] + sorted(filtered_df['year'].dropna().unique().tolist())

    months = list(range(1, 13))

    col_year, col_month_start, col_month_end = st.columns(3)

    with col_year:
        selected_year = st.selectbox("Select Year", years, index=1)

    with col_month_start:
        start_month = st.selectbox("Start Month", months, index=0)

    with col_month_end:
        end_month = st.selectbox("End Month", months, index=11)
    # Ensure valid range
    if start_month > end_month:
        st.warning("Start month cannot be after end month.")
    else:
        st.pyplot(explorer.plot_delay_and_order_trend(selected_year, start_month, end_month))

        

elif mode == 'Predict Delivery Outcome':
    st.subheader("🔮 Delivery Predictor")
    # --- 1. Feature selection UI ---
    selected_features = ui.render_feature_selector(df)
    # st.write("Selected features:", selected_features)

    # 2. Model selection
    selected_model = ui.render_model_selector()
    # st.write("Selected model:", selected_model)

    # 3. Train/test split selector
    train_size = ui.render_train_test_split_selector()
    # st.write(f"Training set: {int(train_size * 100)}%, Test set: {100 - int(train_size * 100)}%")

    # Get model hyperparameters from the UI
    hyperparams, use_gridsearch, gridsearch_params = ui.render_hyperparams(selected_model)
    # st.write("Hyperparameters:", hyperparams)
    # st.write("GridSearch enabled:", use_gridsearch)
    # if use_gridsearch:
    #     # st.write("Grid search params:", gridsearch_params)

    if st.button("Train Model"):
        predictor = DelayPredictor(
            features=selected_features,
            model_type=selected_model,
            hyperparams=hyperparams,
            use_gridsearch=use_gridsearch,
            gridsearch_params=gridsearch_params,
            train_size=train_size  # (from your slider)
        )
        predictor.fit(df)
        st.session_state['predictor'] = predictor  # <-- Store in session

    # --- Always use the predictor from session_state for prediction ---
    predictor = st.session_state.get('predictor', None)
    if predictor is not None:
        st.markdown("### Predict Delay for a New Order")
        user_input_dict = ui.render_prediction_inputs(predictor.features, df)
        if st.button("Predict Delay Probability"):
            prob, pred = predictor.predict_single(user_input_dict)
            st.success(f"Predicted Delay Probability: {prob:.1%} ({'DELAYED' if pred==1 else 'ON TIME'})")

        # --- Display metrics/plots for latest trained model ---
        metrics = predictor.get_metrics()
        st.markdown("### 🏆 Model Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        col2.metric("Precision", f"{metrics['precision']:.3f}")
        col3.metric("Recall", f"{metrics['recall']:.3f}")
        col4.metric("F1 Score", f"{metrics['f1']:.3f}")
        col5.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")

        st.markdown("#### Confusion Matrix")
        explorer = DataExplorer(df)
        fig = explorer.plot_confusion_matrix(metrics["confusion_matrix"], labels=["Not Delayed", "Delayed"])
        st.pyplot(fig)

        fpr, tpr, thresholds, auc = predictor.get_roc_curve()
        fig = explorer.plot_roc_curve(fpr, tpr, auc)
        st.markdown("#### ROC Curve")
        st.pyplot(fig)

        st.markdown("#### Feature Importance")
        feat_df = predictor.get_feature_importance()
        fig = explorer.plot_feature_importance(feat_df)
        st.pyplot(fig)
    else:
        st.info("Train a model or use the default model to enable prediction.")

elif mode == 'Ask the Assistant':
    st.subheader("💬 Ask the Assistant")

    from src.llm import LLMAssistant

    # Initialize LLM Assistant
    assistant = LLMAssistant(df)

    # User input box
    user_question = st.text_input("Enter your question about the data:")

    if st.button("Submit"):
        with st.spinner("Thinking..."):
            code = assistant.generate_code_from_question(user_question)
            result = assistant.execute_code(code)
            explanation = assistant.explain_result(user_question, result)

        # Display everything
        st.markdown("#### 🔍 Generated Code")
        st.code(code, language="python")

        st.markdown("#### 📊 Result")
        st.write(result)

        st.markdown("#### 💬 Explanation")
        st.write(explanation)

elif mode=='Introduction':
    IntroductionSection.show(df)
