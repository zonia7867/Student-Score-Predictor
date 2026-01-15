import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
st.set_page_config(
    page_title="Student Marks Prediction Dashboard",
    layout="wide"
)
st.markdown("""
    <style>
    .main {
        background-color: #FFFFFF;
    }
    h1, h2, h3 {
        color: #1A4E8C;
    }
    .stButton>button {
        background-color: #A23D57;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def load_and_clean_data(file_path):
    try:
        sheets = pd.read_excel(file_path, sheet_name=None)
    except FileNotFoundError:
        return None, "File not found."

    cleaned = []
    for i, (sheet_name, df) in enumerate(sheets.items(), start=1):
        drop_cols = [c for c in df.columns if 'unnamed' in c.lower()]
        df = df.drop(columns=drop_cols, errors='ignore')
        df['sheet_id'] = i
        for col in df.columns:
            if col != 'sheet_id':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        cleaned.append(df)

    data = pd.concat(cleaned, ignore_index=True)
    data = data.fillna(0)

    df_clean = data[~data.astype(str).apply(lambda x: x.str.contains('Weightage', case=False)).any(axis=1)].copy()
    
    cols_to_exclude = ['Student_ID', 'Sheet_ID']
    for col in df_clean.columns:
        if col not in cols_to_exclude:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    df_clean.fillna(0, inplace=True)
    
    return df_clean, None

def plot_workflow():
    plt.rcParams["figure.figsize"] = (5, 8)
    fig, ax = plt.subplots()
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 14)
    ax.axis("off")

    def box(x, y, w, h, text):
        rect = FancyBboxPatch((x, y), w, h,
                            boxstyle="round,pad=0.3",
                            linewidth=1.5,
                            edgecolor="black",
                            facecolor="#097857")
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text,
                ha="center", va="center", fontsize=9, color='white', weight='bold')

    def arrow_between(y_top, y_bottom, x=2):
        arr = FancyArrowPatch((x, y_top), (x, y_bottom),
                            arrowstyle="->", linewidth=1.5,
                            mutation_scale=15, color="black")
        ax.add_patch(arr)

    x = 0.8
    w = 2.4
    h = 0.7
    ys = [13.0, 11.0, 9.0, 7.0, 5.0, 3.0, 1.0]
    steps = [
        "Data Loading",
        "Data Cleaning",
        "Feature Engineering",
        "Train/Test Split\n(80% / 20%)",
        "Model Training",
        "Bootstrapping\n(500 samples)",
        "Evaluation &\nVisualization",
    ]
    for y, label in zip(ys, steps):
        box(x, y, w, h, label)
    for i in range(len(ys) - 1):
        y_top = ys[i]
        y_bottom = ys[i+1] + h
        arrow_between(y_top, y_bottom, x + w/2)
    
    st.pyplot(fig)

mid1, mid2, final = "S-I", "S-II", "Final"
a1, a2, a3, a4, a5, a6 = "As:1", "As:2", "As:3", "As:4", "As:5", "As:6"
q1, q2, q3, q4, q5, q6, q7, q8 = "Qz:1", "Qz:2", "Qz:3", "Qz:4", "Qz:5", "Qz:6", "Qz:7", "Qz:8"

rq_configs = {
    "RQ1: Predict Midterm 1": {
        "x_cols": [a1, a2, q1, q2, q3],
        "y_col": mid1,
        "simple_idx": 2 
    },
    "RQ2: Predict Midterm 2": {
        "x_cols": [a3, a4, q4, q5, q6, mid1],
        "y_col": mid2,
        "simple_idx": 5 
    },
    "RQ3: Predict Final": {
        "x_cols": [a1, a2, a3, a4, a5, a6, q1, q2, q3, q4, q5, q6, q7, q8, mid1, mid2],
        "y_col": final,
        "simple_idx": 15 
    }
}


def run_analysis(df, rq_name):
    config = rq_configs[rq_name]
    X_df = df[config["x_cols"]].copy()
    y = df[config["y_col"]]
    simple_feature_index = config["simple_idx"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )
    
    models = {
        "Dummy (Baseline)": DummyRegressor(strategy="mean"),
        "Simple Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Multiple Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
    }

    results = []
    progress_bar = st.progress(0)
    
    model_items = list(models.items())
    total_steps = len(model_items) * 500
    current_step = 0

    for idx, (name, model) in enumerate(model_items):
        mae_scores = []
        
        for i in range(500):
            X_boot, y_boot = resample(X_train, y_train, random_state=i)
            
            if name == "Simple Linear Regression":
                X_boot_used = X_boot.iloc[:, [simple_feature_index]].values
                X_test_used = X_test.iloc[:, [simple_feature_index]].values
            elif name == "Dummy (Baseline)":
                X_boot_used = X_boot.values
                X_test_used = X_test.values
            else:
                X_boot_used = X_boot.values
                X_test_used = X_test.values

            model.fit(X_boot_used, y_boot)
            y_pred = model.predict(X_test_used)
            mae = mean_absolute_error(y_test, y_pred)
            mae_scores.append(mae)
            

        lower_ci = np.percentile(mae_scores, 2.5)
        upper_ci = np.percentile(mae_scores, 97.5)
        mean_mae = np.mean(mae_scores)

        if name == "Simple Linear Regression":
            X_train_used = X_train.iloc[:, [simple_feature_index]].values
            X_test_used = X_test.iloc[:, [simple_feature_index]].values
        else:
            X_train_used = X_train.values
            X_test_used = X_test.values

        model.fit(X_train_used, y_train)
        y_pred_final = model.predict(X_test_used)
        y_train_pred = model.predict(X_train_used)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred_final))
        r2 = r2_score(y_test, y_pred_final)
        train_r2 = r2_score(y_train, y_train_pred)

        results.append({
            "Model": name,
            "MAE (Mean)": round(mean_mae, 2),
            "95% CI (MAE)": f"[{round(lower_ci, 2)}, {round(upper_ci, 2)}]",
            "RMSE": round(rmse, 2),
            "Test R2": round(r2, 2),
            "Train R2": round(train_r2, 2),
        })
    
    progress_bar.empty()
    return pd.DataFrame(results)

def main():
    st.sidebar.title("Menu")
    options = ["Home", "Data & EDA", "Model Analysis", "Predict Marks"]
    choice = st.sidebar.radio("Go to", options)

    uploaded_file = "marks_dataset.xlsx"
    
    df_clean, error = load_and_clean_data(uploaded_file)
    
    if df_clean is None:
        st.sidebar.error(f"Could not load '{uploaded_file}'. Please upload dataset.")
        uploaded_user_file = st.sidebar.file_uploader("Upload marks_dataset.xlsx", type=['xlsx'])
        if uploaded_user_file:
            df_clean, error = load_and_clean_data(uploaded_user_file)
    if df_clean is None:
        st.warning("Please upload a dataset to proceed.")
        return
    if choice == "Home":
        st.title(" \t\t\tStudent Performance Analytics")
        st.markdown("""
        Welcome to the Assessment Analytics Dashboard. This tool analyzes student performance across 
        Assignments, Quizzes, and Exams to predict future outcomes.
        """)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.info("### Project Scope")
            st.markdown("""
            * **RQ1:** Predict Midterm I scores.
            * **RQ2:** Predict Midterm II scores.
            * **RQ3:** Predict Final Exam scores.
            """)
        with col2:
             st.success(f"**Dataset Loaded:** {df_clean.shape[0]} Rows, {df_clean.shape[1]} Columns")
        st.divider()
        st.subheader("\n\nProject Workflow")
        st.markdown("The following diagram illustrates the data processing and modeling pipeline:")
        plot_workflow()
    elif choice == "Data & EDA":
        st.title("\t\t\tExploratory Data Analysis")
        tab1, tab2, tab3 = st.tabs(["Dataset View", "Distributions", "Correlations"])
        
        with tab1:
            st.write("### Preprocessed Data (First 10 rows)")
            st.dataframe(df_clean.head(10))
            st.write("### Summary Statistics")
            st.dataframe(df_clean.describe().T)

        with tab2:
            st.write("### Target Variable Distributions")
            targets = ['S-I', 'S-II', 'Final']
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            for i, col in enumerate(targets):
                if col in df_clean.columns:
                    sns.histplot(df_clean[col], kde=True, bins=15, color='skyblue', ax=ax[i])
                    ax[i].set_title(f'Distribution of {col}')
            plt.tight_layout()
            st.pyplot(fig)

        with tab3:
            st.write("### Feature Correlations")
            numeric_df = df_clean.select_dtypes(include=['float64', 'int64'])
            correlation_matrix = numeric_df.corr()
            
            fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5, ax=ax_corr)
            st.pyplot(fig_corr)
            
            st.write("#### Top Predictors for Final Exam")
            if 'Final' in correlation_matrix.columns:
                st.dataframe(correlation_matrix['Final'].sort_values(ascending=False).head(10))

    elif choice == "Model Analysis":
        st.title(" \t\t\tModel Training & Evaluation")
        st.markdown("Select a Research Question to train models (Bootstrap n=500).")
        
        rq_selection = st.selectbox("Select Research Question", list(rq_configs.keys()))
        
        if st.button("Train & Evaluate Models"):
            with st.spinner("Training models and performing bootstrapping... this may take a moment."):
                results_df = run_analysis(df_clean, rq_selection)
            
            st.subheader(f"Results for {rq_selection}")
            st.table(results_df)
            
            
            best_model = results_df.sort_values(by="MAE (Mean)").iloc[0]
            st.success(f"**Best Model:** {best_model['Model']}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Best MAE", best_model['MAE (Mean)'])
            with col2:
                st.metric("Best Test R2", best_model['Test R2'])
                
            if best_model["Train R2"] > best_model["Test R2"] + 0.15:
                st.warning("**Interpretation:** Potential Overfitting detected (Train R2 >> Test R2).")
            else:
                st.info("**Interpretation:** Model generalizes well (Balanced Train/Test scores).")

    
    elif choice == "Predict Marks":
        st.title("\t\t\tMake prediction")
        
        rq_pred = st.selectbox("What do you want to predict?", list(rq_configs.keys()))
        config = rq_configs[rq_pred]
        
        st.subheader("Enter Student Scores:")
        
        
        inputs = {}
        cols = st.columns(3)
        for i, col_name in enumerate(config["x_cols"]):
            with cols[i % 3]:
                inputs[col_name] = st.number_input(f"{col_name}", min_value=0.0, max_value=130.0, value=0.0)
        
        if st.button("Predict Score"):
            
            X_input = pd.DataFrame([inputs])
            model = make_pipeline(StandardScaler(), LinearRegression())
            X_full = df_clean[config["x_cols"]]
            y_full = df_clean[config["y_col"]]
            
            model.fit(X_full, y_full)
            prediction = model.predict(X_input)[0]
            
            st.divider()
            st.markdown(f"### Predicted {config['y_col']}:")
            st.write(f"<h1 style='color: #097857; font-size: 50px;'>{prediction:.2f}</h1>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()