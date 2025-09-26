import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib


# ===== 1. Initial Configuration =====
st.set_page_config(
    page_title="Bank Marketing EDA", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom color palette
custom_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
sns.set_palette(custom_colors)
plt.style.use('seaborn-v0_8-whitegrid')

# ===== 2. Load dataset =====
data_path = "data/bank-full.csv"
df = pd.read_csv(data_path, sep=";")

# ===== 3. Sidebar / Control Panel =====
st.sidebar.title("🎛️ Control Panel")
st.sidebar.markdown("---")

# Sidebar - Dataset Info
st.sidebar.subheader("📄 Dataset Information")
st.sidebar.info(f"""
**Total records:** {len(df):,}  
**Variables:** {len(df.columns)}  
**Period:** Bank marketing data
""")

# Sidebar - Section controls
st.sidebar.subheader("🎯 Section Controls")
st.sidebar.markdown("*Available only in 'Exploratory Analysis'*")
show_preview = st.sidebar.checkbox("Show preview", value=True)
show_kpis = st.sidebar.checkbox("Show KPIs", value=True)
show_filters = st.sidebar.checkbox("Show advanced filters", value=True)
show_charts = st.sidebar.checkbox("Show interactive charts", value=True)

# Sidebar - Filters
if show_filters:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Advanced Filters")
    
    # Age range filter
    min_age, max_age = st.sidebar.slider(
        "Age range", 
        int(df["age"].min()), 
        int(df["age"].max()), 
        (20, 60),
        help="Filter clients by age range"
    )
    df_filtered = df[(df["age"] >= min_age) & (df["age"] <= max_age)]

    # Dynamic categorical filter
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    cat_cols = [c for c in cat_cols if c != "y"]  # exclude target variable
    filter_col = st.sidebar.selectbox(
        "Categorical variable:", 
        cat_cols,
        help="Select a variable to filter"
    )

    options = df[filter_col].unique().tolist()
    selected_opts = st.sidebar.multiselect(
        f"Values of {filter_col}:", 
        options, 
        default=options,
        help=f"Select values of {filter_col} to include"
    )

    df_filtered = df_filtered[df_filtered[filter_col].isin(selected_opts)]
else:
    df_filtered = df

# ===== 4. Main Title =====
st.title("📊 Bank Marketing - Exploratory Analysis and Prediction")
st.markdown("### *Discover patterns and make predictions on bank marketing data*")

# ===== 5. MAIN TABS =====
tab1, tab2 = st.tabs(["📈 Exploratory Analysis", "🔮 Prediction"])

# =================== TAB 1: EXPLORATORY ANALYSIS ===================
with tab1:
    st.markdown("---")
    
    # Dataset preview
    if show_preview:
        with st.expander("📋 Dataset Preview", expanded=False):
            st.dataframe(df.head(), use_container_width=True)
            if show_filters and len(df_filtered) != len(df):
                st.info(f"✅ Filtered dataset: {len(df_filtered):,} records out of {len(df):,} total")

    # KPIs
    if show_kpis:
        st.subheader("🎯 Key Metrics")
        
        data_for_kpis = df_filtered if show_filters else df
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📊 Total Records", 
                f"{len(data_for_kpis):,}",
                help="Total number of records in the dataset"
            )
        
        with col2:
            yes_count = data_for_kpis["y"].value_counts().get("yes", 0)
            st.metric(
                "✅ Accepted", 
                f"{yes_count:,}",
                help="Clients who accepted the offer"
            )
        
        with col3:
            no_count = data_for_kpis["y"].value_counts().get("no", 0)
            st.metric(
                "❌ Rejected", 
                f"{no_count:,}",
                help="Clients who rejected the offer"
            )
        
        with col4:
            acceptance_rate = round((data_for_kpis["y"].value_counts(normalize=True).get("yes", 0))*100, 2)
            st.metric(
                "📈 Acceptance Rate", 
                f"{acceptance_rate}%",
                help="Offer acceptance percentage"
            )
        
        st.markdown("---")

    # Interactive Visualizations
    if show_charts:
        st.subheader("📈 Interactive Visualizations")
        
        # Graph type selector
        graph_type = st.selectbox(
            "Select chart type",
            ["Categorical distribution", "Numeric histogram", "Scatterplot", "Boxplot", "Correlation heatmap"],
            help="Choose the type of visualization to display"
        )
        
        plot_data = df_filtered if show_filters else df

        # Chart 1: Categorical distribution
        if graph_type == "Categorical distribution":
            cat_cols = plot_data.select_dtypes(include="object").columns.tolist()
            col = st.selectbox("Categorical variable:", cat_cols)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(data=plot_data, x=col, hue="y", ax=ax, palette=custom_colors[:2])
            plt.xticks(rotation=45, ha='right')
            plt.title(f'Distribution of {col} vs Response', fontsize=16, pad=20)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Chart 2: Numeric histogram
        elif graph_type == "Numeric histogram":
            num_cols = plot_data.select_dtypes(exclude="object").columns.tolist()
            col = st.selectbox("Numeric variable:", num_cols)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(plot_data[col], kde=True, bins=30, ax=ax, color=custom_colors[0])
            plt.title(f'Distribution of {col}', fontsize=16, pad=20)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Chart 3: Scatterplot
        elif graph_type == "Scatterplot":
            num_cols = plot_data.select_dtypes(exclude="object").columns.tolist()
            col1_scatter, col2_scatter = st.columns(2)
            
            with col1_scatter:
                x_var = st.selectbox("X-axis:", num_cols)
            with col2_scatter:
                y_var = st.selectbox("Y-axis:", num_cols)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=plot_data, x=x_var, y=y_var, hue="y", ax=ax, alpha=0.6, palette=custom_colors[:2])
            plt.title(f'{x_var} vs {y_var}', fontsize=16, pad=20)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Chart 4: Boxplot
        elif graph_type == "Boxplot":
            num_cols = plot_data.select_dtypes(exclude="object").columns.tolist()
            y_var = st.selectbox("Numeric variable:", num_cols)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=plot_data, x="y", y=y_var, ax=ax, palette=custom_colors[:2])
            plt.title(f'{y_var} distribution by Response', fontsize=16, pad=20)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Chart 5: Correlation heatmap
        elif graph_type == "Correlation heatmap":
            num_cols = plot_data.select_dtypes(exclude="object").columns.tolist()
            corr = plot_data[num_cols].corr()
            
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, center=0)
            plt.title('Correlation Matrix', fontsize=16, pad=20)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("---")
    
    st.info("💡 Use the sidebar to control what sections to show and apply filters")

# =================== TAB 2: PREDICTION ===================
with tab2:
    st.markdown("---")
    
    # Prediction information
    st.subheader("🔮 Marketing Response Prediction")
    st.markdown("""
    ### How does it work?
    
    This system uses a **machine learning model** trained to predict if a client 
    will accept or reject a bank marketing campaign.
    
    **📝 Instructions:**
    1. Fill out the form with client data
    2. Click 'Predict'
    3. Get instant results
    
    ---
    """)
# Load pipeline and display form
    try:
        pipeline = joblib.load("models/decision_tree_pipeline.pkl")

        # Display model metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "🎯 Model Used",
                "Decision Tree",
                help="Decision tree algorithm"
            )

        with col2:
            st.metric(
                "📊 Status",
                "✅ Loaded",
                help="The model is ready to use"
            )

        with col3:
            st.metric(
                "🔄 Version",
                "v1.0",
                help="Current model version"
            )

        st.markdown("---")

        # ===== ENHANCED FORM IN COLUMNS =====
        st.subheader("📋 Prediction Form")

        with st.form("prediction_form"):
            # Row 1: Personal Information
            st.markdown("**👤 Personal Information**")
            col1, col2, col3 = st.columns(3)

            with col1:
                age = st.number_input(
                    "Age", 
                    min_value=18, 
                    max_value=100, 
                    value=35,
                    help="Client's age in years"
                )

            with col2:
                job_options = df['job'].unique().tolist()
                job = st.selectbox(
                    "Job", 
                    job_options,
                    help="Client's occupation"
                )

            with col3:
                marital_options = df['marital'].unique().tolist()
                marital = st.selectbox(
                    "Marital Status", 
                    marital_options,
                    help="Client's marital status"
                )

            st.markdown("---")

            # Row 2: Education and Financial Info
            st.markdown("**🎓 Education and Financial Information**")
            col1, col2, col3 = st.columns(3)

            with col1:
                education_options = df['education'].unique().tolist()
                education = st.selectbox(
                    "Education", 
                    education_options,
                    help="Client's education level"
                )

            with col2:
                balance = st.number_input(
                    "Balance", 
                    value=0,
                    help="Client's bank account balance"
                )

            with col3:
                default_options = df['default'].unique().tolist()
                default = st.selectbox(
                    "Default", 
                    default_options,
                    help="Does the client have credit defaults?"
                )

            st.markdown("---")

            # Row 3: Contact Information
            st.markdown("**📞 Contact Information**")
            col1, col2, col3 = st.columns(3)

            with col1:
                housing_options = df['housing'].unique().tolist()
                housing = st.selectbox(
                    "housing", 
                    housing_options,
                    help="Does the client have housing?"
                )

            with col2:
                loan_options = df['loan'].unique().tolist()
                loan = st.selectbox(
                    "Personal Loan", 
                    loan_options,
                    help="Does the client have a personal loan?"
                )

            with col3:
                contact_options = df['contact'].unique().tolist()
                contact = st.selectbox(
                    "Contact Type", 
                    contact_options,
                    help="Communication method used"
                )

            st.markdown("---")

            # Row 4: Campaign Information
            st.markdown("**📈 Campaign Information**")
            col1, col2, col3 = st.columns(3)

            with col1:
                duration = st.number_input(
                    "Duration (sec)", 
                    min_value=0, 
                    value=200,
                    help="Duration of the last call in seconds"
                )

            with col2:
                campaign = st.number_input(
                    "Number of Contacts", 
                    min_value=1, 
                    value=2,
                    help="Number of contacts in this campaign"
                )

            with col3:
                pdays = st.number_input(
                    "Days since last contact", 
                    min_value=-1, 
                    value=-1,
                    help="Days since last contact (-1 if never contacted)"
                )

            st.markdown("---")

            # Row 5: Temporal Information
            st.markdown("**📅 Temporal Information**")
            col1, col2, col3 = st.columns(3)

            with col1:
                day = st.number_input(
                    "Day of Month", 
                    min_value=1,
                    max_value=31,
                    value=15,
                    help="Day of the last contact"
                )

            with col2:
                month_options = df['month'].unique().tolist()
                month = st.selectbox(
                    "Month", 
                    month_options,
                    help="Month of the last contact"
                )

            with col3:
                st.write("")  # Empty space

            st.markdown("---")

            # Row 6: Additional Information
            st.markdown("**📊 Additional Information**")
            col1, col2, col3 = st.columns(3)

            with col1:
                previous = st.number_input(
                    "Previous Contacts", 
                    min_value=0, 
                    value=0,
                    help="Number of contacts before this campaign"
                )

            with col2:
                poutcome_options = df['poutcome'].unique().tolist()
                poutcome = st.selectbox(
                    "Previous Outcome", 
                    poutcome_options,
                    help="Outcome of previous marketing campaign"
                )

            st.markdown("---")

            # Centered Prediction Button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                submit_button = st.form_submit_button(
                    "🔮 Make Prediction", 
                    use_container_width=True,
                    type="primary"
                )

            # Process prediction
            if submit_button:
                input_data = pd.DataFrame({
                    'age': [age],
                    'job': [job],
                    'marital': [marital],
                    'education': [education],
                    'default': [default],
                    'balance': [balance],
                    'housing': [housing],
                    'loan': [loan],
                    'contact': [contact],
                    'day': [day],
                    'month': [month],
                    'duration': [duration],
                    'campaign': [campaign],
                    'pdays': [pdays],
                    'previous': [previous],
                    'poutcome': [poutcome]
                })

                # Debug info (optional)
                with st.expander("🔍 Debug - Data sent to model"):
                    st.write("Columns in input_data:", input_data.columns.tolist())
                    st.write("Columns expected by model:", df.columns.tolist())
                    st.dataframe(input_data)

                # Make prediction
                try:
                    prediction = pipeline.predict(input_data)[0]
                    prediction_proba = pipeline.predict_proba(input_data)[0]

                    st.markdown("---")
                    st.subheader("📊 Prediction Result")

                    col1, col2 = st.columns(2)

                    with col1:
                        if prediction == 'yes':
                            st.success("✅ **The client WILL ACCEPT the offer**")
                        else:
                            st.error("❌ **The client WILL REJECT the offer**")

                    with col2:
                        confidence = max(prediction_proba) * 100
                        st.info(f"🎯 **Confidence:** {confidence:.1f}%")

                    st.markdown("**📈 Probabilities:**")
                    prob_col1, prob_col2 = st.columns(2)

                    with prob_col1:
                        st.metric(
                            "Rejection Probability", 
                            f"{prediction_proba[0]*100:.1f}%",
                            help="Probability the client will reject the offer"
                        )

                    with prob_col2:
                        st.metric(
                            "Acceptance Probability", 
                            f"{prediction_proba[1]*100:.1f}%",
                            help="Probability the client will accept the offer"
                        )

                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
                    st.write("Additional error info:")
                    st.write(f"Error type: {type(e)}")
                    if hasattr(e, 'args'):
                        st.write(f"Error arguments: {e.args}")

    except FileNotFoundError:
        st.error("❌ **Trained model not found.**")
        st.markdown("""
        **To use this feature:**
        1. Run the script `train_model.py` first
        2. Make sure the file `models/decision_tree_pipeline.pkl` is created
        3. Reload this page
        """)

        # Help button
        with st.expander("🔧 Technical Help"):
            st.code("""
    # In your terminal, run:
    python train_model.py

    # Or if using Jupyter:
    %run train_model.py
            """)

    except Exception as e:
        st.error(f"❌ **Error loading model:** {str(e)}")
        st.markdown("Contact the system administrator.")

    # ===== 6. Footer =====
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <small>💡 Bank Marketing Analysis & Prediction System | By Mishell Guano <3</small>
    </div>
    """, unsafe_allow_html=True)
