# Causal Uplift Modeling

A comprehensive collection of Jupyter notebooks exploring uplift modeling techniques for targeted marketing and causal inference. This repository demonstrates various approaches to identify customers who are most likely to respond positively to marketing interventions.

## 📊 Dataset

All notebooks use the **Criteo AI Lab** anonymized dataset from incrementality tests:
- **13 million rows** representing individual users
- **12 anonymized features** (f0-f11) as dense float values
- **Treatment indicators**: treatment (1=treated, 0=control) and exposure (effectively exposed)
- **Target variables**: conversion (purchase) and visit (website visit)
- **Global treatment ratio**: 84.6% treated, 15% control

## 📚 Notebooks Overview

### 1. Introduction to Uplift Modeling (`introduction-to-uplift-modeling.ipynb`)
**Purpose**: Quick start guide to uplift modeling using CausalML

**Key Features**:
- Basic exploratory data analysis
- Implementation using CausalML library
- UpliftTreeClassifier and UpliftRandomForestClassifier
- Gain curve visualization
- Beginner-friendly approach

**Libraries**: `causalml`, `pandas`, `numpy`, `matplotlib`, `sklearn`

### 2. Criteo Uplift Modelling (`criteo-uplift-modelling.ipynb`)
**Purpose**: Comprehensive analysis with custom implementation

**Key Features**:
- **Advanced EDA** with correlation analysis and feature exploration
- **Statistical significance testing** using proportions z-test
- **Custom Uplift Model**: Generalized Weighted Uplift Model (LGWUM) with XGBoost
- **Treatment vs Exposure analysis**: Compares partial vs full treatment effects
- **Custom evaluation metrics**: Qini curves and uplift curves
- **Data resampling**: Random undersampling to handle class imbalance

**Methodology**:
- Class Variable Transformation approach
- Four customer segments: Persuadables, Sure Things, Lost Causes, Sleeping Dogs
- Custom uplift score calculation: `P(TR)/P(T) + P(CN)/P(C) - P(TN)/P(T) - P(CR)/P(C)`

### 3. Marketing Campaign Uplift Modeling (`marketing-campaign-uplift-modeling.ipynb`)
**Purpose**: Systematic comparison of different uplift modeling approaches

**Key Features**:
- **Meta-Learners comparison**:
  - **T-Learners**: Separate models for treatment and control groups
  - **S-Learners**: Single model with treatment as feature
- **Multiple base algorithms**: Decision Trees, LightGBM, XGBoost
- **Uplift Trees**: Direct uplift prediction using CausalML
- **Custom implementation** vs library-based approaches
- **Comprehensive evaluation**: Multiple uplift metrics (AUUC, AUQC, Weighted Average Uplift)

**Libraries**: `scikit-uplift`, `causalml`, `lightgbm`, `xgboost`

### 4. Uplift Modeling Using Advertising Data (`uplift-modeling-using-advertising-data.ipynb`)
**Purpose**: Business-focused comparison of Classical vs Uplift modeling

**Key Features**:
- **Classical Modeling (Response Model)**:
  - Traditional approach using only treated users
  - Logistic Regression vs XGBoost comparison
  - SMOTE for handling class imbalance
- **Uplift Modeling**:
  - Single Model and Two Models approaches
  - Business profit/cost analysis with custom value equations
  - Customer segmentation profiles
- **Business Metrics**:
  - Profit curves with realistic cost assumptions ($1 campaign cost, $50 conversion value)
  - Cumulative uplift curves
  - ROI analysis

**Methodology**:
- Data preprocessing: Scaling, stratified splitting
- Statistical testing for treatment effectiveness
- Custom profit calculation functions
- Customer profile analysis using radar charts

## 🔬 Key Methodologies

### Uplift Modeling Approaches
1. **Meta-Learners**:
   - **T-Learner**: Train separate models for treated/control groups
   - **S-Learner**: Single model with treatment as feature
   
2. **Direct Methods**:
   - **Uplift Trees**: Modified decision trees for direct uplift prediction
   - **Class Transformation**: LGWUM approach

### Customer Segmentation
- **Persuadables**: Convert only if treated (target customers)
- **Sure Things**: Convert regardless of treatment (avoid targeting)
- **Lost Causes**: Never convert (avoid targeting)
- **Sleeping Dogs**: Convert only if NOT treated (avoid targeting)

### Evaluation Metrics
- **Qini Coefficient**: Area under Qini curve
- **Uplift AUC**: Area under uplift curve
- **Uplift@K**: Uplift at top K% of customers
- **Weighted Average Uplift**: Overall uplift performance
- **Business Profit Curves**: Real-world ROI analysis

## 🛠️ Technical Requirements

```python
# Core libraries
pandas
numpy
matplotlib
seaborn
scikit-learn

# Uplift modeling specific
causalml
scikit-uplift

# Machine learning models
xgboost
lightgbm

# Statistical analysis
statsmodels

# Data balancing
imbalanced-learn

# Visualization
plotly
```

## 🚀 Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Causal-Uplift-Modeling.git
   cd Causal-Uplift-Modeling
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start with the introduction**:
   - Begin with `introduction-to-uplift-modeling.ipynb` for basic concepts
   - Progress to `criteo-uplift-modelling.ipynb` for advanced techniques
   - Explore `marketing-campaign-uplift-modeling.ipynb` for methodology comparison
   - Review `uplift-modeling-using-advertising-data.ipynb` for business applications

## 📈 Business Applications

- **Targeted Marketing**: Identify customers most likely to respond to promotions
- **Churn Prevention**: Find customers who will stay only if contacted
- **Product Recommendations**: Optimize recommendation strategies
- **A/B Testing**: Move beyond average treatment effects to individual-level insights
- **Resource Optimization**: Reduce marketing costs by avoiding unnecessary targeting

## 🎯 Key Insights

1. **Treatment vs Exposure**: Only 3.6% of treated users were effectively exposed, highlighting implementation challenges
2. **Business Impact**: Uplift modeling can significantly improve ROI compared to random or classical targeting
3. **Model Comparison**: Different approaches work better for different business scenarios
4. **Evaluation Complexity**: Uplift models require specialized metrics beyond traditional classification metrics

## 📝 References

- Criteo AI Lab Dataset: [Kaggle Uplift Modeling](https://www.kaggle.com/arashnic/uplift-modeling)
- Gutierrez, P., & Gérardy, J. Y. (2017). Causal Inference and Uplift Modelling: A Review of the Literature
- Kane, K., Lo, V. S., & Zheng, J. (2014). Mining for the Truly Responsive Customers and Prospects Using True-Lift Modeling

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.