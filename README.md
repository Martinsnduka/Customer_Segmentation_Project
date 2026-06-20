# Customer Segmentation using K-Means Clustering

An unsupervised machine learning project that segments retail customers into distinct groups based on their purchasing behavior, enabling targeted and personalized marketing strategies.

## Customer segmentation Web App
https://martinsnduka-customer-segmentation-project-app-givd1d.streamlit.app/

## Business Problem

A retail company currently applies the **same marketing strategy to all customers**, regardless of their behavior or value. This one-size-fits-all approach has led to:

- Lower marketing campaign effectiveness
- Reduced customer engagement
- Inefficient use of marketing resources

**Goal:** Segment customers based on purchasing behavior to enable personalized marketing campaigns that improve customer retention and sales.

**Hypothesis:** Customers have distinct purchasing behaviors that can be grouped into meaningful, actionable segments.


##  Dataset
The dataset contains **2,240 customer records** with **29 attributes**, covering demographics, spending habits, purchase channels, and marketing campaign responses.

| Category | Example Features |
|---|---|
| Demographics | `Income`, `Education`, `Marital_Status`, `Kidhome`, `Teenhome`, `Year_Birth` |
| Spending | `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds` |
| Purchase Behavior | `NumDealsPurchases`, `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumWebVisitsMonth` |
| Engagement | `Recency`, `Dt_Customer` |
| Campaign Response | `AcceptedCmp1`–`AcceptedCmp5`, `Response` |



##  Project Workflow

### 1. Data Cleaning & Preprocessing
- Removed **24 missing values** in the `Income` column (negligible relative to dataset size)
- Checked for and confirmed no duplicate records
- Verified categorical fields (`Education`, `Marital_Status`) for inconsistencies

### 2. Feature Engineering
New features were engineered to better capture customer value and behavior:

| Feature | Description |
|---|---|
| `Age` | Derived from `Year_Birth` |
| `Total_children` | Sum of `Kidhome` + `Teenhome` |
| `Total_spending` | Sum of spend across all product categories |
| `Customer_Tenure` | Years since customer enrollment (`Dt_Customer`) |
| `Accepted_Campaign` | Binary flag — whether a customer accepted **any** of the 6 past campaigns |
| `Age_group` | Binned age brackets (18–29, 30–39, ... 70+) |

### 3. Exploratory Data Analysis (EDA)
Key insights uncovered during EDA:

- **Spending and purchase-frequency variables were highly right-skewed**, with long tails driven by high-value outlier customers
- **Income generally rose with age group**, peaking before declining slightly in older brackets
- No significant multicollinearity was found among the final selected features

### 4. Data Transformation
Skewed numerical features (income, spending, and purchase counts) were **log-transformed** to reduce skew and stabilize variance — a standard step before applying distance-based clustering algorithms like K-Means.

### 5. Feature Selection
The final feature set used for clustering:

Age, Recency, Income, NumStorePurchases, NumWebPurchases,
NumWebVisitsMonth, Total_spending, Customer_Tenure

**These features were scaled using `StandardScaler`** to ensure no single variable dominated the distance calculations due to scale differences.


### Final Model
KMeans(n_clusters=4, random_state=42, n_init=10)


### Dimensionality Reduction for Visualization
**PCA** (Principal Component Analysis) was used to reduce the feature space to 2 components, allowing the four clusters to be visualized on a 2D scatter plot.

##  Results — Customer Segments

| Cluster | Size | Characteristics | Segment |
|---|---|---|---|
| **3** | 600 | Highest income, highest spending |  **Loyal Customers** |
| **0** | 683 | High income, moderate spending | **High-Potential Customers** |
| **2** | 558 | Medium income, low spending | **Average Customers** |
| **1** | 375 | Lowest income, low spending |  **Price-Conscious Customers** |


**Business takeaway:** Marketing spend and personalization can now be tailored by segment — e.g., loyalty rewards for **Loyal Customers**, upsell campaigns for **High-Potential Customers**, and value-driven promotions for **Price-Conscious Customers**.


##  Deployment

The trained model and scaler were serialized with `joblib` and deployed in an interactive **Streamlit** web app, where a user can input customer attributes and receive a predicted segment in real time.

**App inputs:**
`Age`, `Recency`, `Income`, `Number of Store Purchases`, `Number of Web Purchases`, `Number of Web Visits/Month`, `Total Spending`, `Customer Tenure`


## Tech Stack

- **Python** — Pandas, NumPy
- **Visualization** — Matplotlib, Seaborn
- **Machine Learning** — Scikit-learn (`StandardScaler`, `KMeans`, `PCA`)
- **Model Persistence** — Joblib
- **Deployment** — Streamlit



## Author
email: ndukamartins2019@gmail.com
Feel free to connect or reach out with questions/feedback about this project.
