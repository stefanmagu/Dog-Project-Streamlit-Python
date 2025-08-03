# 🐕 Dog Breed Analysis - Streamlit Application

A comprehensive interactive web application built with Streamlit for exploring and analyzing dog breed characteristics, traits, and geographic distribution. This application provides detailed insights into various dog breeds through data visualization, statistical analysis, and machine learning clustering techniques.

## 🌟 Features

### 🔍 **Search for Dog**
- Interactive breed selector with detailed breed information
- Visual breed characteristics display with progress bars
- Comprehensive trait analysis organized in four categories:
  - **Family Life**: Affection with family, compatibility with children and other dogs
  - **Physical**: Shedding level, grooming frequency, drooling level
  - **Social**: Openness to strangers, playfulness, protective nature, adaptability
  - **Personality**: Trainability, energy level, barking level, mental stimulation needs
- Breed-specific word cloud generation from descriptions
- Height, weight, and life expectancy information

### 📊 **Descriptive Analysis**
- **Word Cloud Visualization**: Most frequently used words in breed descriptions
- **Correlation Matrix**: Heatmap showing relationships between different breed traits
- **Frequency Distribution**: Histograms displaying the distribution of trait ratings (1-5 scale)
- **Statistical Insights**: Analysis of trait correlations with detailed explanations

### 🗺️ **Geographic Analysis (GeoPandas)**
- Interactive world map visualization
- Country-specific breed origin mapping
- Geographic distribution of dog breeds
- Country selection with highlighted regions
- List of breeds originating from selected countries

### 🔬 **Clustering Analysis**
- **Hierarchical Clustering**: Advanced clustering based on breed traits
- **K-Means Clustering**: Size-based clustering using weight and height data
- **Elbow Method**: Optimal cluster number determination
- **Silhouette Analysis**: Cluster quality evaluation
- **Regression Analysis**: 
  - Simple linear regression (height vs. life expectancy)
  - Multiple linear regression (height + weight vs. life expectancy)
- Interactive scatter plots with cluster visualization
- Statistical summary tables for each cluster

## 📁 Project Structure

```
Streamlit-Dog-Project/
├── Home.py                     # Main Streamlit application
├── requirements.txt            # Python dependencies
├── links_datasets.txt          # Data source links
├── README.md                   # Project documentation
├── date_in/                    # Input data directory
│   ├── akc-data-latest.csv     # AKC breed data with descriptions
│   ├── breed_rank.csv          # Breed popularity rankings
│   ├── breed_traits.csv        # Detailed breed characteristics
│   ├── custom.geo.json         # Geographic data for mapping
│   ├── Dog Breads Around The World.csv  # Global breed origins
│   └── dog.png                 # Application icon
└── date_out/                   # Processed data directory
    ├── date.csv                # Merged and cleaned dataset
    ├── date.py                 # Data processing functions
    └── __pycache__/            # Python cache files
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/stefanmagu/Dog-Project-Streamlit-Python.git
   cd Streamlit-Dog-Project
   ```

2. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run Home.py
   ```

4. **Access the application**
   - Open your web browser and navigate to `http://localhost:8501`

## 📦 Dependencies

### Core Libraries
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing

### Visualization
- **matplotlib**: Static plotting
- **plotly**: Interactive visualizations
- **seaborn**: Statistical data visualization
- **wordcloud**: Text visualization

### Geospatial Analysis
- **geopandas**: Geographic data processing
- **geopy**: Geocoding and geographic calculations

### Machine Learning
- **scikit-learn**: Machine learning algorithms
  - KMeans clustering
  - StandardScaler for data preprocessing
  - Silhouette analysis
- **scipy**: Scientific computing (hierarchical clustering)

### Statistical Analysis
- **statsmodels**: Statistical modeling and regression analysis

## 📊 Data Sources

The application uses multiple datasets from Kaggle and other sources:

1. **[Top Dog Breeds Around the World](https://www.kaggle.com/datasets/prajwaldongre/top-dog-breeds-around-the-world/data)** - Geographic distribution data
2. **[Dog Breeds Dataset](https://www.kaggle.com/datasets/sujaykapadnis/dog-breeds/data)** - Breed characteristics and traits
3. **[Dog Breeds Dataset (Alternative)](https://www.kaggle.com/datasets/mexwell/dog-breeds-dataset)** - Additional breed information
4. **[AKC Data](https://tmfilho.github.io/akcdata/)** - American Kennel Club breed descriptions

## 🎯 Key Features Explained

### Trait Rating System
All breed characteristics are rated on a **1-5 scale**:
- **1**: Very Low
- **2**: Low  
- **3**: Moderate
- **4**: High
- **5**: Very High

### Clustering Methodology
- **Data Preprocessing**: StandardScaler normalization
- **Hierarchical Clustering**: Ward linkage method
- **K-Means Clustering**: Elbow method for optimal k selection
- **Evaluation**: Silhouette score for cluster quality assessment

### Statistical Analysis
- **Correlation Analysis**: Pearson correlation coefficients
- **Regression Models**: 
  - Simple: `life_expectancy ~ height`
  - Multiple: `life_expectancy ~ height + weight`

## 🎨 User Interface

The application features a clean, intuitive interface with:
- **Sidebar Navigation**: Easy section switching
- **Custom Styling**: Professional color scheme and typography
- **Interactive Elements**: Progress bars, tabs, and dynamic visualizations
- **Responsive Design**: Optimized for different screen sizes

## 🔧 Technical Implementation

### Data Processing Pipeline
1. **Data Loading**: Multiple CSV files merged and cleaned
2. **Data Cleaning**: Removal of zero values and inconsistencies  
3. **Feature Engineering**: Trait categorization and standardization
4. **Export**: Processed data saved for application use

### Performance Optimization
- **Caching**: Streamlit caching for improved performance
- **Efficient Data Loading**: Optimized pandas operations
- **Memory Management**: Strategic data structure usage

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

