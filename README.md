# Prophet Forecaster

A sophisticated financial forecasting tool that combines technical analysis with Facebook's Prophet model for accurate stock market predictions.

## Features

### 1. Data Ingestion
- Multi-source data retrieval (Yahoo Finance, CSV files, APIs)
- Automated data validation and standardization
- Real-time and historical data support
- Ingestion checkpoints with progress tracking
- Data quality reports and initial visualizations

### 2. Data Visualization
- Interactive price trend charts using Plotly
- Statistical distribution plots
- Missing value analysis
- Volume analysis charts
- Correlation matrices
- Custom matplotlib-based static visualizations

### 3. Data Analysis
- Comprehensive statistical analysis
  - Mean, variance, standard deviation
  - Skewness and kurtosis
  - Outlier detection
- Time series decomposition
- Trend analysis
- Seasonality detection
- Data normalization and standardization
- Missing value handling

### 4. Data Sorting and Categorization
- Automated stock categorization based on:
  - Volatility levels
  - Trading volume
  - Price trends
  - Market capitalization
- Custom sorting algorithms
- Category-specific analysis reports
- Flexible rule-based categorization system

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/prophet-forecaster.git
cd prophet-forecaster

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Data Ingestion

```python
from prophet_forecaster.pipeline.data_ingestion import IngestionManager
from prophet_forecaster.pipeline.data_ingestion.fetchers import YahooFinanceFetcher

# Initialize the ingestion manager
ingestion_manager = IngestionManager()

# Fetch data for specific symbols
symbols = ['AAPL', 'GOOGL', 'MSFT']
data = ingestion_manager.fetch_data(
    symbols=symbols,
    start_date='2020-01-01',
    end_date='2023-12-31',
    fetcher=YahooFinanceFetcher()
)

# Generate ingestion report
ingestion_manager.generate_report()
```

### 2. Data Visualization

```python
from prophet_forecaster.pipeline.data_visualization import Visualizer

# Initialize visualizer
visualizer = Visualizer(data)

# Generate interactive price charts
visualizer.plot_price_trends(
    symbols=['AAPL'],
    start_date='2020-01-01',
    end_date='2023-12-31',
    include_volume=True
)

# Create statistical distribution plots
visualizer.plot_distributions()
```

### 3. Data Analysis

```python
from prophet_forecaster.pipeline.data_analysis import AnalyzerManager

# Initialize analyzer
analyzer = AnalyzerManager(data)

# Perform comprehensive analysis
analysis_results = analyzer.analyze_all(
    calculate_statistics=True,
    detect_outliers=True,
    analyze_seasonality=True
)

# Normalize data
normalized_data = analyzer.normalize_data(method='z-score')
```

### 4. Data Sorting

```python
from prophet_forecaster.pipeline.data_sorting import CategorizationManager

# Initialize categorization manager
categorizer = CategorizationManager()

# Categorize stocks
categories = categorizer.categorize_stocks(
    data,
    volatility_threshold=0.2,
    volume_threshold=1000000,
    min_market_cap=1e9
)

# Generate category reports
categorizer.generate_reports()
```

## Project Structure

```
prophet_forecaster/
├── pipeline/
│   ├── data_ingestion/           # Data retrieval and validation
│   ├── data_visualization/       # Visualization tools
│   ├── data_analysis/           # Statistical analysis
│   ├── data_sorting/            # Stock categorization
│   └── utils/                   # Shared utilities
├── tests/                       # Test suites
├── docs/                        # Documentation
└── notebooks/                   # Jupyter notebooks
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Facebook Prophet team for their forecasting tool
- Yahoo Finance for financial data access
- Contributors and maintainers of key dependencies
