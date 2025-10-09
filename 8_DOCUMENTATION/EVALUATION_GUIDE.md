# MLB Prediction Evaluation Guide

## Overview

This guide explains how to use the MLB prediction evaluation system to assess model performance for specific dates or date ranges.

## Available Scripts

### 1. `evaluate_predictions.py` - Comprehensive Evaluation Interface

This is the main evaluation script with full functionality.

#### Command Line Usage:

```bash
# Evaluate a specific date
python evaluate_predictions.py --date 2024-06-15

# Evaluate a date range
python evaluate_predictions.py --start-date 2024-06-01 --end-date 2024-06-30

# Interactive mode
python evaluate_predictions.py --interactive
```

#### Features:
- Single date evaluation
- Date range evaluation
- Interactive mode for easy use
- Comprehensive summary reports
- Visualization plots
- Detailed error metrics (MAE, RMSE, R², MSE)

### 2. `quick_eval.py` - Quick Evaluation Demo

Simplified script for quick evaluations and demonstrations.

#### Command Line Usage:

```bash
# Evaluate a specific date
python quick_eval.py --date 2024-06-15

# Evaluate last 7 days
python quick_eval.py --recent-days 7

# Run demo with sample dates
python quick_eval.py --demo
```

### 3. `optimized_prediction.py` - Direct Prediction Script

The core prediction script that can be used directly.

```bash
# Run with default settings
python optimized_prediction.py
```

## Evaluation Metrics Explained

### Mean Absolute Error (MAE)
- **What it is**: Average of absolute differences between predicted and actual values
- **Good values**: Lower is better (closer to 0)
- **Interpretation**: Average prediction error in the same units as your target

### Root Mean Square Error (RMSE)
- **What it is**: Square root of average squared differences
- **Good values**: Lower is better (closer to 0)
- **Interpretation**: Penalizes larger errors more heavily than MAE

### R² Score (Coefficient of Determination)
- **What it is**: Proportion of variance in the target explained by the model
- **Good values**: Higher is better (closer to 1.0)
- **Interpretation**: 
  - 1.0 = Perfect predictions
  - 0.0 = No better than predicting the mean
  - Negative = Worse than predicting the mean

### Mean Squared Error (MSE)
- **What it is**: Average of squared differences
- **Good values**: Lower is better (closer to 0)
- **Interpretation**: Heavily penalizes large errors

## Interactive Mode Guide

When using `--interactive`, you'll see:

```
📋 Interactive Mode
Available options:
1. Evaluate single date
2. Evaluate date range
3. Exit

Enter your choice (1-3): 
```

### Option 1: Single Date
- Enter a date in YYYY-MM-DD format
- Get instant evaluation results
- See detailed metrics and statistics

### Option 2: Date Range
- Enter start and end dates
- Get comprehensive analysis across multiple days
- Receive summary statistics and visualization plots

## Example Outputs

### Single Date Evaluation:
```
🔍 Evaluating predictions for 2024-06-15
==================================================
✅ Successfully processed 45 predictions
📊 Mean Absolute Error: 2.341
📈 Root Mean Square Error: 3.127
🎯 R² Score: 0.782
📉 Mean Squared Error: 9.779
🔢 Prediction Range: 0.0 to 23.4

📈 Additional Statistics:
   Mean Prediction: 8.56
   Median Prediction: 7.23
   Standard Deviation: 4.91
```

### Summary Report:
```
📊 SUMMARY REPORT
==================================================
📈 Total Evaluations: 15

🎯 Mean Absolute Error (MAE):
   Mean: 2.456
   Median: 2.341
   Std Dev: 0.423
   Range: 1.892 to 3.234

📈 Root Mean Square Error (RMSE):
   Mean: 3.201
   Median: 3.127
   Std Dev: 0.567
   Range: 2.445 to 4.123

🎯 R² Score:
   Mean: 0.751
   Median: 0.782
   Std Dev: 0.089
   Range: 0.612 to 0.891
```

## Generated Files

### Plots and Visualizations
- `evaluation_plots.png` - Comprehensive evaluation charts
- `prediction_plots.png` - Prediction vs actual value plots

### Data Files
- Evaluation results are displayed in console
- Plots are saved to the app directory
- No temporary files are created

## Troubleshooting

### Common Issues:

1. **"No data available for date"**
   - Check if the date has MLB games
   - Verify the date format (YYYY-MM-DD)
   - Ensure the merged data file exists

2. **"Model not found"**
   - Run the training script first
   - Check model file paths
   - Verify the model was saved properly

3. **"Data path issues"**
   - Verify the merged_fangraphs_data.csv exists
   - Check file permissions
   - Ensure the path is correct

### Performance Tips:

1. **For large date ranges**:
   - Use smaller chunks (e.g., 30 days at a time)
   - Monitor memory usage
   - Consider running overnight for very large ranges

2. **For best results**:
   - Use dates with sufficient historical data
   - Avoid very recent dates (model may not have enough context)
   - Regular season dates work better than off-season dates

## Integration with Other Scripts

### Using with Data Retrieval:
```bash
# Download data for a specific date
python data.retrival.py --date 2024-06-15

# Evaluate predictions for that date
python evaluate_predictions.py --date 2024-06-15
```

### Using with Training:
```bash
# Train the model
python training.py

# Evaluate the trained model
python evaluate_predictions.py --demo
```

## Advanced Usage

### Custom Date Ranges:
```python
from evaluate_predictions import PredictionEvaluator

evaluator = PredictionEvaluator()

# Custom evaluation
result = evaluator.evaluate_date('2024-06-15')
if result:
    print(f"MAE: {result['evaluation']['mae']}")
    print(f"R²: {result['evaluation']['r2']}")
```

### Batch Processing:
```python
# Evaluate multiple specific dates
dates = ['2024-06-15', '2024-06-20', '2024-06-25']
results = []

for date in dates:
    result = evaluator.evaluate_date(date)
    if result:
        results.append(result)

# Generate summary
summary = evaluator.generate_summary_report(results)
```

## Best Practices

1. **Regular Evaluation**: Run evaluations regularly to monitor model performance
2. **Date Range Analysis**: Use date ranges to understand model stability over time
3. **Visualization**: Always generate plots to visually inspect results
4. **Comparative Analysis**: Compare results across different time periods
5. **Documentation**: Keep track of evaluation results for model improvement

## Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure data files are in the correct locations
4. Review the console output for specific error messages
