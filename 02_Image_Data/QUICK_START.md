# 🖼️ Image Data Analysis - Quick Start Guide

## Prerequisites

- Python 3.8+
- Jupyter Notebook
- All packages from requirements.txt

## Step-by-Step Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Complete Analysis

Open and run the notebook:

```bash
jupyter notebook 02_Image_Data/02_Image_Data_Complete.ipynb
```

**OR** run all cells programmatically:

```bash
jupyter nbconvert --to notebook --execute 02_Image_Data/02_Image_Data_Complete.ipynb
```

### 3. View Results Dashboard

```bash
streamlit run streamlit_app/02_Image_Data.py
```

## Expected Runtime

- **Traditional ML algorithms**: ~2-3 minutes total
- **CNN training**: ~3-5 minutes
- **Total analysis**: ~5-8 minutes

## Generated Files

- `data/image_results.json` - Results for dashboard
- Visualizations in notebook output
- Performance comparison charts

## Troubleshooting

### Common Issues:

1. **TensorFlow warnings**: Normal, can be ignored
2. **Memory warnings**: Reduce subset sizes in notebook
3. **Missing packages**: Run `pip install -r requirements.txt`

### Performance Tips:

- Use GPU if available for CNN training
- Reduce `subset_size` for faster traditional ML training
- Close other applications to free memory

## Dashboard Features

- Dataset overview with sample images
- Algorithm performance comparison
- Interactive confusion matrices
- Training time analysis

Navigate to: http://localhost:8501
