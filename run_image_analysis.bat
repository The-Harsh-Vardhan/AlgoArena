@echo off
echo 🖼️ AlgoArena - Image Data Analysis Setup
echo ========================================

echo.
echo 📦 Installing required packages...
pip install -r requirements.txt

echo.
echo 🔄 Running the complete Image Data analysis...
jupyter nbconvert --to notebook --execute 02_Image_Data\02_Image_Data_Complete.ipynb --output 02_Image_Data_Results.ipynb

echo.
echo 🚀 Starting Streamlit dashboard...
echo Navigate to: http://localhost:8501
streamlit run streamlit_app\02_Image_Data.py

pause
