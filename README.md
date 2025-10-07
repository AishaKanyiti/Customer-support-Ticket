# 📊 Customer Support AI Dashboard

A comprehensive Streamlit application for analyzing customer support tickets with RAG-powered insights and ML-based urgency classification.

## 🚀 Features

1. **📊 EDA Dashboard** - Interactive analytics for ticket trends, products, and resolution metrics
2. **🤖 RAG Query Assistant** - Semantic search through historical tickets using TF-IDF and nearest neighbors
3. **🚨 Urgency Classifier** - Multi-method ticket urgency classification:
   - Priority-based: Classify existing tickets by ID or priority level
   - ML-powered: Predict urgency for new ticket descriptions using trained Random Forest model

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/AishaKanyiti/Customer-support-Ticket.git
cd Customer-support-Ticket
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Running the App

### Local Development

Run the app locally:
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## 🌐 Deployment

### Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with your GitHub account
4. Click "New app"
5. Select your repository and branch
6. Set `app.py` as the main file
7. Click "Deploy"

### Deploy to Other Platforms

#### Heroku
```bash
# Create Procfile
echo "web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

#### Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## 📁 Required Files

Ensure these files are present in your deployment:
- `customer_support_tickets_cleaned.csv` - Main dataset (must include 'Ticket Priority' column)
- `urgency_model_engineered.joblib` - Pre-trained urgency prediction model
- `urgency_vectorizer_engineered.joblib` - Text vectorizer for ML predictions
- `urgency_feature_cols.joblib` - Feature columns for ML model

## 🔧 Configuration

The app configuration is stored in `.streamlit/config.toml`. You can customize:
- Theme colors
- Server settings
- Browser behavior

## 📊 Data Format

The CSV file should contain these columns:
- `Ticket Description` - Text description of the issue
- `Product Purchased` - Product name
- `Ticket Status` - Status of the ticket
- `Time to Resolution` - Resolution time
- `Customer Age` - Customer age

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 👥 Author

[Aisha Kanyiti](https://github.com/AishaKanyiti)

## 🐛 Troubleshooting

### Common Issues

**Issue**: Module not found error
```bash
# Solution: Reinstall dependencies
pip install -r requirements.txt --upgrade
```

**Issue**: File not found error
- Ensure all required `.joblib` and `.csv` files are in the app directory
- Check file paths are relative to the app root

**Issue**: Memory error on large datasets
- Consider using a subset of data for development
- Increase server memory limits in production

## 📞 Support

For issues and questions, please open an issue on GitHub.

