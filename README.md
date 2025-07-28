# 📰 Fake News Detection using Machine Learning

This project is a machine learning pipeline to detect **fake news** using Natural Language Processing (NLP) techniques. It uses the `TfidfVectorizer` for text feature extraction and `Multinomial Naive Bayes` for classification.

---

## 📂 Dataset

- **Source**: The dataset contains two CSV files:
  - `Fake.csv` – Fake news articles
  - `True.csv` – Real news articles

Each file has a `text` column containing the article content.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Pickle

---

## 🧠 Model Training Steps

1. **Data Loading**: Load both `Fake.csv` and `True.csv`.
2. **Labeling**: Assign labels – `0` for fake, `1` for real news.
3. **Shuffling**: Combine and shuffle the dataset.
4. **Splitting**: Split into training and testing sets (80/20).
5. **Modeling**: Use a pipeline with `TfidfVectorizer` + `MultinomialNB`.
6. **Saving**: Save the trained model to `fake_news_model.pkl`.

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
