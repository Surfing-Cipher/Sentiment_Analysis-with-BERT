# 📘 Fine-Tuning BERT for Sentiment Analysis

This project demonstrates how to fine-tune a pre-trained BERT model for binary sentiment classification using the **IMDB Movie Reviews Dataset**. It provides an end-to-end workflow, from data preprocessing and model training to evaluation and deployment on the Hugging Face Hub.

---

## 🚀 Key Features

- Loads and preprocesses the IMDB dataset.
- Fine-tunes the `bert-base-uncased` model for sentiment analysis.
- Uses the `transformers` **Trainer API** for efficient training and evaluation.
- Tracks experiments and model performance with **Weights & Biases (W\&B)**.
- Exports the trained model for easy deployment or sharing.
- Deploys the final model to the **Hugging Face Hub**.

---

## 🧠 Model & Libraries Used

| Category       | Component                           | Description                                                                          |
| :------------- | :---------------------------------- | :----------------------------------------------------------------------------------- |
| **Model**      | `bert-base-uncased`                 | A popular BERT model from Hugging Face for tokenization and sequence classification. |
| **Frameworks** | `transformers`, `torch`, `datasets` | The core libraries for building and training the model.                              |
| **Utilities**  | `pandas`, `evaluate`, `numpy`       | Used for data handling, metric calculation, and numerical operations.                |
| **Logging**    | `wandb`                             | Provides powerful experiment tracking and visualization.                             |
| **Deployment** | `huggingface_hub`                   | Enables seamless interaction with the Hugging Face Model Hub.                        |

---

## 🗃️ Dataset

- **Source**: [IMDB Movie Reviews Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- **Size**: 50,000 reviews (25,000 for training, 25,000 for testing).
- **Classes**: Binary sentiment (Positive, Negative).

---

## 📦 Setup & Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/bert-sentiment-analysis.git
    cd bert-sentiment-analysis
    ```
2.  **Create and Activate a Virtual Environment**
    It's recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install Dependencies**
    Install all required packages from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download the Dataset**
    Ensure the `IMDB-Dataset.csv` file is present in your project directory.
5.  **Run the Notebook**
    Launch Jupyter Notebook or Jupyter Lab and open `Fine_Tuning_BERT_for_Sentiment_Analysis.ipynb` to run the code.

---

## 📊 Results

The model achieves high accuracy on binary sentiment classification. Key metrics and insights from the training process are logged to **Weights & Biases**, including loss curves and model performance.

---

## 📌 Future Work

- **Hyperparameter Tuning**: Optimize the learning rate, batch size, and number of epochs for improved generalization.
- **Alternative Models**: Experiment with other pre-trained models like `roberta-base` or `distilbert-base-uncased` to compare performance and efficiency.
- **Text Preprocessing**: Implement additional text cleaning steps, such as removing HTML tags, to further enhance model accuracy.

---

## 🙌 Acknowledgments

- **Hugging Face**: For the incredible `transformers` and `datasets` libraries.
- **Weights & Biases**: For providing powerful experiment tracking tools.
- **Stanford AI Lab**: For the publicly available IMDB dataset.
