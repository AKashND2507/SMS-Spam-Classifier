# SMS Spam Detection Model

This model is designed to predict whether an SMS message is spam or not using a Naive Bayes classifier trained on the SMS Spam Collection dataset.

## Usage

1. **Requirements:**
   - Python 3.x
   - Required libraries: pandas, scikit-learn

2. **Installation:**
   - Clone the repository or download the model files.

3. **Setup:**
   - Ensure Python and required libraries are installed.
   - Place your dataset named 'spam.csv' in the same directory.

4. **Running the Model:**
   - Execute the 'spam_detection_model.py' script.
   - Enter an SMS message when prompted.
   - The model will predict whether the input message is spam or not.

### Additional Notes:
The model uses a Multinomial Naive Bayes classifier.
The dataset is preprocessed to handle missing values and tokenize messages.
Dataset
The model is trained on the SMS Spam Collection dataset. The dataset includes messages labeled as 'spam' or 'ham' (not spam).

### Model Details
- Model Type: Multinomial Naive Bayes.
- Feature Extraction: TF-IDF (Term Frequency-Inverse Document Frequency).
