import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the dataset with 'latin1' encoding
spam_df = pd.read_csv('spam.csv', encoding='latin1', header=None, names=['kind', 'message'])

# Handle missing values by filling NaN with an empty string
spam_df['message'] = spam_df['message'].fillna('')

# Feature extraction
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X = tfidf_vectorizer.fit_transform(spam_df['message'])
y = (spam_df['kind'] == 'spam').astype(int)

# Initialize the Naive Bayes classifier
model = MultinomialNB()

# Train the model
model.fit(X, y)

# Function to predict if a message is spam or not
def predict_spam(message):
    # Feature extraction for the input message
    input_message = tfidf_vectorizer.transform([message])
    
    # Make prediction
    prediction = model.predict(input_message)[0]
    
    # Convert the prediction to human-readable format
    result = "Spam" if prediction == 1 else "Not Spam" if prediction == 0 else "Unknown"
    
    return result
  
# User input for prediction
user_input = input("Enter an SMS message: ")
prediction = predict_spam(user_input)
print(f"The input SMS is predicted as: {prediction}")
