import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

# Step 1: Load the Data
data = pd.read_csv("datasets/Emotion_classify_Data.csv")

# Display the first few rows of the dataset
print(data.head())

# Step 2: Preprocess the Data
texts = data['Comment']  # Assuming the column with sentences is named 'Comment'
labels = data['Emotion']  # Assuming the column with labels is named 'Emotion'


# Step 3: Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)

# Step 4: Vectorize the Text Data
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)



print(f"Training data shape: {X_train_counts.shape}")
print(f"Testing data shape: {X_test_counts.shape}")

# Step 5: Train an MLP Classifier (DNN)
model = MLPClassifier(hidden_layer_sizes=(128, 64,32), max_iter=300)
model.fit(X_train_counts, y_train)

# Step 6: Predict on the Test Set
y_pred = model.predict(X_test_counts)

# Evaluate the Model
print("Accuracy:", accuracy_score(y_test, y_pred))


# Step 7: Test with New Sentences (optional)
test_sentences = [
    "I just received great news about my promotion!",  # joy
    "I'm so furious that they canceled the event without notice.",  # anger
    "I feel a deep sense of dread about the upcoming storm.",  # fear
    "Today has been such a wonderful day with my family!",  # joy
    "It's incredibly frustrating when people don't respect your time.",  # anger
    "Walking alone at night in this neighborhood makes me nervous.",  # fear
    "The surprise party was the best gift I could have imagined!",  # joy
    "I can't stand it when people cut in line; it drives me crazy!",  # anger
    "I worry about what might happen if I fail this test.",  # fear
    "Seeing my favorite band live was a dream come true!",  # joy
]

new_sentences_counts = vectorizer.transform(test_sentences)
predictions = model.predict(new_sentences_counts)



# Print the predictions with labels
print("Predictions for new sentences:", predictions)
