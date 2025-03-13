import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# 1. Load the Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# (Optional) Convert to a Pandas DataFrame for easier viewing
# df = pd.DataFrame(X, columns=wine.feature_names)
# df['target'] = y
# print(df.head())  # Uncomment to inspect

# 2. Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 3. Train a Naïve Bayes classifier (from Exercise 1)
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# 4. Train a Logistic Regression model
log_reg_classifier = LogisticRegression(max_iter=10000, random_state=42)
log_reg_classifier.fit(X_train, y_train)

# 5. Compare metrics: accuracy, precision, and recall for each model
# Note: Since this is a multi-class classification problem, we use 'macro' averaging for precision and recall
y_pred_nb = nb_classifier.predict(X_test)
y_pred_log_reg = log_reg_classifier.predict(X_test)

# Calculate metrics for Naïve Bayes
accuracy_nb = accuracy_score(y_test, y_pred_nb)
precision_nb = precision_score(y_test, y_pred_nb, average='macro')
recall_nb = recall_score(y_test, y_pred_nb, average='macro')

# Calculate metrics for Logistic Regression
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
precision_log_reg = precision_score(y_test, y_pred_log_reg, average='macro')
recall_log_reg = recall_score(y_test, y_pred_log_reg, average='macro')

# 6. Print results
print("Naïve Bayes Classifier:")
print(f"Accuracy: {accuracy_nb:.2f}")
print(f"Precision (macro average): {precision_nb:.2f}")
print(f"Recall (macro average): {recall_nb:.2f}")
print()

print("Logistic Regression Classifier:")
print(f"Accuracy: {accuracy_log_reg:.2f}")
print(f"Precision (macro average): {precision_log_reg:.2f}")
print(f"Recall (macro average): {recall_log_reg:.2f}")
print()

# Optional: Display confusion matrices for both models
print("Confusion Matrix for Naïve Bayes:")
print(confusion_matrix(y_test, y_pred_nb))
print()

print("Confusion Matrix for Logistic Regression:")
print(confusion_matrix(y_test, y_pred_log_reg))
