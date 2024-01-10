import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.feature_selection import VarianceThreshold

# Load the dataset
def load_dataset(file_path):
    return pd.read_csv(file_path)

# Preprocess
def preprocess_dataset(dataset):
    # Drop columns 'MD5' and 'label'
    dataset = dataset.drop(columns=['MD5', 'label'])
    return dataset

# training and test sets 
def split_dataset(dataset, target_column, test_size=0.2, random_state=42):
    X = dataset.drop(columns=[target_column])  
    y = dataset[target_column]                  
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

# Remove features 
def remove_low_variance_features(X_train, X_test, threshold=0.1):
    selector = VarianceThreshold(threshold=threshold)
    X_train_high_variance = selector.fit_transform(X_train)
    X_test_high_variance = selector.transform(X_test)
    return X_train_high_variance, X_test_high_variance

# Decision Tree 
def train_decision_tree(X_train, y_train, max_depth=5):
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=5)
    clf.fit(X_train, y_train)
    return clf

# confusion matrix and classification report
def evaluate_classifier(clf, X_test, y_test):
    predictions = clf.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    print("Confusion Matrix:")
    print(cm)

    report = classification_report(y_test, predictions)
    print("\nClassification Report:")
    print(report)

    # classification report in a PDF 
    with PdfPages('classification_report.pdf') as pdf:
        plt.figure(figsize=(10, 6))
        plot_tree(clf, filled=True, feature_names=list(X_train.columns), class_names=[str(class_label) for class_label in clf.classes_])
        plt.title('Decision Tree')
        pdf.savefig()
        plt.close()
    

    return report

# Main program
if __name__ == "__main__":
    # Task 1
    dataset = load_dataset('dataset.csv')
    dataset = preprocess_dataset(dataset)
    
    # Task 1
    X_train, X_test, y_train, y_test = split_dataset(dataset, 'Target')

    # Task 1 
    thresholds = [0.1]  
    for threshold in thresholds:
        X_train_high_variance, X_test_high_variance = remove_low_variance_features(X_train, X_test, threshold)
        print(f"For threshold {threshold}:")
    
        # Task 3
        clf = train_decision_tree(X_train_high_variance, y_train)
    
        # Task 2
        report = evaluate_classifier(clf, X_test_high_variance, y_test)
    
        # Task 2
        print(report)
