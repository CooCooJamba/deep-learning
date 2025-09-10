"""
Coronavirus Tweet Sentiment Analysis

This script performs sentiment classification on COVID-19 tweets using various NLP techniques:
1. Traditional ML approaches with Bag-of-Words and TF-IDF
2. Deep learning with BERT transformer model

Author: VShulgin
Date: 2022-08-25
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Set random seeds for reproducibility
def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

class TextPreprocessor:
    """Text preprocessing utility for NLP tasks"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text, method='lemmatize'):
        """
        Clean and preprocess text
        
        Args:
            text: Input text to clean
            method: Preprocessing method ('lemmatize', 'stem', or 'basic')
            
        Returns:
            Cleaned text
        """
        text = str(text).lower()
        words = word_tokenize(text)
        
        # Remove stopwords and non-alphabetic tokens
        words = [word for word in words if word not in self.stop_words and word.isalpha()]
        
        if method == 'stem':
            words = [self.stemmer.stem(word) for word in words]
        elif method == 'lemmatize':
            words = [self.lemmatizer.lemmatize(word) for word in words]
        
        return ' '.join(words)

class CoronavirusSentimentAnalyzer:
    """Main class for coronavirus sentiment analysis"""
    
    def __init__(self):
        self.data = None
        self.preprocessor = TextPreprocessor()
        self.label_encoder = LabelEncoder()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
    def load_data(self, file_path, encoding='latin-1'):
        """Load and prepare the dataset"""
        self.data = pd.read_csv(file_path, encoding=encoding)
        self.data = self.data[['OriginalTweet', 'Sentiment']]
        print(f"Dataset loaded with {len(self.data)} samples")
        print("\nDataset info:")
        print(self.data.info())
        return self.data
    
    def explore_data(self):
        """Explore and visualize the dataset"""
        # Check class distribution
        sentiment_counts = self.data['Sentiment'].value_counts()
        print("Sentiment distribution:")
        print(sentiment_counts)
        
        # Plot class distribution
        plt.figure(figsize=(10, 6))
        sentiment_counts.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title('Sentiment Class Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('sentiment_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Check text length statistics
        self.data['text_length'] = self.data['OriginalTweet'].apply(lambda x: len(str(x).split()))
        print(f"\nMedian text length: {self.data['text_length'].median()} words")
        print(f"Mean text length: {self.data['text_length'].mean():.2f} words")
        
        return sentiment_counts
    
    def preprocess_data(self):
        """Preprocess the text data"""
        print("Preprocessing text data...")
        
        # Create cleaned versions
        self.data['CleanedTweet'] = self.data['OriginalTweet'].apply(
            lambda x: self.preprocessor.clean_text(x, method='basic')
        )
        self.data['LemmatizedTweet'] = self.data['OriginalTweet'].apply(
            lambda x: self.preprocessor.clean_text(x, method='lemmatize')
        )
        self.data['StemmedTweet'] = self.data['OriginalTweet'].apply(
            lambda x: self.preprocessor.clean_text(x, method='stem')
        )
        
        # Encode sentiment labels
        self.data['SentimentEncoded'] = self.label_encoder.fit_transform(self.data['Sentiment'])
        
        print("Preprocessing completed!")
        print("\nSample of preprocessed data:")
        print(self.data[['OriginalTweet', 'CleanedTweet', 'Sentiment']].head())
        
        return self.data
    
    def prepare_train_test_split(self, test_size=0.2):
        """Prepare train-test split"""
        X = self.data[['OriginalTweet', 'CleanedTweet', 'LemmatizedTweet', 'StemmedTweet']]
        y = self.data['SentimentEncoded']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        
        print(f"Train set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def train_traditional_models(self, X_train, X_test, y_train, y_test):
        """Train traditional ML models with different text representations"""
        results = {}
        
        # Define different text representations to try
        text_representations = {
            'Cleaned_BoW': ('CleanedTweet', CountVectorizer()),
            'Stemmed_BoW': ('StemmedTweet', CountVectorizer()),
            'Lemmatized_BoW': ('LemmatizedTweet', CountVectorizer()),
            'Cleaned_TFIDF': ('CleanedTweet', TfidfVectorizer()),
            'Stemmed_TFIDF': ('StemmedTweet', TfidfVectorizer()),
            'Lemmatized_TFIDF': ('LemmatizedTweet', TfidfVectorizer()),
        }
        
        for name, (col, vectorizer) in text_representations.items():
            print(f"\nTraining {name}...")
            
            # Create pipeline
            pipeline = make_pipeline(
                vectorizer,
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    class_weight='balanced'
                )
            )
            
            # Train model
            pipeline.fit(X_train[col], y_train)
            
            # Make predictions
            y_pred = pipeline.predict(X_test[col])
            
            # Store results
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = {
                'model': pipeline,
                'accuracy': accuracy,
                'report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            print(f"{name} Accuracy: {accuracy:.4f}")
        
        return results
    
    def plot_model_comparison(self, results):
        """Plot comparison of different model performances"""
        accuracies = {name: result['accuracy'] for name, result in results.items()}
        
        plt.figure(figsize=(12, 6))
        models = list(accuracies.keys())
        scores = list(accuracies.values())
        
        bars = plt.bar(range(len(models)), scores, color='lightblue', edgecolor='black')
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.title('Model Comparison - Traditional Approaches')
        plt.xticks(range(len(models)), models, rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('traditional_models_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return accuracies
    
    def prepare_bert_data(self, X_train, X_test, y_train, y_test, max_length=64):
        """Prepare data for BERT model"""
        print("Preparing data for BERT...")
        
        # Load BERT tokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Tokenize training data
        train_encodings = tokenizer(
            X_train['OriginalTweet'].tolist(),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Tokenize test data
        test_encodings = tokenizer(
            X_test['OriginalTweet'].tolist(),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Create PyTorch datasets
        class TweetDataset(Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels
            
            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels.iloc[idx])
                return item
            
            def __len__(self):
                return len(self.labels)
        
        train_dataset = TweetDataset(train_encodings, y_train)
        test_dataset = TweetDataset(test_encodings, y_test)
        
        return train_dataset, test_dataset, tokenizer
    
    def train_bert_model(self, train_dataset, test_dataset, num_epochs=3):
        """Train BERT model for sentiment classification"""
        print("Training BERT model...")
        
        # Compute class weights for imbalance
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(self.data['SentimentEncoded']),
            y=self.data['SentimentEncoded']
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        
        # Load pre-trained BERT model
        model = AutoModelForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=len(self.label_encoder.classes_),
            problem_type="single_label_classification"
        )
        model.to(self.device)
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir='./bert_results',
            num_train_epochs=num_epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=50,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
        )
        
        # Custom trainer with class weights
        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.get('logits')
                loss_fct = nn.CrossEntropyLoss(weight=class_weights)
                loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
                return (loss, outputs) if return_outputs else loss
        
        # Initialize trainer
        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )
        
        # Train model
        trainer.train()
        
        # Evaluate model
        eval_results = trainer.evaluate()
        print(f"BERT Evaluation results: {eval_results}")
        
        return trainer, model
    
    def evaluate_bert_model(self, trainer, test_dataset, X_test, y_test):
        """Evaluate BERT model performance"""
        print("Evaluating BERT model...")
        
        # Get predictions
        predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=self.label_encoder.classes_)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"BERT Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('BERT Model - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('bert_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return accuracy, report, cm
    
    def run_complete_analysis(self, file_path):
        """Run complete sentiment analysis pipeline"""
        print("Starting Coronavirus Sentiment Analysis Pipeline")
        print("=" * 50)
        
        # Load data
        self.load_data(file_path)
        
        # Explore data
        self.explore_data()
        
        # Preprocess data
        self.preprocess_data()
        
        # Prepare train-test split
        X_train, X_test, y_train, y_test = self.prepare_train_test_split()
        
        # Train traditional models
        print("\n" + "="*50)
        print("TRAINING TRADITIONAL ML MODELS")
        print("="*50)
        traditional_results = self.train_traditional_models(X_train, X_test, y_train, y_test)
        
        # Plot traditional model results
        self.plot_model_comparison(traditional_results)
        
        # Train BERT model
        print("\n" + "="*50)
        print("TRAINING BERT MODEL")
        print("="*50)
        train_dataset, test_dataset, tokenizer = self.prepare_bert_data(
            X_train, X_test, y_train, y_test, max_length=64
        )
        
        trainer, bert_model = self.train_bert_model(train_dataset, test_dataset, num_epochs=3)
        
        # Evaluate BERT
        bert_accuracy, bert_report, bert_cm = self.evaluate_bert_model(
            trainer, test_dataset, X_test, y_test
        )
        
        # Final comparison
        print("\n" + "="*50)
        print("FINAL RESULTS COMPARISON")
        print("="*50)
        
        # Get best traditional model
        best_traditional = max(traditional_results.items(), key=lambda x: x[1]['accuracy'])
        print(f"Best Traditional Model: {best_traditional[0]}")
        print(f"Best Traditional Accuracy: {best_traditional[1]['accuracy']:.4f}")
        print(f"BERT Model Accuracy: {bert_accuracy:.4f}")
        
        return {
            'traditional_results': traditional_results,
            'bert_accuracy': bert_accuracy,
            'bert_report': bert_report,
            'bert_model': bert_model,
            'label_encoder': self.label_encoder
        }

def main():
    """Main function to run the sentiment analysis"""
    analyzer = CoronavirusSentimentAnalyzer()
    
    try:
        results = analyzer.run_complete_analysis("Corona_NLP.csv")
        
        print("\nAnalysis completed successfully!")
        print("Results saved in current directory:")
        print("- sentiment_distribution.png")
        print("- traditional_models_comparison.png")
        print("- bert_confusion_matrix.png")
        
    except FileNotFoundError:
        print("Error: Corona_NLP.csv file not found. Please ensure the file is in the current directory.")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise e

if __name__ == "__main__":
    main()

