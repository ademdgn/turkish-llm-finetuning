"""
Turkish Sentiment Analysis Model Training Module
Türkçe sentiment analizi için BERT model eğitimi
"""

import os
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)

from datasets import DatasetDict, load_from_disk
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# WandB import (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ WandB bulunamadı. Experiment tracking olmayacak.")

class TurkishSentimentTrainer:
    """
    Türkçe sentiment analizi için model eğitici sınıfı
    """
    
    def __init__(self, 
                 model_name="dbmdz/bert-base-turkish-cased",
                 num_labels=3,
                 use_wandb=True):
        """
        Args:
            model_name (str): Kullanılacak BERT model adı
            num_labels (int): Sınıf sayısı (positive, negative, neutral)
            use_wandb (bool): WandB kullanılsın mı
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
        # Model ve tokenizer'ı yükle
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
        
        # Label mapping
        self.label_names = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
        
        # GPU kontrolü
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Kullanılan device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        self.model.to(self.device)
        
    def compute_metrics(self, eval_pred):
        """
        Evaluasyon metrikleri hesapla
        
        Args:
            eval_pred: Trainer'dan gelen tahmin ve label'lar
            
        Returns:
            dict: Hesaplanan metrikler
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Temel metrikler
        accuracy = accuracy_score(labels, predictions)
        f1_macro = f1_score(labels, predictions, average='macro')
        f1_weighted = f1_score(labels, predictions, average='weighted')
        precision = precision_score(labels, predictions, average='weighted')
        recall = recall_score(labels, predictions, average='weighted')
        
        # Sınıf bazında F1 skorları
        f1_per_class = f1_score(labels, predictions, average=None)
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision': precision,
            'recall': recall
        }
        
        # Sınıf bazında metrikler ekle
        for i, class_name in enumerate(self.label_names):
            metrics[f'f1_{class_name.lower()}'] = f1_per_class[i]
        
        return metrics
    
    def plot_confusion_matrix(self, predictions, labels, save_path=None):
        """
        Confusion matrix görselleştir
        
        Args:
            predictions (array): Model tahminleri
            labels (array): Gerçek label'lar
            save_path (str): Kaydedilecek dosya yolu
        """
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_names,
                   yticklabels=self.label_names)
        plt.title('Confusion Matrix')
        plt.ylabel('Gerçek Label')
        plt.xlabel('Tahmin Edilen Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Confusion matrix kaydedildi: {save_path}")
        
        plt.show()
        
    def load_dataset(self, dataset_path="data/processed/tokenized_dataset"):
        """
        İşlenmiş dataset'i yükle
        
        Args:
            dataset_path (str): Dataset dosya yolu
            
        Returns:
            DatasetDict: Yüklenmiş dataset
        """
        try:
            dataset = load_from_disk(dataset_path)
            print(f"✅ Dataset yüklendi: {dataset_path}")
            print(f"📊 Train samples: {len(dataset['train'])}")
            print(f"📊 Test samples: {len(dataset['test'])}")
            return dataset
        except Exception as e:
            print(f"❌ Dataset yüklenemedi: {str(e)}")
            print("💡 Önce data preprocessing çalıştırın: python src/data_preprocessing.py")
            return None
    
    def train(self, 
              dataset=None,
              output_dir="./models/checkpoints",
              num_epochs=3,
              batch_size=16,
              learning_rate=2e-5,
              warmup_steps=500,
              weight_decay=0.01,
              eval_steps=500,
              save_steps=500,
              logging_steps=100):
        """
        Modeli eğit
        
        Args:
            dataset (DatasetDict): Eğitim verisi
            output_dir (str): Model kaydetme dizini
            num_epochs (int): Epoch sayısı
            batch_size (int): Batch boyutu
            learning_rate (float): Öğrenme oranı
            warmup_steps (int): Warmup adım sayısı
            weight_decay (float): Weight decay
            eval_steps (int): Evaluation adım sayısı
            save_steps (int): Kaydetme adım sayısı
            logging_steps (int): Log adım sayısı
            
        Returns:
            Trainer: Eğitilmiş trainer objesi
        """
        print("🚀 Model eğitimi başlıyor...")
        
        # Dataset yükle
        if dataset is None:
            dataset = self.load_dataset()
            if dataset is None:
                return None
        
        train_dataset = dataset['train']
        eval_dataset = dataset['test']
        
        # WandB başlat
        if self.use_wandb:
            run_name = f"turkish-sentiment-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(
                project="turkish-sentiment-analysis",
                name=run_name,
                config={
                    "model_name": self.model_name,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "warmup_steps": warmup_steps,
                    "weight_decay": weight_decay
                }
            )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,  # Effective batch size = 4*4=16
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir='./logs',
            logging_steps=logging_steps,
            eval_strategy="steps",  # evaluation_strategy -> eval_strategy
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="f1_weighted",
            greater_is_better=True,
            report_to="wandb" if self.use_wandb else None,
            dataloader_num_workers=0,  # Windows uyumluluğu için
            remove_unused_columns=False,
            push_to_hub=False,
            fp16=True,  # Mixed precision training for memory efficiency
            optim="adamw_torch",  # More memory efficient optimizer
            max_grad_norm=1.0  # Gradient clipping
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Eğitimi başlat
        print(f"📚 Eğitim başlıyor...")
        print(f"   - Model: {self.model_name}")
        print(f"   - Train samples: {len(train_dataset)}")
        print(f"   - Eval samples: {len(eval_dataset)}")
        print(f"   - Epochs: {num_epochs}")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Learning rate: {learning_rate}")
        
        trainer.train()
        
        # Final değerlendirme
        print("📊 Final değerlendirme yapılıyor...")
        eval_results = trainer.evaluate()
        
        print("✅ Eğitim tamamlandı!")
        print("📈 Final sonuçlar:")
        for key, value in eval_results.items():
            if key.startswith('eval_'):
                metric_name = key.replace('eval_', '')
                if isinstance(value, float):
                    print(f"   - {metric_name}: {value:.4f}")
        
        # Final modeli kaydet
        final_model_path = "./models/final"
        trainer.save_model(final_model_path)
        print(f"💾 Final model kaydedildi: {final_model_path}")
        
        # Confusion matrix oluştur ve kaydet
        predictions = trainer.predict(eval_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        self.plot_confusion_matrix(
            y_pred, y_true, 
            save_path=results_dir / "confusion_matrix.png"
        )
        
        # Sonuçları JSON olarak kaydet
        final_metrics = {
            "model_name": self.model_name,
            "training_time": datetime.now().isoformat(),
            "final_metrics": eval_results,
            "training_config": {
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "warmup_steps": warmup_steps,
                "weight_decay": weight_decay
            }
        }
        
        with open(results_dir / "training_results.json", "w", encoding="utf-8") as f:
            json.dump(final_metrics, f, ensure_ascii=False, indent=2)
        
        # WandB sonlandır
        if self.use_wandb:
            wandb.finish()
        
        return trainer
    
    def predict_text(self, text, model_path="./models/final"):
        """
        Tek bir metin için tahmin yap
        
        Args:
            text (str): Tahmin yapılacak metin
            model_path (str): Model dosya yolu
            
        Returns:
            dict: Tahmin sonucu
        """
        # Model ve tokenizer yükle
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Metni tokenize et
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        
        # Tahmin yap
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Sonuçları formatla
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
        
        result = {
            "text": text,
            "predicted_label": self.label_names[predicted_class],
            "confidence": confidence,
            "all_scores": {
                self.label_names[i]: predictions[0][i].item() 
                for i in range(len(self.label_names))
            }
        }
        
        return result

def main():
    """
    Ana eğitim fonksiyonu
    """
    print("🇹🇷 Turkish Sentiment Analysis Training")
    print("=" * 50)
    
    # Trainer'ı başlat
    trainer = TurkishSentimentTrainer(
        model_name="dbmdz/bert-base-turkish-cased",
        num_labels=3,
        use_wandb=False  # İlk test için False
    )
    
    # Eğitimi başlat
    trained_trainer = trainer.train(
        num_epochs=2,  # GPU memory için daha az epoch
        batch_size=4,  # 4.3GB GPU için daha küçük batch
        learning_rate=2e-5,
        warmup_steps=100,
        eval_steps=500,  # Daha az frequent evaluation
        save_steps=500,
        logging_steps=100
    )
    
    if trained_trainer:
        print("\n🎯 Sonraki adımlar:")
        print("   1. python demos/gradio_demo.py - Demo uygulaması")
        print("   2. jupyter notebook notebooks/analysis.ipynb - Detaylı analiz")
        print("   3. python src/evaluation.py - Kapsamlı değerlendirme")
        
        # Örnek tahmin yap
        print("\n🔮 Örnek tahminler:")
        test_texts = [
            "Bu film gerçekten harika, çok beğendim!",
            "Berbat bir deneyimdi, hiç memnun kalmadım.",
            "Fena değil, ortalama bir ürün."
        ]
        
        for text in test_texts:
            result = trainer.predict_text(text)
            print(f"   📝 '{text[:50]}...'")
            print(f"   🏷️  {result['predicted_label']} (güven: {result['confidence']:.3f})")
            print()

if __name__ == "__main__":
    main()
