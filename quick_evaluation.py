"""
Quick Model Evaluation Script
Hızlı model değerlendirmesi için optimize edilmiş script
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

class QuickEvaluator:
    """
    Hızlı model değerlendirme sınıfı
    """
    
    def __init__(self, model_path="./models/final", dataset_path="data/processed/tokenized_dataset"):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.label_names = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
        
        # Model yükle
        try:
            print(f"🔄 Model yükleniyor: {model_path}")
            self.classifier = pipeline(
                "text-classification",
                model=model_path,
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True
            )
            print("✅ Model başarıyla yüklendi")
            self.model_loaded = True
        except Exception as e:
            print(f"❌ Model yüklenemedi: {e}")
            self.model_loaded = False
            return
        
        # Dataset yükle
        try:
            self.dataset = load_from_disk(dataset_path)
            print(f"✅ Dataset yüklendi: {len(self.dataset['test'])} test samples")
        except Exception as e:
            print(f"❌ Dataset yüklenemedi: {e}")
            self.dataset = None
    
    def quick_evaluate(self, sample_size=1000):
        """
        Hızlı değerlendirme (subset ile)
        """
        if not self.model_loaded or self.dataset is None:
            print("❌ Model veya dataset yüklenmemiş!")
            return None
        
        print(f"📊 Hızlı değerlendirme başlıyor ({sample_size} sample)...")
        
        # Test dataset'inden sample al
        test_dataset = self.dataset['test']
        if sample_size > len(test_dataset):
            sample_size = len(test_dataset)
            
        # Random sample al
        indices = random.sample(range(len(test_dataset)), sample_size)
        
        predictions = []
        true_labels = []
        confidences = []
        
        # Tokenizer yükle
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        print("🔮 Tahminler yapılıyor...")
        for i, idx in enumerate(indices):
            if i % 100 == 0:
                print(f"İşlenen: {i}/{sample_size}")
            
            example = test_dataset[idx]
            
            # Text'i decode et
            text = tokenizer.decode(example['input_ids'], skip_special_tokens=True)
            
            try:
                # Tahmin yap
                result = self.classifier(text)
                best_pred = max(result[0], key=lambda x: x['score'])
                
                # Label formatını düzelt
                if best_pred['label'].startswith('LABEL_'):
                    pred_label = int(best_pred['label'].split('_')[1])
                else:
                    label_map = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}
                    pred_label = label_map.get(best_pred['label'], 0)
                
                predictions.append(pred_label)
                true_labels.append(example['labels'])
                confidences.append(best_pred['score'])
                
            except Exception as e:
                print(f"Hata (index {idx}): {e}")
                continue
        
        # Metrikleri hesapla
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        confidences = np.array(confidences)
        
        accuracy = accuracy_score(true_labels, predictions)
        f1_macro = f1_score(true_labels, predictions, average='macro')
        f1_weighted = f1_score(true_labels, predictions, average='weighted')
        precision = precision_score(true_labels, predictions, average='weighted')
        recall = recall_score(true_labels, predictions, average='weighted')
        
        print(f"\n📈 Sonuçlar ({sample_size} sample):")
        print(f"  🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        print(f"  📊 F1-Score (Macro): {f1_macro:.4f}")
        print(f"  📊 F1-Score (Weighted): {f1_weighted:.4f}")
        print(f"  🔍 Precision: {precision:.4f}")
        print(f"  🎪 Recall: {recall:.4f}")
        print(f"  💪 Ortalama Güven: {confidences.mean():.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_names,
                   yticklabels=self.label_names)
        plt.title(f'Confusion Matrix (n={sample_size})')
        plt.ylabel('Gerçek Label')
        plt.xlabel('Tahmin')
        
        # Sonuçları kaydet
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        plt.savefig(results_dir / "quick_evaluation_cm.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Classification report
        print("\n📋 Detaylı Classification Report:")
        print(classification_report(true_labels, predictions, target_names=self.label_names))
        
        # Sonuçları JSON olarak kaydet
        results = {
            "sample_size": sample_size,
            "accuracy": float(accuracy),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "precision": float(precision),
            "recall": float(recall),
            "avg_confidence": float(confidences.mean()),
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(true_labels, predictions, target_names=self.label_names, output_dict=True)
        }
        
        with open(results_dir / "quick_evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Sonuçlar kaydedildi: {results_dir}")
        
        return results
    
    def test_custom_examples(self):
        """
        Özel örnekler test et
        """
        if not self.model_loaded:
            print("❌ Model yüklenmemiş!")
            return
        
        examples = [
            "Bu film gerçekten harika, çok beğendim!",
            "Berbat bir deneyimdi, hiç memnun kalmadım.",
            "Fena değil, ortalama bir ürün.",
            "Mükemmel hizmet, kesinlikle tavsiye ederim!",
            "Çok kötü, paramın hakkını veremediler.",
            "İdare eder, ne iyi ne kötü.",
            "Bu proje gerçekten başarılı oldu!",
            "Eğitim sonuçları harika çıktı!"
        ]
        
        print("🧪 Özel örnekler test ediliyor...")
        print("=" * 60)
        
        for i, text in enumerate(examples, 1):
            try:
                result = self.classifier(text)
                best_pred = max(result[0], key=lambda x: x['score'])
                
                # Emoji ekle
                emoji_map = {'NEGATIVE': '😞', 'NEUTRAL': '😐', 'POSITIVE': '😊'}
                if best_pred['label'].startswith('LABEL_'):
                    label_idx = int(best_pred['label'].split('_')[1])
                    predicted_label = self.label_names[label_idx]
                else:
                    predicted_label = best_pred['label']
                
                emoji = emoji_map.get(predicted_label, '❓')
                
                print(f"{i:2d}. Text: {text}")
                print(f"    {emoji} {predicted_label} (güven: {best_pred['score']:.3f})")
                print()
                
            except Exception as e:
                print(f"{i:2d}. Text: {text}")
                print(f"    ❌ Hata: {e}")
                print()

def main():
    print("🚀 Quick Model Evaluation")
    print("=" * 40)
    
    evaluator = QuickEvaluator()
    
    if evaluator.model_loaded and evaluator.dataset:
        # Hızlı değerlendirme
        results = evaluator.quick_evaluate(sample_size=2000)  # 2000 sample ile test
        
        # Özel örnekler
        evaluator.test_custom_examples()
        
        print("\n✅ Hızlı değerlendirme tamamlandı!")
        print("📁 Detaylı sonuçlar için: results/quick_evaluation_results.json")
        
    else:
        print("❌ Model veya dataset yüklenemedi!")

if __name__ == "__main__":
    main()
