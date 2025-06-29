"""
Model Evaluation Module
Eğitilmiş modelin kapsamlı değerlendirmesi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc,
    precision_recall_curve
)
from datasets import load_from_disk
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """
    Model değerlendirme sınıfı
    """
    
    def __init__(self, model_path="./models/final", dataset_path="data/processed/tokenized_dataset"):
        """
        Args:
            model_path (str): Eğitilmiş model yolu
            dataset_path (str): Test dataset yolu
        """
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.label_names = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
        
        # Model ve tokenizer yükle
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.classifier = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                return_all_scores=True
            )
            print(f"✅ Model yüklendi: {model_path}")
        except Exception as e:
            print(f"❌ Model yüklenemedi: {str(e)}")
            self.model = None
            return
        
        # Dataset yükle
        try:
            self.dataset = load_from_disk(dataset_path)
            print(f"✅ Dataset yüklendi: {dataset_path}")
        except Exception as e:
            print(f"❌ Dataset yüklenemedi: {str(e)}")
            self.dataset = None
    
    def evaluate_model(self):
        """
        Modeli kapsamlı şekilde değerlendir
        
        Returns:
            dict: Değerlendirme sonuçları
        """
        if self.model is None or self.dataset is None:
            print("❌ Model veya dataset yüklenmemiş!")
            return None
        
        print("📊 Model değerlendirmesi başlıyor...")
        
        # Test verilerini hazırla
        test_dataset = self.dataset['test']
        
        # Tahminleri al
        print("🔮 Tahminler hesaplanıyor...")
        predictions, probabilities, true_labels = self._get_predictions(test_dataset)
        
        # Temel metrikler
        metrics = self._calculate_metrics(true_labels, predictions, probabilities)
        
        # Görselleştirmeler
        self._create_visualizations(true_labels, predictions, probabilities)
        
        # Hata analizi
        error_analysis = self._analyze_errors(test_dataset, true_labels, predictions)
        
        # Sonuçları kaydet
        results = {
            "metrics": metrics,
            "error_analysis": error_analysis,
            "model_path": self.model_path,
            "dataset_path": self.dataset_path
        }
        
        self._save_results(results)
        
        return results
    
    def _get_predictions(self, test_dataset):
        """
        Test verisi için tahminleri al
        """
        predictions = []
        probabilities = []
        true_labels = []
        
        # Model'i evaluation moduna al
        self.model.eval()
        device = next(self.model.parameters()).device
        
        with torch.no_grad():
            for i, example in enumerate(test_dataset):
                if i % 100 == 0:
                    print(f"İşlenen: {i}/{len(test_dataset)}")
                
                # Tahmin yap - tensor'ları GPU'ya taşı
                inputs = {
                    'input_ids': torch.tensor([example['input_ids']]).to(device),
                    'attention_mask': torch.tensor([example['attention_mask']]).to(device)
                }
                
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                pred = torch.argmax(probs, dim=-1).item()
                prob = probs[0].cpu().numpy()
                
                predictions.append(pred)
                probabilities.append(prob)
                true_labels.append(example['labels'])  # 'label' -> 'labels'
        
        return np.array(predictions), np.array(probabilities), np.array(true_labels)
    
    def _calculate_metrics(self, true_labels, predictions, probabilities):
        """
        Değerlendirme metriklerini hesapla
        """
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        metrics = {
            "accuracy": accuracy_score(true_labels, predictions),
            "f1_macro": f1_score(true_labels, predictions, average='macro'),
            "f1_weighted": f1_score(true_labels, predictions, average='weighted'),
            "precision_macro": precision_score(true_labels, predictions, average='macro'),
            "recall_macro": recall_score(true_labels, predictions, average='macro')
        }
        
        # Sınıf bazında metrikler
        class_report = classification_report(
            true_labels, predictions, 
            target_names=self.label_names, 
            output_dict=True
        )
        
        metrics["classification_report"] = class_report
        
        print("📈 Değerlendirme sonuçları:")
        print(f"   - Accuracy: {metrics['accuracy']:.4f}")
        print(f"   - F1 (macro): {metrics['f1_macro']:.4f}")
        print(f"   - F1 (weighted): {metrics['f1_weighted']:.4f}")
        print(f"   - Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"   - Recall (macro): {metrics['recall_macro']:.4f}")
        
        return metrics
    
    def _create_visualizations(self, true_labels, predictions, probabilities):
        """
        Görselleştirmeler oluştur
        """
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # 1. Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(true_labels, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_names,
                   yticklabels=self.label_names)
        plt.title('Confusion Matrix')
        plt.ylabel('Gerçek Label')
        plt.xlabel('Tahmin Edilen Label')
        plt.tight_layout()
        plt.savefig(results_dir / "confusion_matrix_detailed.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Sınıf bazında doğruluk dağılımı
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, label_name in enumerate(self.label_names):
            class_probs = probabilities[true_labels == i, i]
            axes[i].hist(class_probs, bins=20, alpha=0.7, color=f'C{i}')
            axes[i].set_title(f'{label_name} Sınıfı Güven Skorları')
            axes[i].set_xlabel('Güven Skoru')
            axes[i].set_ylabel('Frekans')
        
        plt.tight_layout()
        plt.savefig(results_dir / "confidence_distributions.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Hata dağılımı
        incorrect_predictions = true_labels != predictions
        error_confidence = np.max(probabilities[incorrect_predictions], axis=1)
        correct_confidence = np.max(probabilities[~incorrect_predictions], axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.hist(correct_confidence, bins=20, alpha=0.7, label='Doğru Tahminler', color='green')
        plt.hist(error_confidence, bins=20, alpha=0.7, label='Yanlış Tahminler', color='red')
        plt.xlabel('Maksimum Güven Skoru')
        plt.ylabel('Frekans')
        plt.title('Doğru vs Yanlış Tahminlerin Güven Skorları')
        plt.legend()
        plt.tight_layout()
        plt.savefig(results_dir / "error_confidence_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _analyze_errors(self, test_dataset, true_labels, predictions):
        """
        Hata analizi yap
        """
        # Yanlış tahminleri bul
        incorrect_mask = true_labels != predictions
        incorrect_indices = np.where(incorrect_mask)[0]
        
        # Hata türlerini analiz et
        error_types = {}
        for true_label, pred_label in zip(true_labels[incorrect_mask], predictions[incorrect_mask]):
            error_key = f"{self.label_names[true_label]} -> {self.label_names[pred_label]}"
            error_types[error_key] = error_types.get(error_key, 0) + 1
        
        print("🔍 Hata analizi:")
        print(f"   - Toplam hata: {len(incorrect_indices)}")
        print(f"   - Hata oranı: {len(incorrect_indices)/len(true_labels):.4f}")
        print("   - Hata türleri:")
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"     {error_type}: {count}")
        
        # En çok hata yapılan örnekleri kaydet
        error_examples = []
        for i in incorrect_indices[:10]:  # İlk 10 hata
            example = test_dataset[int(i)]
            error_examples.append({
                "text": self.tokenizer.decode(example['input_ids'], skip_special_tokens=True),
                "true_label": self.label_names[true_labels[i]],
                "predicted_label": self.label_names[predictions[i]],
                "index": int(i)
            })
        
        return {
            "total_errors": len(incorrect_indices),
            "error_rate": len(incorrect_indices)/len(true_labels),
            "error_types": error_types,
            "error_examples": error_examples
        }
    
    def _save_results(self, results):
        """
        Sonuçları kaydet
        """
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # JSON formatında kaydet
        with open(results_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        # CSV formatında özet kaydet
        summary_data = {
            "Metric": ["Accuracy", "F1 (Macro)", "F1 (Weighted)", "Precision", "Recall"],
            "Score": [
                results["metrics"]["accuracy"],
                results["metrics"]["f1_macro"],
                results["metrics"]["f1_weighted"],
                results["metrics"]["precision_macro"],
                results["metrics"]["recall_macro"]
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(results_dir / "metrics_summary.csv", index=False)
        
        print(f"💾 Sonuçlar kaydedildi: {results_dir}")
    
    def test_custom_examples(self, examples=None):
        """
        Özel örnekler üzerinde test yap
        
        Args:
            examples (list): Test edilecek metin listesi
        """
        if examples is None:
            examples = [
                "Bu film gerçekten harika, çok beğendim!",
                "Berbat bir deneyimdi, hiç memnun kalmadım.",
                "Fena değil, ortalama bir ürün.",
                "Mükemmel hizmet, kesinlikle tavsiye ederim!",
                "Çok kötü, paramın hakkını veremediler.",
                "İdare eder, ne iyi ne kötü.",
                "Hayal kırıklığı yaşadım, beklentimi karşılamadı.",
                "Süper bir deneyim, tekrar geleceğim!",
                "Vasat, daha iyisini görmüştüm.",
                "Nefret ettim, asla tekrar almam."
            ]
        
        print("🧪 Özel örnekler test ediliyor...")
        print("=" * 50)
        
        results = []
        for text in examples:
            prediction = self.classifier(text)
            
            # En yüksek skorlu tahmini al
            best_pred = max(prediction[0], key=lambda x: x['score'])
            
            result = {
                "text": text,
                "predicted_label": best_pred['label'],
                "confidence": best_pred['score'],
                "all_scores": {pred['label']: pred['score'] for pred in prediction[0]}
            }
            
            results.append(result)
            
            print(f"📝 Text: {text}")
            print(f"🏷️  Prediction: {best_pred['label']} (confidence: {best_pred['score']:.3f})")
            print(f"📊 All scores: {', '.join([f'{pred['label']}: {pred['score']:.3f}' for pred in prediction[0]])}")
            print("-" * 50)
        
        return results

def main():
    """
    Ana değerlendirme fonksiyonu
    """
    print("📊 Model Evaluation")
    print("=" * 30)
    
    # Evaluator'ı başlat
    evaluator = ModelEvaluator()
    
    if evaluator.model is None:
        print("❌ Model bulunamadı. Önce eğitim yapın: python src/model_training.py")
        return
    
    # Kapsamlı değerlendirme
    results = evaluator.evaluate_model()
    
    # Özel örnekler test et
    evaluator.test_custom_examples()
    
    print("\n✅ Değerlendirme tamamlandı!")
    print("📁 Sonuçlar 'results/' klasöründe kaydedildi.")

if __name__ == "__main__":
    main()
