"""
Inference Module
Eğitilmiş model için tahmin yapma modülü
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from pathlib import Path
import json
import logging
from typing import Dict, List, Union, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TurkishSentimentInference:
    """
    Türkçe sentiment analizi için inference sınıfı
    """
    
    def __init__(self, model_path: str = "./models/final"):
        """
        Args:
            model_path (str): Eğitilmiş model yolu
        """
        self.model_path = model_path
        self.label_names = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
        self.label_emojis = ["😞", "😐", "😊"]
        
        # Model yükle
        self._load_model()
    
    def _load_model(self):
        """
        Model ve tokenizer'ı yükle
        """
        try:
            logger.info(f"Model yükleniyor: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            
            # Pipeline oluştur
            self.classifier = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                return_all_scores=True,
                device=0 if torch.cuda.is_available() else -1
            )
            
            self.model_loaded = True
            logger.info("✅ Model başarıyla yüklendi")
            
        except Exception as e:
            logger.error(f"❌ Model yüklenemedi: {str(e)}")
            self.model_loaded = False
    
    def predict(self, text: str, return_probabilities: bool = True) -> Dict:
        """
        Tek bir metin için sentiment tahmini yap
        
        Args:
            text (str): Analiz edilecek metin
            return_probabilities (bool): Tüm sınıf olasılıklarını döndür
            
        Returns:
            Dict: Tahmin sonucu
        """
        if not self.model_loaded:
            raise RuntimeError("Model yüklenmemiş!")
        
        if not text or not text.strip():
            raise ValueError("Geçerli bir metin girin!")
        
        try:
            # Tahmin yap
            results = self.classifier(text)
            
            # En yüksek skorlu tahmini al
            best_pred = max(results[0], key=lambda x: x['score'])
            
            # Label formatını düzelt
            if best_pred['label'].startswith('LABEL_'):
                label_idx = int(best_pred['label'].split('_')[1])
                predicted_label = self.label_names[label_idx]
            else:
                predicted_label = best_pred['label']
            
            # Sonucu formatla
            result = {
                "text": text,
                "predicted_label": predicted_label,
                "predicted_emoji": self.label_emojis[self.label_names.index(predicted_label)],
                "confidence": best_pred['score'],
                "status": "success"
            }
            
            if return_probabilities:
                probabilities = {}
                for pred in results[0]:
                    label = pred['label']
                    if label.startswith('LABEL_'):
                        label_idx = int(label.split('_')[1])
                        label = self.label_names[label_idx]
                    probabilities[label] = pred['score']
                
                result["probabilities"] = probabilities
            
            return result
            
        except Exception as e:
            logger.error(f"Tahmin hatası: {str(e)}")
            return {
                "text": text,
                "error": str(e),
                "status": "error"
            }
    
    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Birden fazla metin için toplu tahmin
        
        Args:
            texts (List[str]): Metin listesi
            batch_size (int): Batch boyutu
            
        Returns:
            List[Dict]: Tahmin sonuçları listesi
        """
        if not self.model_loaded:
            raise RuntimeError("Model yüklenmemiş!")
        
        results = []
        
        # Batch'ler halinde işle
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                batch_results = self.classifier(batch_texts)
                
                for j, text in enumerate(batch_texts):
                    pred_results = batch_results[j]
                    best_pred = max(pred_results, key=lambda x: x['score'])
                    
                    # Label formatını düzelt
                    if best_pred['label'].startswith('LABEL_'):
                        label_idx = int(best_pred['label'].split('_')[1])
                        predicted_label = self.label_names[label_idx]
                    else:
                        predicted_label = best_pred['label']
                    
                    result = {
                        "text": text,
                        "predicted_label": predicted_label,
                        "predicted_emoji": self.label_emojis[self.label_names.index(predicted_label)],
                        "confidence": best_pred['score'],
                        "status": "success"
                    }
                    
                    results.append(result)
                    
            except Exception as e:
                logger.error(f"Batch tahmin hatası: {str(e)}")
                for text in batch_texts:
                    results.append({
                        "text": text,
                        "error": str(e),
                        "status": "error"
                    })
        
        return results
    
    def analyze_sentiment_distribution(self, texts: List[str]) -> Dict:
        """
        Metinlerin sentiment dağılımını analiz et
        
        Args:
            texts (List[str]): Analiz edilecek metinler
            
        Returns:
            Dict: Sentiment dağılım analizi
        """
        predictions = self.predict_batch(texts)
        
        # Başarılı tahminleri filtrele
        successful_preds = [p for p in predictions if p['status'] == 'success']
        
        if not successful_preds:
            return {"error": "Hiç başarılı tahmin yapılamadı"}
        
        # Dağılımı hesapla
        label_counts = {}
        total_confidence = 0
        
        for pred in successful_preds:
            label = pred['predicted_label']
            label_counts[label] = label_counts.get(label, 0) + 1
            total_confidence += pred['confidence']
        
        # Yüzdeleri hesapla
        total_predictions = len(successful_preds)
        label_percentages = {
            label: (count / total_predictions) * 100
            for label, count in label_counts.items()
        }
        
        # En dominant sentiment
        dominant_sentiment = max(label_counts.items(), key=lambda x: x[1])
        
        return {
            "total_texts": len(texts),
            "successful_predictions": total_predictions,
            "failed_predictions": len(texts) - total_predictions,
            "label_counts": label_counts,
            "label_percentages": label_percentages,
            "dominant_sentiment": {
                "label": dominant_sentiment[0],
                "count": dominant_sentiment[1],
                "percentage": label_percentages[dominant_sentiment[0]]
            },
            "average_confidence": total_confidence / total_predictions,
            "status": "success"
        }
    
    def get_model_info(self) -> Dict:
        """
        Model bilgilerini döndür
        
        Returns:
            Dict: Model bilgileri
        """
        info = {
            "model_path": self.model_path,
            "model_loaded": self.model_loaded,
            "label_names": self.label_names,
            "num_labels": len(self.label_names)
        }
        
        if self.model_loaded:
            info.update({
                "model_type": self.model.config.model_type,
                "vocab_size": self.tokenizer.vocab_size,
                "max_position_embeddings": self.model.config.max_position_embeddings,
                "hidden_size": self.model.config.hidden_size,
                "num_attention_heads": self.model.config.num_attention_heads,
                "num_hidden_layers": self.model.config.num_hidden_layers
            })
        
        return info

# CLI Interface
def main():
    """
    Command line interface
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Turkish Sentiment Analysis Inference")
    parser.add_argument("--model-path", default="./models/final", help="Model dosya yolu")
    parser.add_argument("--text", type=str, help="Analiz edilecek metin")
    parser.add_argument("--file", type=str, help="Analiz edilecek metin dosyası")
    parser.add_argument("--output", type=str, help="Sonuçları kaydetme yolu")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch boyutu")
    
    args = parser.parse_args()
    
    # Inference objesi oluştur
    inferencer = TurkishSentimentInference(args.model_path)
    
    if not inferencer.model_loaded:
        print("❌ Model yüklenemedi!")
        return
    
    if args.text:
        # Tek metin analizi
        result = inferencer.predict(args.text)
        print("\n🔮 Sentiment Analizi Sonucu:")
        print(f"📝 Metin: {result['text']}")
        print(f"🏷️  Tahmin: {result.get('predicted_emoji', '')} {result.get('predicted_label', 'ERROR')}")
        print(f"📊 Güven: {result.get('confidence', 0):.3f}")
        
        if 'probabilities' in result:
            print("📈 Tüm skorlar:")
            for label, prob in result['probabilities'].items():
                print(f"   {label}: {prob:.3f}")
    
    elif args.file:
        # Dosyadan toplu analiz
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                if args.file.endswith('.txt'):
                    texts = [line.strip() for line in f if line.strip()]
                else:
                    import pandas as pd
                    df = pd.read_csv(args.file)
                    if 'text' in df.columns:
                        texts = df['text'].tolist()
                    else:
                        texts = df.iloc[:, 0].tolist()
            
            print(f"📊 {len(texts)} metin analiz ediliyor...")
            
            # Batch prediction
            results = inferencer.predict_batch(texts, args.batch_size)
            
            # Dağılım analizi
            distribution = inferencer.analyze_sentiment_distribution(texts)
            
            print("\n📈 Sonuçlar:")
            print(f"✅ Başarılı tahmin: {distribution['successful_predictions']}")
            print(f"❌ Başarısız tahmin: {distribution['failed_predictions']}")
            print(f"🎯 Ortalama güven: {distribution['average_confidence']:.3f}")
            
            print("\n📊 Sentiment Dağılımı:")
            for label, percentage in distribution['label_percentages'].items():
                count = distribution['label_counts'][label]
                emoji = inferencer.label_emojis[inferencer.label_names.index(label)]
                print(f"   {emoji} {label}: {count} ({percentage:.1f}%)")
            
            # Sonuçları kaydet
            if args.output:
                output_data = {
                    "predictions": results,
                    "distribution": distribution,
                    "summary": {
                        "total_texts": len(texts),
                        "model_path": args.model_path,
                        "batch_size": args.batch_size
                    }
                }
                
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                
                print(f"💾 Sonuçlar kaydedildi: {args.output}")
        
        except Exception as e:
            print(f"❌ Dosya okuma hatası: {str(e)}")
    
    else:
        # Interactive mode
        print("🇹🇷 Turkish Sentiment Analysis - Interactive Mode")
        print("'quit' yazarak çıkabilirsiniz.\n")
        
        while True:
            try:
                text = input("📝 Metin girin: ").strip()
                
                if text.lower() in ['quit', 'exit', 'q']:
                    print("👋 Görüşürüz!")
                    break
                
                if not text:
                    continue
                
                result = inferencer.predict(text)
                
                if result['status'] == 'success':
                    print(f"🏷️  Tahmin: {result['predicted_emoji']} {result['predicted_label']}")
                    print(f"📊 Güven: {result['confidence']:.3f}")
                    
                    if 'probabilities' in result:
                        print("📈 Tüm skorlar:")
                        for label, prob in result['probabilities'].items():
                            emoji = inferencer.label_emojis[inferencer.label_names.index(label)]
                            print(f"   {emoji} {label}: {prob:.3f}")
                else:
                    print(f"❌ Hata: {result.get('error', 'Bilinmeyen hata')}")
                
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Görüşürüz!")
                break
            except Exception as e:
                print(f"❌ Hata: {str(e)}")

if __name__ == "__main__":
    main()
