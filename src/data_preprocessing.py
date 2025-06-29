"""
Turkish Text Preprocessing Module
Türkçe metinler için veri ön işleme sınıfı
"""

import pandas as pd
import re
import numpy as np
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import json
from pathlib import Path

class TurkishTextPreprocessor:
    """
    Türkçe metinler için özel ön işleme sınıfı
    """
    
    def __init__(self, model_name="dbmdz/bert-base-turkish-cased"):
        """
        Args:
            model_name (str): Kullanılacak BERT model adı
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label_mapping = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
        self.reverse_label_mapping = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
        
    def clean_text(self, text):
        """
        Türkçe metinleri temizle
        
        Args:
            text (str): Temizlenecek metin
            
        Returns:
            str: Temizlenmiş metin
        """
        if not isinstance(text, str):
            return ""
            
        # HTML etiketlerini temizle
        text = re.sub(r'<[^>]+>', '', text)
        
        # URL'leri temizle
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # E-mail adreslerini temizle
        text = re.sub(r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b', '', text)
        
        # Fazla boşlukları temizle
        text = re.sub(r'\\s+', ' ', text)
        
        # Özel karakterleri normalize et
        text = re.sub(r'[^a-zA-ZğĞıİöÖüÜşŞçÇ0-9\\s.,!?]', '', text)
        
        # Başlangıç ve bitiş boşluklarını temizle
        text = text.strip()
        
        return text
    
    def analyze_text_stats(self, texts):
        """
        Metin istatistiklerini analiz et
        
        Args:
            texts (list): Metin listesi
            
        Returns:
            dict: İstatistik bilgileri
        """
        lengths = [len(text.split()) for text in texts]
        char_lengths = [len(text) for text in texts]
        
        stats = {
            "total_texts": len(texts),
            "avg_word_length": np.mean(lengths),
            "max_word_length": np.max(lengths),
            "min_word_length": np.min(lengths),
            "avg_char_length": np.mean(char_lengths),
            "max_char_length": np.max(char_lengths),
            "min_char_length": np.min(char_lengths),
            "word_length_std": np.std(lengths),
            "char_length_std": np.std(char_lengths)
        }
        
        return stats
    
    def _json_serializer(self, obj):
        """
        JSON serializer for numpy types
        """
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def tokenize_function(self, examples, max_length=128):
        """
        Metinleri tokenize et
        
        Args:
            examples (dict): Dataset örnekleri
            max_length (int): Maksimum token uzunluğu
            
        Returns:
            dict: Tokenize edilmiş veriler
        """
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=max_length
            # return_tensors="pt" kaldırıldı - Dataset.map ile uyumsuz
        )
    
    def prepare_dataset(self, data_path=None, train_df=None, test_df=None, max_length=128):
        """
        Dataset'i eğitime hazırla
        
        Args:
            data_path (str): Veri dosyası yolu
            train_df (DataFrame): Train verisi
            test_df (DataFrame): Test verisi
            max_length (int): Maksimum token uzunluğu
            
        Returns:
            DatasetDict: Hazırlanmış dataset
        """
        print("🔄 Dataset hazırlanıyor...")
        
        # Veriyi yükle
        if data_path:
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith('.json'):
                df = pd.read_json(data_path)
        elif train_df is not None and test_df is not None:
            # Dataframe'ler verilmişse
            pass
        else:
            # Raw dosyalarından yükle
            try:
                train_df = pd.read_csv("data/raw/train_dataset.csv")
                test_df = pd.read_csv("data/raw/test_dataset.csv")
                print("✅ CSV dosyalarından veri yüklendi")
            except:
                print("❌ Veri dosyaları bulunamadı!")
                return None
        
        # Train ve test dataframe'lerini birleştir analiz için
        if train_df is not None and test_df is not None:
            all_df = pd.concat([train_df, test_df], ignore_index=True)
        else:
            all_df = df
            # Train-test split yap
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
        
        print(f"📊 Dataset boyutları:")
        print(f"   - Train: {len(train_df)} samples")
        print(f"   - Test: {len(test_df)} samples")
        print(f"   - Toplam: {len(all_df)} samples")
        
        # Metinleri temizle
        print("🧹 Metinler temizleniyor...")
        train_df['text'] = train_df['text'].apply(self.clean_text)
        test_df['text'] = test_df['text'].apply(self.clean_text)
        
        # Boş metinleri kaldır
        train_df = train_df[train_df['text'].str.len() > 0]
        test_df = test_df[test_df['text'].str.len() > 0]
        
        # İstatistikleri hesapla
        train_stats = self.analyze_text_stats(train_df['text'].tolist())
        test_stats = self.analyze_text_stats(test_df['text'].tolist())
        
        print(f"📈 Train set istatistikleri:")
        print(f"   - Ortalama kelime sayısı: {train_stats['avg_word_length']:.1f}")
        print(f"   - Maksimum kelime sayısı: {train_stats['max_word_length']}")
        print(f"   - Ortalama karakter sayısı: {train_stats['avg_char_length']:.1f}")
        
        # Dataset'e çevir
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)
        
        # Tokenize et
        print("🔤 Tokenization yapılıyor...")
        train_dataset = train_dataset.map(
            lambda x: self.tokenize_function(x, max_length),
            batched=True,
            remove_columns=["text"]
        )
        
        test_dataset = test_dataset.map(
            lambda x: self.tokenize_function(x, max_length),
            batched=True,
            remove_columns=["text"]
        )
        
        # DatasetDict oluştur
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })
        
        # İşlenmiş veriyi kaydet
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_dict.save_to_disk(processed_dir / "tokenized_dataset")
        
        # Meta bilgileri kaydet (numpy tiplerini düzelt)
        meta_info = {
            "model_name": self.model_name,
            "max_length": max_length,
            "train_size": int(len(train_dataset)),
            "test_size": int(len(test_dataset)),
            "train_stats": {k: float(v) if isinstance(v, (np.integer, np.floating)) else int(v) if isinstance(v, (np.integer,)) else v for k, v in train_stats.items()},
            "test_stats": {k: float(v) if isinstance(v, (np.integer, np.floating)) else int(v) if isinstance(v, (np.integer,)) else v for k, v in test_stats.items()},
            "label_mapping": self.label_mapping
        }
        
        with open(processed_dir / "preprocessing_info.json", "w", encoding="utf-8") as f:
            json.dump(meta_info, f, ensure_ascii=False, indent=2, default=self._json_serializer)
        
        print("✅ Dataset hazırlama tamamlandı!")
        print(f"💾 Kaydedilen dosyalar:")
        print(f"   - {processed_dir / 'tokenized_dataset'}")
        print(f"   - {processed_dir / 'preprocessing_info.json'}")
        
        return dataset_dict
    
    def load_processed_dataset(self, path="data/processed/tokenized_dataset"):
        """
        İşlenmiş dataset'i yükle
        
        Args:
            path (str): Dataset yolu
            
        Returns:
            DatasetDict: Yüklenmiş dataset
        """
        try:
            dataset = DatasetDict.load_from_disk(path)
            print(f"✅ İşlenmiş dataset yüklendi: {path}")
            return dataset
        except Exception as e:
            print(f"❌ Dataset yüklenemedi: {str(e)}")
            return None

def main():
    """
    Ana işleme fonksiyonu
    """
    print("🇹🇷 Turkish Text Preprocessing")
    print("=" * 40)
    
    # Preprocessor'ı başlat
    preprocessor = TurkishTextPreprocessor()
    
    # Dataset'i hazırla
    dataset = preprocessor.prepare_dataset()
    
    if dataset:
        print("\n🎯 Sonraki adım: python src/model_training.py")
    else:
        print("\n❌ Dataset hazırlanamadı. Önce data/download_dataset.py çalıştırın.")

if __name__ == "__main__":
    main()
