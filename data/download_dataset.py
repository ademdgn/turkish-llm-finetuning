"""
Turkish Sentiment Analysis Dataset Downloader
Bu script Türkçe sentiment analizi için gerekli veri setini indirir ve ön işleme yapar.
"""

from datasets import load_dataset
import pandas as pd
import os
import json
from pathlib import Path

def download_and_prepare_dataset():
    """
    Türkçe sentiment analysis dataset'ini indir ve hazırla
    """
    print("🔄 Dataset indiriliyor...")
    
    try:
        # Ana dataset: Türkçe sentiment analysis
        dataset = load_dataset("winvoker/turkish-sentiment-analysis-dataset")
        
        print("✅ Dataset başarıyla indirildi!")
        print(f"📊 Dataset bilgileri:")
        print(f"   - Train samples: {len(dataset['train'])}")
        print(f"   - Test samples: {len(dataset['test'])}")
        
        # Veri setini incele
        print("\n📋 Örnek veriler:")
        for i in range(3):
            print(f"   Text: {dataset['train'][i]['text'][:100]}...")
            print(f"   Label: {dataset['train'][i]['label']}")
            print("   " + "-" * 50)
        
        # Label dağılımını kontrol et
        train_labels = [example['label'] for example in dataset['train']]
        test_labels = [example['label'] for example in dataset['test']]
        
        label_counts_train = {}
        label_counts_test = {}
        
        for label in train_labels:
            label_counts_train[label] = label_counts_train.get(label, 0) + 1
            
        for label in test_labels:
            label_counts_test[label] = label_counts_test.get(label, 0) + 1
        
        print(f"\n📈 Train set label dağılımı: {label_counts_train}")
        print(f"📈 Test set label dağılımı: {label_counts_test}")
        
        # Dataset'i kaydet
        data_dir = Path("data/raw")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON formatında kaydet
        with open(data_dir / "train_dataset.json", "w", encoding="utf-8") as f:
            json.dump(dataset['train'].to_dict(), f, ensure_ascii=False, indent=2)
            
        with open(data_dir / "test_dataset.json", "w", encoding="utf-8") as f:
            json.dump(dataset['test'].to_dict(), f, ensure_ascii=False, indent=2)
        
        # CSV formatında da kaydet
        train_df = pd.DataFrame(dataset['train'])
        test_df = pd.DataFrame(dataset['test'])
        
        train_df.to_csv(data_dir / "train_dataset.csv", index=False, encoding="utf-8")
        test_df.to_csv(data_dir / "test_dataset.csv", index=False, encoding="utf-8")
        
        print(f"\n💾 Dataset kaydedildi:")
        print(f"   - {data_dir / 'train_dataset.json'}")
        print(f"   - {data_dir / 'test_dataset.json'}")
        print(f"   - {data_dir / 'train_dataset.csv'}")
        print(f"   - {data_dir / 'test_dataset.csv'}")
        
        # Dataset meta bilgilerini kaydet
        meta_info = {
            "dataset_name": "Turkish Sentiment Analysis",
            "source": "winvoker/turkish-sentiment-analysis-dataset",
            "train_size": len(dataset['train']),
            "test_size": len(dataset['test']),
            "label_distribution_train": label_counts_train,
            "label_distribution_test": label_counts_test,
            "features": list(dataset['train'].features.keys()),
            "label_names": list(set(train_labels))
        }
        
        with open(data_dir / "dataset_info.json", "w", encoding="utf-8") as f:
            json.dump(meta_info, f, ensure_ascii=False, indent=2)
        
        print("✅ Dataset hazırlama işlemi tamamlandı!")
        return dataset, meta_info
        
    except Exception as e:
        print(f"❌ Hata: {str(e)}")
        print("💡 Alternatif olarak başka bir Türkçe sentiment dataset deneyelim...")
        
        # Alternatif dataset dene
        try:
            print("🔄 Alternatif dataset deneniyor...")
            dataset = load_dataset("turkish_movie_sentiment")
            print("✅ Alternatif dataset başarıyla yüklendi!")
            return dataset, None
        except:
            print("❌ Alternatif dataset de yüklenemedi.")
            print("💡 Manuel olarak dataset hazırlanacak...")
            return create_sample_dataset()

def create_sample_dataset():
    """
    Eğer online dataset yüklenemezse, örnek bir dataset oluştur
    """
    print("🔨 Örnek dataset oluşturuluyor...")
    
    sample_data = {
        "text": [
            "Bu film gerçekten harika, çok beğendim!",
            "Berbat bir deneyimdi, hiç memnun kalmadım.",
            "Fena değil, ortalama bir ürün.",
            "Mükemmel hizmet, kesinlikle tavsiye ederim!",
            "Çok kötü, paramın hakkını veremediler.",
            "İdare eder, ne iyi ne kötü.",
            "Hayal kırıklığı yaşadım, beklentimi karşılamadı.",
            "Süper bir deneyim, tekrar geleceğim!",
            "Vasat, daha iyisini görmüştüm.",
            "Nefret ettim, asla tekrar almam.",
        ],
        "label": [2, 0, 1, 2, 0, 1, 0, 2, 1, 0]  # 0: negative, 1: neutral, 2: positive
    }
    
    # Veriyi genişlet
    extended_data = {"text": [], "label": []}
    for _ in range(100):  # Her örneği 100 kez tekrarla
        extended_data["text"].extend(sample_data["text"])
        extended_data["label"].extend(sample_data["label"])
    
    # DataFrame'e çevir
    df = pd.DataFrame(extended_data)
    
    # Train-test split
    train_size = int(0.8 * len(df))
    train_df = df[:train_size]
    test_df = df[train_size:]
    
    # Kaydet
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(data_dir / "train_dataset.csv", index=False, encoding="utf-8")
    test_df.to_csv(data_dir / "test_dataset.csv", index=False, encoding="utf-8")
    
    print(f"✅ Örnek dataset oluşturuldu: {len(train_df)} train, {len(test_df)} test samples")
    
    return train_df, test_df

if __name__ == "__main__":
    print("🇹🇷 Turkish Sentiment Analysis Dataset Downloader")
    print("=" * 50)
    
    dataset, meta_info = download_and_prepare_dataset()
    
    print("\n🎯 Sonraki adım: python src/data_preprocessing.py")
