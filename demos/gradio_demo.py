"""
Turkish Sentiment Analysis Gradio Demo
Türkçe sentiment analizi için interaktif web demo
"""

import gradio as gr
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json

class TurkishSentimentDemo:
    """
    Türkçe sentiment analizi demo sınıfı
    """
    
    def __init__(self, model_path="./models/final"):
        """
        Args:
            model_path (str): Eğitilmiş model yolu
        """
        self.model_path = model_path
        self.label_names = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
        self.label_emojis = ["😞", "😐", "😊"]
        self.label_colors = ["#ff6b6b", "#feca57", "#48dbfb"]
        
        # Model yükle
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
            self.model_loaded = True
        except Exception as e:
            print(f"❌ Model yüklenemedi: {str(e)}")
            self.model_loaded = False
    
    def predict_sentiment(self, text):
        """
        Sentiment tahmin et
        
        Args:
            text (str): Analiz edilecek metin
            
        Returns:
            tuple: (sonuç_dict, güven_grafiği)
        """
        if not self.model_loaded:
            return "❌ Model yüklenmemiş! Önce modeli eğitin.", None
        
        if not text or not text.strip():
            return "⚠️ Lütfen analiz etmek için bir metin girin!", None
        
        try:
            # Tahmin yap
            results = self.classifier(text)
            
            # Sonuçları formatla
            predictions = {}
            scores = []
            labels = []
            
            for result in results[0]:
                label = result['label']
                score = result['score']
                
                # Label formatını düzelt
                if label.startswith('LABEL_'):
                    label_idx = int(label.split('_')[1])
                    label = self.label_names[label_idx]
                
                emoji = self.label_emojis[self.label_names.index(label)]
                predictions[f"{emoji} {label}"] = f"{score:.1%}"
                
                scores.append(score)
                labels.append(f"{emoji} {label}")
            
            # En yüksek skor
            max_idx = np.argmax(scores)
            dominant_sentiment = labels[max_idx]
            confidence = scores[max_idx]
            
            # Güven grafiği oluştur
            chart = self.create_confidence_chart(labels, scores)
            
            # Sonuç mesajı
            if confidence > 0.7:
                confidence_text = "Yüksek güven"
            elif confidence > 0.5:
                confidence_text = "Orta güven"
            else:
                confidence_text = "Düşük güven"
            
            result_text = f"""
## 🎯 Tahmin Sonucu

**Dominant Sentiment:** {dominant_sentiment}  
**Güven Seviyesi:** {confidence:.1%} ({confidence_text})

### 📊 Detaylı Skorlar:
"""
            
            for label, score in predictions.items():
                result_text += f"- **{label}:** {score}\n"
            
            return result_text, chart
            
        except Exception as e:
            return f"❌ Hata oluştu: {str(e)}", None
    
    def create_confidence_chart(self, labels, scores):
        """
        Güven skorları için grafik oluştur
        
        Args:
            labels (list): Label listesi
            scores (list): Skor listesi
            
        Returns:
            matplotlib.figure: Grafik objesi
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(labels, scores, color=self.label_colors, alpha=0.8)
        
        # Değerleri çubukların üzerine yaz
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.1%}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylim(0, 1)
        ax.set_ylabel('Güven Skoru', fontsize=12)
        ax.set_title('Sentiment Analizi Sonuçları', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Y eksenini yüzde olarak formatla
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        plt.tight_layout()
        return fig
    
    def analyze_batch(self, file):
        """
        Toplu analiz yap
        
        Args:
            file: Yüklenen dosya
            
        Returns:
            tuple: (sonuç_tablosu, özet_grafik)
        """
        if not self.model_loaded:
            return "❌ Model yüklenmemiş!", None
        
        if file is None:
            return "⚠️ Lütfen bir dosya yükleyin!", None
        
        try:
            # Dosyayı oku
            if file.name.endswith('.csv'):
                df = pd.read_csv(file.name)
            elif file.name.endswith('.txt'):
                with open(file.name, 'r', encoding='utf-8') as f:
                    texts = f.readlines()
                df = pd.DataFrame({'text': [t.strip() for t in texts if t.strip()]})
            else:
                return "❌ Desteklenmeyen dosya formatı! (CSV veya TXT)", None
            
            if 'text' not in df.columns:
                return "❌ 'text' sütunu bulunamadı!", None
            
            # Tahminleri yap
            results = []
            for text in df['text'][:100]:  # İlk 100 satır
                try:
                    prediction = self.classifier(text)
                    best_pred = max(prediction[0], key=lambda x: x['score'])
                    
                    # Label formatını düzelt
                    label = best_pred['label']
                    if label.startswith('LABEL_'):
                        label_idx = int(label.split('_')[1])
                        label = self.label_names[label_idx]
                    
                    results.append({
                        'Metin': text[:100] + '...' if len(text) > 100 else text,
                        'Sentiment': label,
                        'Güven': f"{best_pred['score']:.1%}"
                    })
                except:
                    results.append({
                        'Metin': text[:100] + '...' if len(text) > 100 else text,
                        'Sentiment': 'ERROR',
                        'Güven': '0%'
                    })
            
            results_df = pd.DataFrame(results)
            
            # Özet grafik
            sentiment_counts = results_df['Sentiment'].value_counts()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(sentiment_counts.index, sentiment_counts.values, 
                         color=self.label_colors[:len(sentiment_counts)])
            
            # Değerleri çubukların üzerine yaz
            for bar, count in zip(bars, sentiment_counts.values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{count}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_ylabel('Metin Sayısı')
            ax.set_title('Sentiment Dağılımı')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            return results_df, fig
            
        except Exception as e:
            return f"❌ Hata oluştu: {str(e)}", None
    
    def get_examples(self):
        """
        Örnek metinleri döndür
        
        Returns:
            list: Örnek metin listesi
        """
        return [
            "Bu film gerçekten harika, çok beğendim! Kesinlikle tekrar izlerim.",
            "Berbat bir deneyimdi, hiç memnun kalmadım. Paramın hakkını veremediler.",
            "Fena değil, ortalama bir ürün. Ne çok iyi ne çok kötü.",
            "Müthiş bir konsert, sanatçılar harikaydı! Unutulmaz bir geceydi.",
            "Hizmet çok yavaştı, çalışanlar ilgisizdi. Bir daha gelmem.",
            "Ürün beklentimi karşıladı, kaliteli ve uygun fiyatlı.",
            "Rezalet! Bu kadar kötü bir hizmet görmemiştim.",
            "Güzel bir mekan, atmosfer hoş ama fiyatlar biraz yüksek.",
            "Süper bir deneyim yaşadım, herkese tavsiye ederim!",
            "Vasat, özel bir şey yok ama idare eder."
        ]
    
    def create_interface(self):
        """
        Gradio arayüzünü oluştur
        
        Returns:
            gr.Interface: Gradio arayüzü
        """
        # CSS stilleri
        css = """
        .gradio-container {
            font-family: 'Segoe UI', sans-serif;
        }
        .main-header {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .result-box {
            background: #f8f9fa;
            border-left: 4px solid #007bff;
            padding: 1rem;
            border-radius: 5px;
        }
        """
        
        # Ana arayüz
        with gr.Blocks(css=css, title="🇹🇷 Türkçe Sentiment Analizi") as demo:
            
            gr.HTML("""
            <div class="main-header">
                <h1>🇹🇷 Türkçe Sentiment Analizi</h1>
                <p>BERT tabanlı derin öğrenme modeli ile Türkçe metinlerin duygusal analizi</p>
            </div>
            """)
            
            with gr.Tabs():
                
                # Tab 1: Tekli Analiz
                with gr.TabItem("📝 Tekli Analiz"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            text_input = gr.Textbox(
                                label="Analiz edilecek metin",
                                placeholder="Türkçe metninizi buraya yazın...",
                                lines=4
                            )
                            
                            with gr.Row():
                                analyze_btn = gr.Button("🔍 Analiz Et", variant="primary")
                                clear_btn = gr.Button("🗑️ Temizle")
                            
                            # Örnek metinler
                            gr.Examples(
                                examples=self.get_examples(),
                                inputs=text_input,
                                label="Örnek metinler"
                            )
                        
                        with gr.Column(scale=3):
                            result_output = gr.Markdown(label="Analiz Sonucu")
                            chart_output = gr.Plot(label="Güven Skorları")
                
                # Tab 2: Toplu Analiz
                with gr.TabItem("📊 Toplu Analiz"):
                    with gr.Row():
                        with gr.Column():
                            file_input = gr.File(
                                label="CSV veya TXT dosyası yükleyin",
                                file_types=[".csv", ".txt"]
                            )
                            batch_btn = gr.Button("📊 Toplu Analiz", variant="primary")
                            
                            gr.Markdown("""
                            **Dosya formatı:**
                            - CSV: 'text' sütunu içermeli
                            - TXT: Her satırda bir metin
                            - Maksimum 100 metin işlenir
                            """)
                        
                        with gr.Column():
                            batch_results = gr.Dataframe(label="Analiz Sonuçları")
                            batch_chart = gr.Plot(label="Sentiment Dağılımı")
                
                # Tab 3: Model Bilgileri
                with gr.TabItem("ℹ️ Model Bilgileri"):
                    gr.Markdown(f"""
                    ## 🤖 Model Detayları
                    
                    **Model Tipi:** BERT (Bidirectional Encoder Representations from Transformers)
                    **Temel Model:** dbmdz/bert-base-turkish-cased
                    **Eğitim Verisi:** Türkçe sentiment analysis dataset
                    **Sınıflar:** 
                    - 😞 **NEGATIVE:** Olumsuz duygular
                    - 😐 **NEUTRAL:** Nötr/tarafsız duygular  
                    - 😊 **POSITIVE:** Olumlu duygular
                    
                    **Model Yolu:** `{self.model_path}`
                    **Model Durumu:** {"✅ Yüklendi" if self.model_loaded else "❌ Yüklenmedi"}
                    
                    ## 📈 Performans Metrikleri
                    Model eğitim sonrası performans sonuçları için `results/` klasörünü kontrol edin.
                    
                    ## 🔧 Teknik Detaylar
                    - **Tokenizer:** Turkish BERT tokenizer
                    - **Max Length:** 128 tokens
                    - **Architecture:** BERT + Classification Head
                    - **Framework:** PyTorch + Transformers
                    """)
            
            # Event handlers
            analyze_btn.click(
                fn=self.predict_sentiment,
                inputs=text_input,
                outputs=[result_output, chart_output]
            )
            
            clear_btn.click(
                fn=lambda: ("", None, None),
                outputs=[text_input, result_output, chart_output]
            )
            
            batch_btn.click(
                fn=self.analyze_batch,
                inputs=file_input,
                outputs=[batch_results, batch_chart]
            )
        
        return demo

def main():
    """
    Demo uygulamasını başlat
    """
    print("🚀 Turkish Sentiment Analysis Demo")
    print("=" * 40)
    
    # Demo objesini oluştur
    demo_app = TurkishSentimentDemo()
    
    if not demo_app.model_loaded:
        print("❌ Model yüklenemedi!")
        print("💡 Önce modeli eğitin: python src/model_training.py")
        return
    
    # Gradio arayüzünü oluştur
    demo = demo_app.create_interface()
    
    # Uygulamayı başlat
    print("🌐 Demo uygulaması başlatılıyor...")
    print("📱 Tarayıcınızda otomatik olarak açılacak")
    print("🔗 Manuel erişim: http://localhost:7860")
    
    demo.launch(
        server_name="0.0.0.0",  # Dış erişim için
        server_port=7860,
        share=True,  # Gradio public link
        show_error=True,
        debug=True
    )

if __name__ == "__main__":
    main()
