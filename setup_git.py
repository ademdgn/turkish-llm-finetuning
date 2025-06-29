"""
Git Setup Script
GitHub'a yükleme için Git komutları
"""

import os
import subprocess

def run_command(command, description):
    """
    Komutu çalıştır ve sonucu göster
    """
    print(f"🔄 {description}")
    print(f"   Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Başarılı!")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
        else:
            print(f"❌ Hata: {result.stderr.strip()}")
        print()
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Komut çalıştırılamadı: {e}")
        print()
        return False

def setup_git():
    """
    Git repository'sini hazırla
    """
    print("🚀 Git Repository Setup")
    print("=" * 40)
    
    commands = [
        ("git --version", "Git versiyonu kontrol et"),
        ("git init", "Git repository başlat"),
        ("git config user.name \"Adem Doğan\"", "Git kullanıcı adı ayarla"),
        ("git config user.email \"your-email@example.com\"", "Git email ayarla"),
        ("git add .", "Tüm dosyaları stage'e ekle"),
        ("git commit -m \"🎉 Initial commit: Turkish Sentiment Analysis with BERT\n\n✨ Features:\n- BERT fine-tuning for Turkish sentiment analysis\n- 96.8% accuracy on test set\n- Production-ready Gradio demo\n- Comprehensive evaluation and analysis\n- GPU-optimized training pipeline\"", "İlk commit oluştur"),
        ("git branch -M main", "Ana branch'i main olarak ayarla"),
    ]
    
    for command, description in commands:
        success = run_command(command, description)
        if not success and "git init" in command:
            print("⚠️ Git zaten başlatılmış olabilir, devam ediliyor...")
    
    print("🎯 Git setup tamamlandı!")
    print("\n📋 GitHub'da repository oluşturun:")
    print("1. https://github.com/new adresine gidin")
    print("2. Repository name: turkish-llm-finetuning")
    print("3. Description: 🇹🇷 Turkish Sentiment Analysis with BERT - 96.8% Accuracy")
    print("4. Public olarak oluşturun")
    print("5. README, .gitignore, LICENSE eklemeyin (zaten var)")
    
    print("\n🔗 Repository oluşturduktan sonra:")
    print("git remote add origin https://github.com/ademdgn/turkish-llm-finetuning.git")
    print("git push -u origin main")

def add_remote_and_push():
    """
    Remote ekle ve push yap
    """
    print("\n🌐 GitHub'a Push Yapmaya Hazır!")
    
    remote_url = input("GitHub repository URL'sini girin (örn: https://github.com/ademdgn/turkish-llm-finetuning.git): ")
    
    if remote_url.strip():
        commands = [
            (f"git remote add origin {remote_url}", "Remote repository ekle"),
            ("git push -u origin main", "İlk push yap")
        ]
        
        for command, description in commands:
            success = run_command(command, description)
            if not success:
                print("⚠️ Hata oluştu, manuel olarak deneyin")
                break
        else:
            print("🎉 Başarıyla GitHub'a yüklendi!")
            print(f"🔗 Repository: {remote_url}")
    else:
        print("❌ URL girilmedi, manuel olarak ekleyin:")
        print("git remote add origin YOUR_REPO_URL")
        print("git push -u origin main")

def main():
    """
    Ana fonksiyon
    """
    print("🚀 Turkish Sentiment Analysis - GitHub Setup")
    print("=" * 50)
    
    # Git kurulu mu kontrol et
    if not run_command("git --version", "Git kontrolü"):
        print("❌ Git kurulu değil! Lütfen Git'i yükleyin:")
        print("https://git-scm.com/downloads")
        return
    
    # Setup git
    setup_git()
    
    # Remote eklemek için sor
    add_remote = input("\nŞimdi GitHub'a push yapmak istiyor musunuz? (y/n): ")
    if add_remote.lower() in ['y', 'yes', 'evet']:
        add_remote_and_push()
    else:
        print("\n📝 Manuel olarak şunları yapın:")
        print("1. GitHub'da repository oluşturun")
        print("2. git remote add origin YOUR_REPO_URL")
        print("3. git push -u origin main")

if __name__ == "__main__":
    main()
