# app.py

import fitz  # PyMuPDF
import docx
import re
import nltk
from transformers import pipeline, logging

# sumy kütüphaneleri (TextRank için)
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer as SumyTokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

from rouge_score import rouge_scorer  # <-- ROUGE İÇİN YENİ İÇE AKTARMA

import os

# transformers kütüphanesinin gereksiz uyarılarını gizle
logging.set_verbosity_error()

# --- NLTK Veri İndirme ---
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("NLTK verileri indiriliyor (punkt, stopwords, punkt_tab)...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    print("İndirme tamamlandı.")


# --- A. Metin Ön İşleme (Text Preprocessing) [İyileştirildi] ---

def load_document(file_path):
    """PDF veya DOCX dosyalarını okuyup metne dönüştürür."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")

    text = ""
    if file_path.endswith(".pdf"):
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text() + " " 
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    else:
        raise ValueError("Desteklenmeyen dosya formatı. Lütfen .pdf veya .docx kullanın.")
    
    return text

def clean_text(text):
    """
    [İyileştirilmiş Fonksiyon - v2]
    Metni temizler: Sadece hikayenin ana gövdesini alır ve gürültüyü filtreler.
    """
    
    # Metni küçük harfe dönüştürme (filtrelemeden önce yapmak daha tutarlı olur)
    text = text.lower()
    
    # --- Agresif Kırpma (Trimming) ---
    # Hikayenin başladığı "Chapter 1" ifadesini bul
    start_keyword = "chapter 1"
    start_index = text.find(start_keyword)
    
    # Hikayenin bittiği "Activities" (Alıştırmalar) bölümünü bul
    end_keyword = "activities"
    end_index = text.find(end_keyword)
    
    # Eğer "Chapter 1" ve "Activities" bulunduysa, sadece bu aralığı al
    if start_index != -1 and end_index != -1 and start_index < end_index:
        print(f"[Temizleme] 'Chapter 1' ve 'Activities' arası alınıyor...")
        text = text[start_index:end_index]
    else:
        print(f"[Temizleme Uyarısı] 'Chapter 1' veya 'Activities' bulunamadı. Tam metin kullanılıyor.")
    
    # --- Kalan Gürültüyü Filtreleme ---
    # URL'leri, e-postaları ve web sitelerini (www, .com) kaldır
    text = re.sub(r'https?://\S+|www\.\S+|\S+@\S+|\S+\.com\S*', '', text, flags=re.IGNORECASE)
    
    # "Chapter X", "Page X" ve Roma rakamlarını (örn. 'iii') kaldır
    text = re.sub(r'\b(page \d+|chapter \d+|[ivxclmd]+)\b', '', text, flags=re.IGNORECASE)
    
    # Başıboş sayıları (örn. sayfa numaraları) kaldır
    text = re.sub(r'\s+\d+\s+', ' ', text)
    
    # Kalan ISBN, yayıncı vb. gürültüleri temizle
    text = re.sub(r'\b(isbn|pearson|longman|puffin|copyright|free sample)\b', '', text, flags=re.IGNORECASE)
    
    # Birden fazla boşluğu tek boşluğa indirge
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


# --- B. Hibrit Model Mimarisi ---

# 1) Çıkarımsal Aşama (TextRank)
def summarize_textrank(text, sentence_count=5):
    """Verilen metne TextRank uygulayarak en önemli cümleleri seçer."""
    parser = PlaintextParser.from_string(text, SumyTokenizer("english"))
    summarizer = TextRankSummarizer()
    summary_sentences = summarizer(parser.document, sentence_count)
    extractive_summary = " ".join([str(sentence) for sentence in summary_sentences])
    return extractive_summary

# 2) Üretimsel Aşama (LLM) [Model 'bart-large-cnn' olarak Yükseltildi]
def summarize_abstractive(text_to_summarize):
    """
    [Yükseltilmiş Fonksiyon]
    Verilen metni (TextRank çıktısı) 'facebook/bart-large-cnn' modeli
    kullanarak akıcı ve üretimsel bir özete dönüştürür.
    Bu model, özetleme için özel olarak eğitilmiştir.
    """
    print("Üretimsel (Abstractive) LLM modeli (facebook/bart-large-cnn) yükleniyor...")
    
    try:
        summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
    except Exception as e:
        print(f"Model yüklenirken hata oluştu: {e}")
        return "Model yüklenemedi."

    print("LLM modeli yüklendi. Özetleme yapılıyor...")

    # BART bir talimat (prompt) modeli değildir, doğrudan metni alır
    summary_result = summarizer_pipeline(text_to_summarize, max_length=250, min_length=50, do_sample=False)
    
    return summary_result[0]['summary_text']


# --- C. Ana Uygulama (Summary Generation) ---

def main():
    FILE_PATH = "data/level_1_-_The_Adventures_of_Tom_Sawyer_-_Penguin_Readers-min.pdf"
    
    try:
        print(f"'{FILE_PATH}' dosyası yükleniyor...")
        raw_text = load_document(FILE_PATH)
        
        print("Metin ön işleniyor (temizleme, standartlaştırma)...")
        cleaned_text = clean_text(raw_text)
        
        if not cleaned_text:
            print("Dosya boş veya metin çıkarılamadı.")
            return

        print("-" * 50)
        
        # Aşama 1: Metni Parçalara Ayır (Chunking)
        print("Aşama 1: Metin parçalara (chunks) ayrılıyor...")
        max_chunk_length = 750
        words = cleaned_text.split()
        chunks = [' '.join(words[i:i + max_chunk_length]) for i in range(0, len(words), max_chunk_length)]
        print(f"Metin {len(chunks)} parçaya ayrıldı.")
        
        # Aşama 2: Kademeli TextRank
        print("Aşama 2: Her parça için TextRank (Çıkarımsal) özet oluşturuluyor...")
        intermediate_summary_parts = []
        for i, chunk in enumerate(chunks):
            print(f"Parça {i+1}/{len(chunks)} özetleniyor...")
            chunk_summary = summarize_textrank(chunk, sentence_count=5)
            intermediate_summary_parts.append(chunk_summary)
        
        extractive_summary = " ".join(intermediate_summary_parts)
        
        print("\n--- BİRLEŞTİRİLMİŞ TEXTRANK (ÇIKARIMSAL) ÖZET ---\n")
        print(extractive_summary)
        
        print("\n" + "-" * 50)
        
        # Aşama 3: Final LLM (Abstractive)
        print("Aşama 3: LLM (Üretimsel) özet için birleştirilmiş TextRank çıktıları kullanılıyor...")
        hybrid_summary = summarize_abstractive(extractive_summary)
        
        print("\n--- HİBRİT LLM (ÜRETİMSAL) ÖZET (BART-LARGE) ---\n")
        print(hybrid_summary)

        # 1. ROUGE için bir 'Referans' (insan tarafından yazılmış) özete ihtiyacımız var.
        reference_summary = """Tom Sawyer is a boy who dislikes school and seeks adventure. He tricks his friends into painting a fence. With his friend Huck Finn, he witnesses a murder in a graveyard. Later, Tom and Becky get lost in a cave, where Tom sees the murderer, Injun Joe. Tom and Huck later find Injun Joe's treasure, and Huck is adopted by Mrs. Douglas."""
        
        # 2. ROUGE hesaplayıcıyı başlat
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # 3. Skorları hesapla (Referans vs. Model Çıktısı)
        scores = scorer.score(reference_summary, hybrid_summary)
        
        print("\n--- ROUGE SKORLARI (vs. Referans Özet) ---\n")
        
        # ROUGE-1 (Tekli Kelime Eşleşmesi)
        print(f"ROUGE-1 (Tekli Kelime Eşleşmesi):")
        print(f"  Precision: {scores['rouge1'].precision:.4f}")
        print(f"  Recall:    {scores['rouge1'].recall:.4f}")
        print(f"  F1-Score:  {scores['rouge1'].fmeasure:.4f}")
        print("-" * 20)
        
        # ROUGE-2 (İkili Kelime Eşleşmesi)
        print(f"ROUGE-2 (İkili Kelime Eşleşmesi):")
        print(f"  Precision: {scores['rouge2'].precision:.4f}")
        print(f"  Recall:    {scores['rouge2'].recall:.4f}")
        print(f"  F1-Score:  {scores['rouge2'].fmeasure:.4f}")
        print("-" * 20)
        
        # ROUGE-L (En Uzun Ortak Cümle)
        print(f"ROUGE-L (En Uzun Ortak Cümle):")
        print(f"  Precision: {scores['rougeL'].precision:.4f}")
        print(f"  Recall:    {scores['rougeL'].recall:.4f}")
        print(f"  F1-Score:  {scores['rougeL'].fmeasure:.4f}")
        
        print("\n" + "-" * 50)
        print("Proje başarıyla tamamlandı.")
        
        print("\nDeğerlendirme Notu: İki özet arasındaki akıcılık ve tutarlılık farkını inceleyin.")

    except Exception as e:
        print(f"\nBir hata oluştu: {e}")

if __name__ == "__main__":
    main()