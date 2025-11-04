import fitz 
import docx
import re
import nltk
from transformers import pipeline, logging
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer as SumyTokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

import os
logging.set_verbosity_error()

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


# --- A.Text Preprocessing ---

def load_document(file_path):
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
    
    text = text.lower()
    
    start_keyword = "chapter 1"
    start_index = text.find(start_keyword)
    
    end_keyword = "activities"
    end_index = text.find(end_keyword)

    if start_index != -1 and end_index != -1 and start_index < end_index:
        print(f"[Temizleme] 'Chapter 1' ve 'Activities' arası alınıyor...")
        text = text[start_index:end_index]
    else:
        print(f"[Temizleme Uyarısı] 'Chapter 1' veya 'Activities' bulunamadı. Tam metin kullanılıyor.")

    text = re.sub(r'https?://\S+|www\.\S+|\S+@\S+|\S+\.com\S*', '', text, flags=re.IGNORECASE)
    
    text = re.sub(r'\b(page \d+|chapter \d+|[ivxclmd]+)\b', '', text, flags=re.IGNORECASE)
   
    text = re.sub(r'\s+\d+\s+', ' ', text)
    
    text = re.sub(r'\b(isbn|pearson|longman|puffin|copyright|free sample)\b', '', text, flags=re.IGNORECASE)

    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


# --- B. Hybrit Model  ---

# 1)TextRank
def summarize_textrank(text, sentence_count=5):
    
    parser = PlaintextParser.from_string(text, SumyTokenizer("english"))
    summarizer = TextRankSummarizer()
    summary_sentences = summarizer(parser.document, sentence_count)
    extractive_summary = " ".join([str(sentence) for sentence in summary_sentences])
    return extractive_summary

# 2) LLM [Model: 'bart-large-cnn']
def summarize_abstractive(text_to_summarize):
    
    print("Üretimsel (Abstractive) LLM modeli (facebook/bart-large-cnn) yükleniyor...")
    
    try:
        summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
    except Exception as e:
        print(f"Model yüklenirken hata oluştu: {e}")
        return "Model yüklenemedi."

    print("LLM modeli yüklendi. Özetleme yapılıyor...")

    summary_result = summarizer_pipeline(text_to_summarize, max_length=250, min_length=50, do_sample=False)
    
    return summary_result[0]['summary_text']


# --- C.Summary Generation ---

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
        
        # Stage 1:Chunking
        print("Aşama 1: Metin parçalara (chunks) ayrılıyor...")
        max_chunk_length = 750
        words = cleaned_text.split()
        chunks = [' '.join(words[i:i + max_chunk_length]) for i in range(0, len(words), max_chunk_length)]
        print(f"Metin {len(chunks)} parçaya ayrıldı.")
        
        # Stage 2:TextRank
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
        
        # Stage 3: Final LLM Abstractive
        print("Aşama 3: LLM (Üretimsel) özet için birleştirilmiş TextRank çıktıları kullanılıyor...")
        hybrid_summary = summarize_abstractive(extractive_summary)
        
        print("\n--- HİBRİT LLM (ÜRETİMSAL) ÖZET (BART-LARGE) ---\n")
        print(hybrid_summary)
        
        print("\n" + "-" * 50)
        print("Proje başarıyla tamamlandı.")
        
        print("\nDeğerlendirme Notu: İki özet arasındaki akıcılık ve tutarlılık farkını inceleyin.")

    except Exception as e:
        print(f"\nBir hata oluştu: {e}")

if __name__ == "__main__":
    main()