import re
import logging
import nltk
from nltk.corpus import stopwords
import string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Türkçe stopwords listesini yükle
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    tr_stopwords = set(stopwords.words('turkish'))
except:
    logger.warning("Türkçe stopwords listesi yüklenemedi, boş liste kullanılıyor")
    tr_stopwords = set()

def turkce_cumle_tokenize(text):
    """Türkçe metin için cümle tokenizasyonu"""
    if not text or not isinstance(text, str):
        return []
    
    # Türkçe cümle bölücü: Nokta, ünlem, soru işareti sonrası boşluk ile ayır
    # Kısaltmalar için dikkatli olunmalı (Dr., vb.)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def turkce_kelime_tokenize(text):
    """Türkçe metin için kelime tokenizasyonu"""
    if not text or not isinstance(text, str):
        return []
    
    # Noktalama işaretlerini temizle ve kelimelere ayır
    text = text.lower()  # Metni küçük harfe çevir
    
    # Noktalama işaretlerini kaldır
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    
    # Kelimelere ayır
    words = re.findall(r'\b\w+\b', text)
    return [word for word in words if word]

def stopwords_temizle(words):
    """Türkçe stopwords temizle"""
    if not words or not isinstance(words, list):
        return []
    
    return [word for word in words if word not in tr_stopwords]

def metin_on_isleme(text):
    """
    Metin ön işleme: Küçük harfe çevirme, noktalama temizleme, 
    stopwords temizleme ve tokenizasyon
    """
    if not text or not isinstance(text, str):
        return []
    
    try:
        # Küçük harfe çevir
        text = text.lower()
        
        # Noktalama işaretlerini kaldır
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
        
        # Kelimelere ayır
        words = turkce_kelime_tokenize(text)
        
        # Stopwords temizle
        words = stopwords_temizle(words)
        
        return words
    except Exception as e:
        logger.error(f"Metin ön işleme hatası: {str(e)}")
        return []

def kelime_koku_cikar(word):
    """
    Türkçe kelime kökü çıkarma (basit implementasyon)
    Not: Gerçek bir uygulama için Zemberek gibi daha kapsamlı bir kütüphane kullanılmalı
    """
    # Basit ek kesme (gerçek bir stemmer değil)
    common_suffixes = ['lar', 'ler', 'da', 'de', 'ta', 'te', 'dan', 'den', 
                      'tan', 'ten', 'a', 'e', 'i', 'ı', 'u', 'ü']
    
    for suffix in common_suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)]
    
    return word