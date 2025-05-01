import textstat
from collections import Counter
import numpy as np
import logging
import os
from text_processing import turkce_cumle_tokenize, turkce_kelime_tokenize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_flesch_score(text):
    """Hata yönetimli Flesch okunabilirlik skoru hesaplama"""
    try:
        return textstat.flesch_reading_ease(text)
    except Exception as e:
        logger.warning(f"Okunabilirlik skoru hesaplanırken hata: {str(e)}")
        return 50.0  # Varsayılan değer

def metin_analiz_et(metin):
    """Metin için temel analizleri yap"""
    try:
        if not metin or not isinstance(metin, str):
            return {
                "kelime_sayisi": 0,
                "cumle_sayisi": 0,
                "ortalama_kelime_uzunlugu": 0.0,
                "ortalama_cumle_uzunlugu": 0.0,
                "okunabilirlik_skoru": 0.0,
                "en_yaygin_kelimeler": []
            }
        
        # Cümlelere ayır
        cumleler = turkce_cumle_tokenize(metin)
        cumle_sayisi = len(cumleler)
        
        # Kelimelere ayır
        kelimeler = turkce_kelime_tokenize(metin)
        kelime_sayisi = len(kelimeler)
        
        # Kelime uzunlukları
        if kelime_sayisi > 0:
            kelime_uzunluklari = [len(kelime) for kelime in kelimeler]
            ortalama_kelime_uzunlugu = np.mean(kelime_uzunluklari)
        else:
            ortalama_kelime_uzunlugu = 0.0
        
        # Cümle uzunlukları
        if cumle_sayisi > 0:
            cumle_uzunluklari = [len(turkce_kelime_tokenize(cumle)) for cumle in cumleler]
            ortalama_cumle_uzunlugu = np.mean(cumle_uzunluklari)
        else:
            ortalama_cumle_uzunlugu = 0.0
        
        # Okunabilirlik skoru
        okunabilirlik_skoru = safe_flesch_score(metin)
        
        # En yaygın kelimeler
        kelime_sayimi = Counter(kelimeler).most_common(10)
        en_yaygin_kelimeler = [kelime for kelime, _ in kelime_sayimi]
        
        return {
            "kelime_sayisi": kelime_sayisi,
            "cumle_sayisi": cumle_sayisi,
            "ortalama_kelime_uzunlugu": float(ortalama_kelime_uzunlugu),
            "ortalama_cumle_uzunlugu": float(ortalama_cumle_uzunlugu),
            "okunabilirlik_skoru": float(okunabilirlik_skoru),
            "en_yaygin_kelimeler": en_yaygin_kelimeler
        }
    except Exception as e:
        logger.error(f"Metin analizi sırasında hata: {str(e)}")
        raise

def gelismis_analiz_et(metin, analiz_turu="tam"):
    """Metin için gelişmiş analizler yap"""
    try:
        temel_analiz = metin_analiz_et(metin)
        
        if analiz_turu == "temel":
            return temel_analiz
        
        # Kelime çeşitliliği (TTR - Type-Token Ratio)
        kelimeler = turkce_kelime_tokenize(metin)
        unique_kelimeler = set(kelimeler)
        ttr = len(unique_kelimeler) / len(kelimeler) if kelimeler else 0
        
        # N-gram analizi (bigram)
        bigramlar = []
        for i in range(len(kelimeler) - 1):
            bigramlar.append(f"{kelimeler[i]} {kelimeler[i+1]}")
        
        en_yaygin_bigramlar = Counter(bigramlar).most_common(5)
        
        # Sonuçları birleştir
        gelismis_sonuclar = {
            **temel_analiz,
            "kelime_cesitliligi": float(ttr),
            "en_yaygin_bigramlar": [bigram for bigram, _ in en_yaygin_bigramlar]
        }
        
        return gelismis_sonuclar
    except Exception as e:
        logger.error(f"Gelişmiş analiz sırasında hata: {str(e)}")
        raise