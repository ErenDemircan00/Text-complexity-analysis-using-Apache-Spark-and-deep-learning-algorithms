#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Türkçe Metin Analiz Sistemi - Ana Modül
Bu modül, Türkçe metinlerin analizi için gereken tüm işlevleri çalıştırır.
"""

import os
import sys
import argparse
import logging
import time
from pyspark.sql import SparkSession

# Proje modüllerini import et
from spark_integration import (
    spark_oturumu_baslat,
    dosya_yukle,
    metinleri_analiz_et,
    sonuclari_kaydet,
    spark_kapat
)

# Loglama ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('metin_analiz.log')
    ]
)
logger = logging.getLogger(__name__)

def arguman_ayrıstır():
    """Komut satırı argümanlarını ayrıştır"""
    parser = argparse.ArgumentParser(description='Türkçe Metin Analiz Sistemi')
    
    parser.add_argument(
        '--klasor', 
        type=str, 
        default='texts',
        help='Analiz edilecek metin dosyalarının bulunduğu klasör (varsayılan: texts)'
    )
    
    parser.add_argument(
        '--analiz_seviyesi', 
        type=str, 
        choices=['temel', 'gelismis', 'tam'], 
        default='temel',
        help='Analiz seviyesi: temel, gelismis veya tam (varsayılan: temel)'
    )
    
    parser.add_argument(
        '--cikti_formati', 
        type=str, 
        choices=['csv', 'parquet', 'json'], 
        default='csv',
        help='Çıktı formatı: csv, parquet veya json (varsayılan: csv)'
    )
    
    parser.add_argument(
        '--cikti_yolu', 
        type=str, 
        default='sonuclar',
        help='Analiz sonuçlarının kaydedileceği klasör (varsayılan: sonuclar)'
    )
    
    return parser.parse_args()

def main():
    """Ana çalıştırma fonksiyonu"""
    try:
        # Başlangıç zamanı
        baslangic_zamani = time.time()
        
        # Komut satırı argümanlarını al
        args = arguman_ayrıstır()
        
        logger.info("Türkçe Metin Analiz Sistemi başlatılıyor...")
        logger.info(f"Analiz seviyesi: {args.analiz_seviyesi}")
        logger.info(f"Dosya klasörü: {args.klasor}")
        
        # Spark oturumu başlat
        spark = spark_oturumu_baslat("Türkçe Metin Analiz Sistemi")
        
        # Dosyaları yükle
        df = dosya_yukle(spark, args.klasor)
        
        if df is None or df.count() == 0:
            logger.error(f"'{args.klasor}' klasöründe analiz edilecek metin dosyası bulunamadı!")
            return
        
        # Metinleri analiz et
        sonuc_df = metinleri_analiz_et(spark, df, args.analiz_seviyesi)
        
        # Sonuçları göster
        logger.info("Analiz sonuçları:")
        sonuc_df.show(5, truncate=False)
        
        # Toplam metin sayısı
        toplam_metin = sonuc_df.count()
        logger.info(f"Toplam {toplam_metin} metin analiz edildi")
        
        # Toplam kelime sayısı
        if "temel_analiz" in sonuc_df.columns:
            toplam_kelime = sonuc_df.select("temel_analiz.kelime_sayisi").agg({"kelime_sayisi": "sum"}).collect()[0][0]
            logger.info(f"Toplam {toplam_kelime} kelime analiz edildi")
        
        # Sonuçları kaydet
        sonuclari_kaydet(sonuc_df, args.cikti_formati, args.cikti_yolu)
        
        # Spark oturumunu kapat
        spark_kapat(spark)
        
        # Bitiş zamanı ve toplam süre
        bitis_zamani = time.time()
        toplam_sure = bitis_zamani - baslangic_zamani
        logger.info(f"Analiz tamamlandı. Toplam süre: {toplam_sure:.2f} saniye")
        
    except Exception as e:
        logger.error(f"Program çalışırken hata: {str(e)}")
        raise

def bagimsiz_test():
    """Bağımsız test fonksiyonu"""
    try:
        # Test metni
        test_metni = """
        Bu bir test metnidir. Türkçe metin analizi sisteminin çalışıp çalışmadığını kontrol etmek için kullanılmaktadır.
        Bu metin kısa cümlelerden oluşur. Amaç, temel metin özelliklerini test etmektir.
        Spark ve derin öğrenme bileşenleri de test edilecektir.
        """
        
        # Spark oturumu başlat
        spark = spark_oturumu_baslat("Test Oturumu")
        
        # Test DataFrame'i oluştur
        from pyspark.sql import Row
        test_df = spark.createDataFrame([
            Row(dosya_adi="test.txt", icerik=test_metni)
        ])
        
        # Temel analiz yap
        sonuc_df = metinleri_analiz_et(spark, test_df, "temel")
        
        # Sonuçları göster
        logger.info("Test analiz sonuçları:")
        sonuc_df.show(truncate=False)
        
        # Spark oturumunu kapat
        spark_kapat(spark)
        
        logger.info("Bağımsız test başarılı!")
        return True
        
    except Exception as e:
        logger.error(f"Bağımsız test sırasında hata: {str(e)}")
        return False

if __name__ == "__main__":
    # Ana program çalıştır
    main()
    
    # İsterseniz bağımsız testi de çalıştırabilirsiniz
    # Yorum satırından çıkarmak için:
    # bagimsiz_test()