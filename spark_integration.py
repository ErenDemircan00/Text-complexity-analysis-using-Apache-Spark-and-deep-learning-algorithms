from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, struct
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, ArrayType
import os
import logging
from deep_learning_model import DerinOgrenmeModel

# Loglama ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def spark_oturumu_baslat(app_adi="Türkçe Metin Analizleri"):
    """Spark oturumu başlat"""
    try:
        spark = SparkSession.builder \
            .appName(app_adi) \
            .config("spark.executor.memory", "2g") \
            .config("spark.driver.memory", "4g") \
            .config("spark.sql.shuffle.partitions", "8") \
            .getOrCreate()
        
        logger.info("Spark oturumu başarıyla başlatıldı")
        return spark
    except Exception as e:
        logger.error(f"Spark oturumu başlatılırken hata: {str(e)}")
        raise

def dosya_yukle(spark, klasor_yolu="texts"):
    """Dosyaları Spark DataFrame'e yükle"""
    try:
        from pyspark.sql import Row
        veriler = []
        
        for dosya in os.listdir(klasor_yolu):
            if dosya.endswith(".txt"):
                dosya_yolu = os.path.join(klasor_yolu, dosya)
                with open(dosya_yolu, "r", encoding="utf-8") as f:
                    icerik = f.read()
                
                veriler.append(Row(
                    dosya_adi=dosya,
                    icerik=icerik
                ))
        
        if not veriler:
            logger.warning(f"{klasor_yolu} klasöründe hiç .txt dosyası bulunamadı")
            return None
        
        # DataFrame oluştur
        df = spark.createDataFrame(veriler)
        logger.info(f"Toplam {df.count()} adet dosya yüklendi")
        return df
    
    except Exception as e:
        logger.error(f"Dosyalar yüklenirken hata: {str(e)}")
        raise

def temel_analiz_udf_tanimla(spark):
    """Temel metin analizi için UDF tanımla"""
    from text_processing import metin_analiz_et
    
    # Temel analiz için şema
    temel_schema = StructType([
        StructField("kelime_sayisi", IntegerType(), True),
        StructField("cumle_sayisi", IntegerType(), True),
        StructField("ortalama_kelime_uzunlugu", DoubleType(), True),
        StructField("ortalama_cumle_uzunlugu", DoubleType(), True),
        StructField("okunabilirlik_skoru", DoubleType(), True),
        StructField("en_yaygin_kelimeler", ArrayType(StringType()), True)
    ])
    
    # UDF tanımla
    temel_analiz_udf = udf(lambda text: metin_analiz_et(text), temel_schema)
    return temel_analiz_udf

def gelismis_analiz_udf_tanimla(spark):
    """Gelişmiş metin analizi için UDF tanımla"""
    from text_processing import gelismis_analiz_et
    
    # Gelişmiş analiz için şema
    gelismis_schema = StructType([
        StructField("kelime_sayisi", IntegerType(), True),
        StructField("cumle_sayisi", IntegerType(), True),
        StructField("ortalama_kelime_uzunlugu", DoubleType(), True),
        StructField("ortalama_cumle_uzunlugu", DoubleType(), True),
        StructField("okunabilirlik_skoru", DoubleType(), True),
        StructField("en_yaygin_kelimeler", ArrayType(StringType()), True),
        StructField("kelime_cesitliligi", DoubleType(), True),
        StructField("en_yaygin_bigramlar", ArrayType(StringType()), True)
    ])
    
    # UDF tanımla
    gelismis_analiz_udf = udf(lambda text: gelismis_analiz_et(text, "tam"), gelismis_schema)
    return gelismis_analiz_udf

def derin_analiz_udf_tanimla(spark, model_yolu='model/en_iyi_model.h5', tokenizer_yolu='model/tokenizer.pickle'):
    """Derin öğrenme analizi için UDF tanımla"""
    # Derin öğrenme analizi için şema
    derin_schema = StructType([
        StructField("duygu_skoru", DoubleType(), True),
        StructField("karmasiklik_skoru", DoubleType(), True),
        StructField("dil_kullanim_skoru", DoubleType(), True)
    ])
    
    # Modeli yükle
    model = DerinOgrenmeModel()
    model.model_yukle(model_yolu, tokenizer_yolu)
    
    # UDF tanımla
    derin_analiz_udf = udf(lambda text: model.metin_degerlendir(text), derin_schema)
    return derin_analiz_udf

def metinleri_analiz_et(spark, df, analiz_seviyesi="temel"):
    """DataFrame'deki metinleri analiz et"""
    try:
        # Temel analiz her zaman yapılır
        temel_analiz_udf = temel_analiz_udf_tanimla(spark)
        sonuc_df = df.withColumn("temel_analiz", temel_analiz_udf(col("icerik")))
        
        # Gelişmiş analiz
        if analiz_seviyesi in ["gelismis", "tam"]:
            gelismis_analiz_udf = gelismis_analiz_udf_tanimla(spark)
            sonuc_df = sonuc_df.withColumn("gelismis_analiz", gelismis_analiz_udf(col("icerik")))
        
        # Derin öğrenme analizi
        if analiz_seviyesi == "tam":
            try:
                derin_analiz_udf = derin_analiz_udf_tanimla(spark)
                sonuc_df = sonuc_df.withColumn("derin_analiz", derin_analiz_udf(col("icerik")))
            except Exception as e:
                logger.warning(f"Derin öğrenme analizi uygulanamadı: {str(e)}")
                logger.warning("Derin öğrenme analizi olmadan devam ediliyor...")
        
        return sonuc_df
    
    except Exception as e:
        logger.error(f"Metinleri analiz ederken hata: {str(e)}")
        raise

def sonuclari_kaydet(df, cikti_formati="csv", cikti_yolu="sonuclar"):
    """Analiz sonuçlarını kaydet"""
    try:
        # Klasör yoksa oluştur
        if not os.path.exists(cikti_yolu):
            os.makedirs(cikti_yolu)
        
        # Çıktı formatına göre kaydet
        if cikti_formati.lower() == "csv":
            cikti_dosyasi = os.path.join(cikti_yolu, "analiz_sonuclari.csv")
            df.write.mode("overwrite").option("header", "true").csv(cikti_dosyasi)
            logger.info(f"Sonuçlar CSV olarak kaydedildi: {cikti_dosyasi}")
        
        elif cikti_formati.lower() == "parquet":
            cikti_dosyasi = os.path.join(cikti_yolu, "analiz_sonuclari.parquet")
            df.write.mode("overwrite").parquet(cikti_dosyasi)
            logger.info(f"Sonuçlar Parquet olarak kaydedildi: {cikti_dosyasi}")
        
        elif cikti_formati.lower() == "json":
            cikti_dosyasi = os.path.join(cikti_yolu, "analiz_sonuclari.json")
            df.write.mode("overwrite").json(cikti_dosyasi)
            logger.info(f"Sonuçlar JSON olarak kaydedildi: {cikti_dosyasi}")
        
        else:
            logger.warning(f"Desteklenmeyen çıktı formatı: {cikti_formati}, CSV kullanılıyor")
            cikti_dosyasi = os.path.join(cikti_yolu, "analiz_sonuclari.csv")
            df.write.mode("overwrite").option("header", "true").csv(cikti_dosyasi)
        
        return True
    
    except Exception as e:
        logger.error(f"Sonuçlar kaydedilirken hata: {str(e)}")
        raise

def spark_kapat(spark):
    """Spark oturumunu kapat"""
    try:
        spark.stop()
        logger.info("Spark oturumu kapatıldı")
    except Exception as e:
        logger.warning(f"Spark oturumu kapatılırken hata: {str(e)}")