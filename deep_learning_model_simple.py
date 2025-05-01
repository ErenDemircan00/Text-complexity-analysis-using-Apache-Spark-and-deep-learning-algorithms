"""
Bu modül, derin öğrenme modeli kullanmadan sahte bir DerinOgrenmeModel sınıfı sağlar.
TensorFlow yüklü değilken test için kullanılabilir.
"""

import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DerinOgrenmeModel:
    """Sahte derin öğrenme modeli sınıfı"""
    
    def __init__(self):
        """Modeli başlat"""
        self.hazir = False
        logger.info("TensorFlow yüklü değil - Sahte derin öğrenme modeli kullanılıyor")
    
    def model_yukle(self, model_yolu=None, tokenizer_yolu=None):
        """Model yükleme - TensorFlow olmadığı için sadece sahte bir fonksiyon"""
        logger.warning("TensorFlow yüklü değil! Gerçek model yüklenmedi, sahte skorlar üretilecek")
        self.hazir = True
        return True
    
    def metin_degerlendir(self, metin):
        """Metni değerlendir - TensorFlow olmadığı için sahte skorlar döndürür"""
        if not self.hazir:
            self.model_yukle()
            
        # Rastgele skorlar üret (0.0 - 1.0 arası)
        duygu_skoru = random.uniform(0.3, 0.8)
        karmasiklik_skoru = random.uniform(0.2, 0.7)
        dil_kullanim_skoru = random.uniform(0.4, 0.9)
        
        return {
            "duygu_skoru": float(duygu_skoru),
            "karmasiklik_skoru": float(karmasiklik_skoru),
            "dil_kullanim_skoru": float(dil_kullanim_skoru)
        }