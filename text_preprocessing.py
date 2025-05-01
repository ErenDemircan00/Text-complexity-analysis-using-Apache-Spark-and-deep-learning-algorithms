import re
import string
import numpy as np
from collections import Counter
import logging
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Loglama ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    def __init__(self, max_words=10000, max_sequence_length=100):
        """
        Metin önişleme için sınıf
        
        Args:
            max_words: Tokenizer için maksimum kelime sayısı
            max_sequence_length: Maksimum dizi uzunluğu
        """
        self.max_words = max_words
        self.max_sequence_length = max_sequence_length
        self.tokenizer = None
    
    def fit_tokenizer(self, texts):
        """Tokenizer'ı eğitir"""
        try:
            self.tokenizer = Tokenizer(num_words=self.max_words)
            self.tokenizer.fit_on_texts(texts)
            logger.info(f"Tokenizer {len(self.tokenizer.word_index)} kelime ile eğitildi")
            return True
        except Exception as e:
            logger.error(f"Tokenizer eğitimi sırasında hata: {str(e)}")
            return False
    
    def save_tokenizer(self, filepath):
        """Tokenizer'ı kaydet"""
        import pickle
        
        if self.tokenizer is None:
            logger.error("Kaydedilecek tokenizer bulunamadı")
            return False
        
        try:
            with open(filepath, 'wb') as handle:
                pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Tokenizer başarıyla kaydedildi: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Tokenizer kaydedilirken hata: {str(e)}")
            return False
    
    def load_tokenizer(self, filepath):
        """Kaydedilmiş bir tokenizer'ı yükle"""
        import pickle
        
        try:
            with open(filepath, 'rb') as handle:
                self.tokenizer = pickle.load(handle)
            logger.info(f"Tokenizer başarıyla yüklendi: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Tokenizer yüklenirken hata: {str(e)}")
            return False
    
    def clean_text(self, text):
        """
        Temel metin temizleme işlemleri
        """
        if not text or not isinstance(text, str):
            return ""
        
        try:
            # Küçük harfe çevir
            text = text.lower()
            
            # HTML etiketlerini kaldır
            text = re.sub(r'<.*?>', '', text)
            
            # URL'leri kaldır
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
            
            # Sayıları kaldır
            text = re.sub(r'\d+', '', text)
            
            # Noktalama işaretlerini kaldır
            translator = str.maketrans('', '', string.punctuation)
            text = text.translate(translator)
            
            # Fazla boşlukları kaldır
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        except Exception as e:
            logger.error(f"Metin temizleme sırasında hata: {str(e)}")
            return text
    
    def texts_to_sequences(self, texts):
        """Metinleri sayı dizilerine dönüştür"""
        if self.tokenizer is None:
            logger.error("Tokenizer henüz eğitilmemiş")
            return None
        
        try:
            # Temizlenmiş metinleri sayı dizilerine dönüştür
            cleaned_texts = [self.clean_text(text) for text in texts]
            sequences = self.tokenizer.texts_to_sequences(cleaned_texts)
            return sequences
        except Exception as e:
            logger.error(f"Metin dizileştirme sırasında hata: {str(e)}")
            return None
    
    def pad_sequences(self, sequences):
        """Dizileri doldur (padding)"""
        try:
            padded_sequences = pad_sequences(
                sequences, 
                maxlen=self.max_sequence_length,
                padding='post',
                truncating='post'
            )
            return padded_sequences
        except Exception as e:
            logger.error(f"Dizi doldurma sırasında hata: {str(e)}")
            return None
    
    def prepare_input(self, texts):
        """
        Metinleri model girişi için hazırla (temizle, sayı dizisine dönüştür ve doldur)
        """
        if not texts:
            return None
        
        try:
            sequences = self.texts_to_sequences(texts)
            if sequences:
                padded_sequences = self.pad_sequences(sequences)
                return padded_sequences
            return None
        except Exception as e:
            logger.error(f"Giriş hazırlama sırasında hata: {str(e)}")
            return None

# Basit kullanım örneği
def ornek_kullanim():
    texts = [
        "Bu bir örnek metin.",
        "Metin işleme için örnek kullanım.",
        "Türkçe doğal dil işleme."
    ]
    
    preprocessor = TextPreprocessor(max_words=1000, max_sequence_length=50)
    preprocessor.fit_tokenizer(texts)
    
    sequences = preprocessor.texts_to_sequences(texts)
    padded_sequences = preprocessor.pad_sequences(sequences)
    
    print("Padded Sequences:")
    print(padded_sequences)
    
    # Tokenizer'ı kaydet
    preprocessor.save_tokenizer("tokenizer.pickle")

if __name__ == "__main__":
    ornek_kullanim()