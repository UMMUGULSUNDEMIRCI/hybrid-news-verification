import os
from flask import Flask, render_template, request
import psycopg2
import requests
import re
import numpy as np
import joblib
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec

# Klasör yapısının şaşmaması için mutlak yol (absolute path) ayarı
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))

# ==========================================
# 1. GÜVENLİK, API VE VERİTABANI AYARLARI
# ==========================================
NEWSDATA_API_KEY = "pub_1740d5ba773b4c15b4c65f55ae55b886"
POSTGRES_URI = "postgresql://postgres:meoBKAAQ8aywkn83@db.pgzgqtzrvbzxlbdkrmyq.supabase.co:5432/postgres"

# NLTK verilerinin bu temiz klasöre indirilmesini zorunlu kılıyoruz
nltk.data.path.append(os.path.join(BASE_DIR, "nltk_data"))
nltk.download('stopwords', download_dir=os.path.join(BASE_DIR, "nltk_data"), quiet=True)
nltk.download('punkt', download_dir=os.path.join(BASE_DIR, "nltk_data"), quiet=True)
nltk.download('punkt_tab', download_dir=os.path.join(BASE_DIR, "nltk_data"), quiet=True)

# ==========================================
# 2. YAPAY ZEKA MODELLERİNİN YÜKLENMESİ
# ==========================================
print("📂 Yapay Zeka Modelleri Temiz Klasörden Yükleniyor...")
try:
    w2v = Word2Vec.load(os.path.join(BASE_DIR, "word2vec_teyit.model"))
    model = joblib.load(os.path.join(BASE_DIR, "random_forest_final.pkl"))
    le = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
    VECTOR_SIZE = w2v.vector_size
    print(f"✅ Modeller başarıyla bağlandı. Sınıflar: {list(le.classes_)}")
except Exception as e:
    print(f"❌ Model yükleme hatası: {e}")
    exit(1)

def nlp_metin_temizle(metin):
    if not metin or str(metin).strip() == "": return ""
    metin = str(metin).lower()
    metin = re.sub(r'[^a-zçğışüö\s]', '', metin)
    try:
        kelimeler = nltk.word_tokenize(metin, language='turkish')
    except Exception:
        kelimeler = metin.split()
    durak_kelimeler = set(stopwords.words('turkish'))
    temiz_kelimeler = [k for k in kelimeler if k not in durak_kelimeler and len(k) > 1]
    return " ".join(temiz_kelimeler)

def newsdata_sorgula(sorgu_metni):
    url = f"https://newsdata.io/api/1/latest?apikey=pub_1740d5ba773b4c15b4c65f55ae55b886"
    try:
        response = requests.get(url).json()
        sonuclar = response.get('results', [])
        toplam_haber = len(sonuclar)
        if toplam_haber >= 3: return 95.0
        elif toplam_haber == 2: return 75.0
        elif toplam_haber == 1: return 50.0
        else: return 15.0
    except Exception as e:
        return 50.0

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        haber_basligi = request.form.get("haber_basligi", "").strip()
        haber_detayi = request.form.get("haber_detayi", "").strip()
        
        if not haber_basligi:
            return render_template("index.html", error="Lütfen en azından bir haber başlığı girin.")
            
        conn = psycopg2.connect(POSTGRES_URI)
        cursor = conn.cursor()
        cursor.execute("SELECT model_skoru, api_skoru, final_dogruluk_yuzdesi, karar FROM arama_gecmisi WHERE haber_basligi = %s", (haber_basligi,))
        kayit = cursor.fetchone()
        
        if kayit:
            cursor.close()
            conn.close()
            return render_template("index.html", sonuc=True, baslik=haber_basligi, detay=haber_detayi, model_skoru=kayit[0], api_skoru=kayit[1], final_skor=kayit[2], karar=kayit[3], kaynak="Bulut Veritabanı (Önbellek)")

        birlesik_metin = haber_basligi + " " + haber_detayi
        temiz_metin = nlp_metin_temizle(birlesik_metin)
        kelimeler = temiz_metin.split()
        
        v_list = [w2v.wv[k] for k in kelimeler if k in w2v.wv]
        if len(v_list) > 0:
            vektor = np.mean(v_list, axis=0).reshape(1, -1)
        else:
            vektor = np.zeros((1, VECTOR_SIZE))
            
        olasiliklar = model.predict_proba(vektor)[0]
        
        dogru_indeks = None
        for i, sinif_ismi in enumerate(le.classes_):
            if str(sinif_ismi).lower() in ['doğru', 'dogru', 'true', 'gerçek', 'gercek']:
                dogru_indeks = i
                break
        
        if dogru_indeks is not None:
            model_skoru = float(olasiliklar[dogru_indeks] * 100)
        else:
            model_skoru = float(np.max(olasiliklar) * 100)

        api_skoru = newsdata_sorgula(haber_basligi)
        final_skor = (model_skoru * 0.40) + (api_skoru * 0.60)
        
        if final_skor >= 70: karar = "GÜVENİLİR / DOĞRU"
        elif final_skor >= 45: karar = "ŞÜPHELİ / KANIT YETERSİZ"
        else: karar = "YALAN / DEZENFORMASYON"

        try:
            cursor.execute("INSERT INTO arama_gecmisi (haber_basligi, haber_detayi, model_skoru, api_skoru, final_dogruluk_yuzdesi, karar) VALUES (%s, %s, %s, %s, %s, %s)", (haber_basligi, haber_detayi, model_skoru, api_skoru, final_skor, karar))
            conn.commit()
        except Exception as e:
            print(f"Veritabanı hatası: {e}")
            
        cursor.close()
        conn.close()
        
        return render_template("index.html", sonuc=True, baslik=haber_basligi, detay=haber_detayi, model_skoru=model_skoru, api_skoru=api_skoru, final_skor=final_skor, karar=karar, kaynak="Canlı Analiz (Yapay Zeka + Newsdata API)")
                               
    return render_template("index.html", sonuc=False)

if __name__ == "__main__":
    app.run(debug=True)