import os
import re
import psycopg2
import requests
import joblib
import numpy as np
from flask import Flask, render_template, request
from gensim.models import Word2Vec
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.corpus import stopwords

app = Flask(__name__)
                    
NEWSDATA_API_KEY = "pub_1740d5ba773b4c15b4c65f55ae55b886"
POSTGRES_URI = "postgresql://postgres.pgzgqtzrvbzxlbdkrmyq:meoBKAAQ8aywkn83@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# --- MODELLERİN HAFIZAYA ALINMASI ---
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    w2v_model = Word2Vec.load(os.path.join(BASE_DIR, "word2vec_teyit.model"))
    rf_model = joblib.load(os.path.join(BASE_DIR, "random_forest_final.pkl"))
    label_encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
    print(f"Modeller başarıyla bağlandı. Sınıflar: {list(label_encoder.classes_)}")
except Exception as e:
    print(f"Model yükleme hatası: {e}")
    exit(1)

# --- METİN ÖN İŞLEME ---
def temizle_ve_vektorize_et(text, model, vector_size=100):
    if not text or str(text).strip() == "":
        return np.zeros(vector_size)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    stop_words = set(stopwords.words('turkish'))
    kelimeler = [k for k in text.split() if k not in stop_words]
    vektorler = [model.wv[k] for k in kelimeler if k in model.wv]
    return np.mean(vektorler, axis=0) if vektorler else np.zeros(vector_size)

# --- NEWSDATA MEDYA TARAMA ---
def medya_tara(haber_basligi):
    """
    Newsdata API'den dönen ilk 10 sonuca bakarak medya durumunu belirler.
    Döndürdüğü değer: medya_durumu (str)

    Olası çıktılar:
    - "Güvenilir medyada yayınlandı: aa.com.tr, ntv.com.tr"
    - "Yalnızca doğrulama platformunda yayınlandı: teyit.org"
    - "Medyada sonuç bulunamadı"
    """
    guvenilir_kaynaklar = [
        "aa.com.tr", "trthaber.com", "bbc.com", "bbc.com/turkce",
        "sozcu.com.tr", "hurriyet.com.tr", "milliyet.com.tr",
        "haberturk.com", "ntv.com.tr", "cumhuriyet.com.tr"
    ]
    dogrulama_platformlari = ["teyit.org", "dogrulukpayi.com"]

    sonuclar = []
    try:
        url = (
            f"https://newsdata.io/api/1/latest"
            f"?apikey={NEWSDATA_API_KEY}"
            f"&q={requests.utils.quote(haber_basligi)}"
            f"&language=tr"
        )
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            sonuclar = r.json().get("results", [])
            print(f"   Newsdata: {len(sonuclar)} sonuç")
    except Exception as e:
        print(f"   Newsdata API hatası: {e}")

    if not sonuclar:
        return "Medyada sonuç bulunamadı"

    aranan_kelimeler = set(haber_basligi.lower().split())
    bulunan_guvenilir = []
    bulunan_dogrulama = []

    for sonuc in sonuclar:
        link = sonuc.get("link", "").lower()
        baslik = sonuc.get("title", "").lower()
        ortak = set(baslik.split()) & aranan_kelimeler

        # Güvenilir medya — en az 2 ortak kelime şartı
        for domain in guvenilir_kaynaklar:
            if domain in link and len(ortak) >= 2:
                bulunan_guvenilir.append(domain)
                break

        # Doğrulama platformu — kelime şartı yok
        for domain in dogrulama_platformlari:
            if domain in link:
                bulunan_dogrulama.append(domain)
                break

    if bulunan_guvenilir:
        siteler = ", ".join(sorted(set(bulunan_guvenilir)))
        return f"Güvenilir medyada yayınlandı: {siteler}"

    if bulunan_dogrulama:
        siteler = ", ".join(sorted(set(bulunan_dogrulama)))
        return f"Yalnızca doğrulama platformunda yayınlandı: {siteler}"

    return "Medyada sonuç bulunamadı"


# --- ANA SAYFA ---
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    haber_basligi = request.form.get("title", "").strip()
    haber_detayi = request.form.get("content", "").strip()

    if not haber_basligi:
        return render_template("index.html", error="Lütfen en azından bir haber başlığı girin.")

    # --- ADIM A: ÖNBELLEK KONTROLÜ ---
    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(POSTGRES_URI, connect_timeout=5)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT model_skoru, medya_durumu, karar FROM arama_gecmisi WHERE haber_basligi = %s",
            (haber_basligi,)
        )
        kayit = cursor.fetchone()
        if kayit:
            cursor.close()
            conn.close()
            return render_template(
                "index.html", sonuc=True,
                baslik=haber_basligi, detay=haber_detayi,
                model_skoru=kayit[0], medya_durumu=kayit[1], karar=kayit[2],
                kaynak="Bulut Veritabanı (Önbellek)"
            )
    except Exception as db_err:
        print(f"Önbellek okuma hatası: {db_err}")
        conn = None
        cursor = None

    # --- ADIM B: ML MODELİ ---
    tam_metin = haber_basligi + " " + haber_detayi
    metin_vektoru = temizle_ve_vektorize_et(tam_metin, w2v_model).reshape(1, -1)
    olasiliklar = rf_model.predict_proba(metin_vektoru)[0]

    dogru_indeksi = None
    for i, sinif in enumerate(label_encoder.classes_):
        if str(sinif).lower() in ['doğru', 'dogru', 'true', 'gerçek', 'gercek']:
            dogru_indeksi = i
            break
    model_skoru = round(
        float(olasiliklar[dogru_indeksi]) * 100 if dogru_indeksi is not None
        else float(np.max(olasiliklar)) * 100,
        2
    )

    # --- ADIM C: MEDYA TARAMA ---
    print(f"   medya_tara çağrılıyor: '{haber_basligi}'")
    medya_durumu = medya_tara(haber_basligi)
    print(f"   Medya durumu: {medya_durumu}")

    # --- ADIM D: KARAR ---
    if model_skoru >= 70:
        karar = "GÜVENİLİR / DOĞRULANMIŞ HABER"
    elif model_skoru >= 45:
        karar = "ŞÜPHELİ / KANIT YETERSİZ"
    else:
        karar = "YANLIŞ / DEZENFORMASYON"

    # --- ADIM E: VERİTABANINA KAYDET ---
    if conn and cursor:
        try:
            cursor.execute(
                """INSERT INTO arama_gecmisi 
                   (haber_basligi, haber_detayi, model_skoru, medya_durumu, karar) 
                   VALUES (%s, %s, %s, %s, %s)""",
                (haber_basligi, haber_detayi, model_skoru, medya_durumu, karar)
            )
            conn.commit()
        except Exception as save_err:
            print(f"Veritabanı kaydetme hatası: {save_err}")
        finally:
            cursor.close()
            conn.close()

    return render_template(
        "index.html", sonuc=True,
        baslik=haber_basligi, detay=haber_detayi,
        model_skoru=model_skoru, medya_durumu=medya_durumu, karar=karar,
        kaynak="Canlı Analiz (Yapay Zeka + Medya Tarama)"
    )

if __name__ == "__main__":
    app.run(debug=True)
