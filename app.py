import os
import re
import psycopg2
import requests
import joblib
import numpy as np
from flask import Flask, render_template, request
from gensim.models import Word2Vec
import nltk
from bs4 import BeautifulSoup

# --- NLTK BULUT AYARI ---
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.corpus import stopwords

app = Flask(__name__)

# --- SİSTEM AYARLARI VE BAĞLANTILAR ---
NEWSDATA_API_KEY = "pub_43034969efcc5b0267f56cf8f5413df18b955"
POSTGRES_URI = "postgresql://postgres.pgzgqtzrvbzxlbdkrmyq:meoBKAAQ8aywkn83@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres"

# --- MODELLERİN HAFIZAYA ALINMASI ---
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    w2v_model = Word2Vec.load(os.path.join(BASE_DIR, "word2vec_teyit.model"))
    rf_model = joblib.load(os.path.join(BASE_DIR, "random_forest_final.pkl"))
    label_encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
    print(f"✅ Modeller başarıyla bağlandı. Sınıflar: {list(label_encoder.classes_)}")
except Exception as e:
    print(f"❌ Model yükleme hatası: {e}")

# --- METİN ÖN İŞLEME FONKSİYONU ---
def temizle_ve_vektorize_et(text, model, vector_size=100):
    if not text or str(text).strip() == "":
        return np.zeros(vector_size)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    stop_words = set(stopwords.words('turkish'))
    kelimeler = [k for k in text.split() if k not in stop_words]
    vektorler = [model.wv[k] for k in kelimeler if k in model.wv]
    if len(vektorler) > 0:
        return np.mean(vektorler, axis=0)
    else:
        return np.zeros(vector_size)

# --- DOĞRULAMA PLATFORMU SCRAPING FONKSİYONU ---
def dogrulama_platformu_tara(kaynak_linki):
    """
    teyit.org veya dogrulukpayi.com linkine gidip etiket mührünü okur.
    Döndürdüğü değer: (api_skoru: float, basarili: bool)
    """
    olumsuz_kelimeler = [
        "yanlış", "yanlis", "yalan", "uydurma",
        "asılsız", "asilsiz", "iddia yanlış", "iddia yanlis",
        "doğru değil", "dogru degil", "gerçek değil",
        "manipülasyon", "manipulasyon", "montaj", "parodi"
    ]
    olumlu_kelimeler = [
        "doğru", "dogru", "gerçek", "gercek",
        "onaylandı", "doğrulandı"
    ]
    belirsiz_kelimeler = [
        "kısmen", "yanıltıcı", "bağlam", "abartı", "karma"
    ]

    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        sayfa = requests.get(kaynak_linki, headers=headers, timeout=5)

        if sayfa.status_code != 200:
            return None, False  # Sayfaya erişilemedi → fallback

        soup = BeautifulSoup(sayfa.text, 'html.parser')
        bulunan_muhur = ""

        if "teyit.org" in kaynak_linki:
            # <span class="text-uppercase">Yanlış</span>
            element = soup.find("span", class_="text-uppercase")
            if element:
                bulunan_muhur = element.get_text(strip=True).lower()

        elif "dogrulukpayi.com" in kaynak_linki:
            # <div class="user-content">İddia Yanlış</div>
            element = soup.find("div", class_="user-content")
            if element:
                bulunan_muhur = element.get_text(strip=True).lower()[:50]

        if not bulunan_muhur:
            return 40.0, True  # Sayfa açıldı ama etiket bulunamadı

        if any(k in bulunan_muhur for k in olumsuz_kelimeler):
            return 0.0, True   # Kesin yalan
        elif any(k in bulunan_muhur for k in olumlu_kelimeler):
            return 100.0, True  # Kesin doğru
        elif any(k in bulunan_muhur for k in belirsiz_kelimeler):
            return 35.0, True   # Kısmen doğru / yanıltıcı
        else:
            return 40.0, True   # Etiket okunamadı

    except Exception as scrap_error:
        print(f"Scraping hatası: {scrap_error}")
        return None, False  # Hata → fallback

# --- ANA SAYFA VE ANALİZ MOTORU ---
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    haber_basligi = request.form.get("title", "").strip()
    haber_detayi = request.form.get("content", "").strip()

    if not haber_basligi:
        return render_template("index.html", error="Lütfen en azından bir haber başlığı girin.")

    # --- ADIM A: POSTGRESQL ÖNBELLEK KONTROLÜ ---
    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(POSTGRES_URI, connect_timeout=5)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT model_skoru, api_skoru, final_dogruluk_yuzdesi, karar FROM arama_gecmisi WHERE haber_basligi = %s",
            (haber_basligi,)
        )
        kayit = cursor.fetchone()
        if kayit:
            cursor.close()
            conn.close()
            return render_template(
                "index.html", sonuc=True, baslik=haber_basligi, detay=haber_detayi,
                model_skoru=kayit[0], api_skoru=kayit[1], final_skor=kayit[2], karar=kayit[3],
                kaynak="Bulut Veritabanı (Önbellek)"
            )
    except Exception as db_err:
        print(f"⚠️ Önbellek okuma hatası: {db_err}")
        conn = None
        cursor = None

    # --- ADIM B: MAKİNE ÖĞRENMESİ TAHMİNİ ---
    tam_metin = haber_basligi + " " + haber_detayi
    metin_vektoru = temizle_ve_vektorize_et(tam_metin, w2v_model).reshape(1, -1)
    olasiliklar = rf_model.predict_proba(metin_vektoru)[0]

    # Label encoder'daki 'DOĞRU' sınıfının indeksini bul
    dogru_indeksi = None
    for i, sinif in enumerate(label_encoder.classes_):
        if str(sinif).lower() in ['doğru', 'dogru', 'true', 'gerçek', 'gercek']:
            dogru_indeksi = i
            break
    if dogru_indeksi is not None:
        model_skoru = round(float(olasiliklar[dogru_indeksi]) * 100, 2)
    else:
        model_skoru = round(float(np.max(olasiliklar)) * 100, 2)

    # --- ADIM C: NEWSDATA API TARAMASI ---
    sonuclar = []
    try:
        url = f"https://newsdata.io/api/1/latest?apikey={NEWSDATA_API_KEY}&q={requests.utils.quote(haber_basligi)}&language=tr"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            sonuclar = response.json().get("results", [])
    except Exception as api_err:
        print(f"API bağlantı hatası: {api_err}")

    # --- ADIM D: MEDYA TARAMA VE SCRAPING MOTORU ---
    guvenilir_kaynaklar = [
        "aa.com.tr", "trthaber.com", "bbc.com/turkce", "sozcu.com.tr",
        "hurriyet.com.tr", "milliyet.com.tr", "haberturk.com",
        "ntv.com.tr", "cumhuriyet.com.tr"
    ]
    dogrulama_platformlari = ["teyit.org", "dogrulukpayi.com"]

    api_skoru = 20.0
    nlp_sadece_calissin = False

    if sonuclar:
        aranan_baslik = haber_basligi.lower()

        # ✅ Tüm sonuçlara bak, sadece ilkine değil
        for sonuc in sonuclar:
            kaynak_linki = sonuc.get("link", "").lower()
            bulunan_baslik = sonuc.get("title", "").lower()
            ortak_kelimeler = set(bulunan_baslik.split()) & set(aranan_baslik.split())

            # --- DURUM 1: DOĞRULAMA PLATFORMU ---
            if any(domain in kaynak_linki for domain in dogrulama_platformlari):
                skor, basarili = dogrulama_platformu_tara(kaynak_linki)
                if basarili:
                    api_skoru = skor
                else:
                    nlp_sadece_calissin = True  # Sayfaya erişilemedi → sadece ML
                break  # Doğrulama platformu bulunduysa diğerlerine bakma

            # --- DURUM 2: GÜVENİLİR MEDYA ---
            elif (
                any(domain in kaynak_linki for domain in guvenilir_kaynaklar)
                and len(ortak_kelimeler) >= 2  # 3'ten 2'ye düşürüldü
            ):
                api_skoru = 90.0
                break

        # Hiçbir eşleşme yoksa varsayılan 20.0 kalır
    else:
        api_skoru = 5.0  # İnternette hiç sonuç yok

    # --- ADIM E: NİHAİ HİBRİT KARAR ---
    if nlp_sadece_calissin:
        final_skor = round(model_skoru, 2)
        kaynak_metni = "Canlı Analiz (Sadece Yapay Zeka - NLP)"
    else:
        final_skor = round((model_skoru * 0.50) + (api_skoru * 0.50), 2)
        kaynak_metni = "Canlı Analiz (Hibrit: Yapay Zeka + Canlı Web Scraping)"

    if final_skor >= 70:
        karar = "GÜVENİLİR / DOĞRULANMIŞ HABER"
    elif final_skor >= 45:
        karar = "ŞÜPHELİ / KANIT YETERSİZ"
    else:
        karar = "YANLIŞ / DEZENFORMASYON"

    # --- ADIM F: POSTGRESQL'E KAYDET ---
    if conn and cursor:
        try:
            cursor.execute(
                "INSERT INTO arama_gecmisi (haber_basligi, haber_detayi, model_skoru, api_skoru, final_dogruluk_yuzdesi, karar) VALUES (%s, %s, %s, %s, %s, %s)",
                (haber_basligi, haber_detayi, model_skoru, api_skoru, final_skor, karar)
            )
            conn.commit()
        except Exception as save_err:
            print(f"Veritabanı kaydetme hatası: {save_err}")
        finally:
            cursor.close()
            conn.close()

    return render_template(
        "index.html", sonuc=True, baslik=haber_basligi, detay=haber_detayi,
        model_skoru=model_skoru, api_skoru=api_skoru, final_skor=final_skor,
        karar=karar, kaynak=kaynak_metni
    )

if __name__ == "__main__":
    app.run(debug=True)
