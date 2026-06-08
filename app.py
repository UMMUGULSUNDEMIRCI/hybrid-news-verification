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

# --- SİSTEM AYARLARI ---
NEWSDATA_API_KEY = "pub_43034969efcc5b0267f56cf8f5413df18b955"
POSTGRES_URI = "postgresql://postgres.pgzgqtzrvbzxlbdkrmyq:KENDI_SIFREN@aws-0-eu-central-1.pooler.supabase.com:6543/postgres"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

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

# --- ETİKET OKUMA YARDIMCI FONKSİYONU ---
def etiketten_skor_cikart(bulunan_muhur):
    """Okunan mühür metninden skor döndürür."""
    olumsuz = [
        "yanlış", "yanlis", "yalan", "uydurma", "asılsız", "asilsiz",
        "iddia yanlış", "doğru değil", "dogru degil", "gerçek değil",
        "manipülasyon", "manipulasyon", "montaj", "parodi"
    ]
    olumlu = ["doğru", "dogru", "gerçek", "gercek", "onaylandı", "doğrulandı"]
    belirsiz = ["kısmen", "yanıltıcı", "bağlam", "abartı", "karma"]

    if any(k in bulunan_muhur for k in olumsuz):
        return 0.0
    elif any(k in bulunan_muhur for k in olumlu):
        return 100.0
    elif any(k in bulunan_muhur for k in belirsiz):
        return 35.0
    else:
        return 40.0  # Etiket okunamadı ama sayfa açıldı

# --- TEYİT.ORG DOĞRUDAN ARAMA FONKSİYONU (Newsdata'dan bağımsız) ---
def teyit_org_ara(haber_basligi):
    """
    Teyit.org'un kendi arşivinde haberi arar.
    Önce WordPress REST API'yi dener, başarısız olursa ?s= arama sayfasını scrape eder.
    Döndürdüğü değer: (skor: float | None, kaynak_aciklama: str)
    """
    sorgu = requests.utils.quote(haber_basligi)

    # --- YÖNTEM 1: WordPress REST API (en temiz yol) ---
    try:
        api_url = f"https://teyit.org/wp-json/wp/v2/posts?search={sorgu}&per_page=3&_fields=link,title"
        r = requests.get(api_url, headers=HEADERS, timeout=6)
        if r.status_code == 200:
            posts = r.json()
            if posts and isinstance(posts, list):
                link = posts[0].get("link", "")
                print(f"✅ Teyit WP-API: {link}")
                skor, basarili = teyit_sayfasini_oku(link)
                if basarili:
                    return skor, f"Teyit.org Arşiv (WP-API): {link}"
    except Exception as e:
        print(f"Teyit WP-API hatası: {e}")

    # --- YÖNTEM 2: Arama Sayfası Scraping (?s= parametresi) ---
    try:
        arama_url = f"https://teyit.org/?s={sorgu}"
        r = requests.get(arama_url, headers=HEADERS, timeout=6)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, 'html.parser')
            # Arama sonuçlarındaki ilk analiz linkini bul
            link_elementi = soup.find("a", href=lambda h: h and "/analiz/" in h)
            if link_elementi:
                link = link_elementi["href"]
                print(f"✅ Teyit Arama Scrape: {link}")
                skor, basarili = teyit_sayfasini_oku(link)
                if basarili:
                    return skor, f"Teyit.org Arşiv (Arama): {link}"
    except Exception as e:
        print(f"Teyit arama scrape hatası: {e}")

    return None, "Teyit.org'da bulunamadı"

# --- TEYİT.ORG SAYFA OKUMA FONKSİYONU ---
def teyit_sayfasini_oku(link):
    """
    Verilen teyit.org linkine gidip etiket mührünü okur.
    Döndürdüğü değer: (skor: float, basarili: bool)
    """
    try:
        r = requests.get(link, headers=HEADERS, timeout=6)
        if r.status_code != 200:
            return None, False
        soup = BeautifulSoup(r.text, 'html.parser')
        element = soup.find("span", class_="text-uppercase")
        if element:
            muhur = element.get_text(strip=True).lower()
            print(f"   Teyit mühürü: '{muhur}'")
            return etiketten_skor_cikart(muhur), True
        return 40.0, True  # Sayfa açıldı ama etiket bulunamadı
    except Exception as e:
        print(f"Teyit sayfa okuma hatası: {e}")
        return None, False

# --- DOĞRULUK PAYI SCRAPING FONKSİYONU ---
def dogrulukpayi_tara(kaynak_linki):
    """
    Dogrulukpayi.com sayfasını okur.
    Döndürdüğü değer: (skor: float, basarili: bool)
    """
    try:
        r = requests.get(kaynak_linki, headers=HEADERS, timeout=6)
        if r.status_code != 200:
            return None, False
        soup = BeautifulSoup(r.text, 'html.parser')
        element = soup.find("div", class_="user-content")
        if element:
            muhur = element.get_text(strip=True).lower()[:50]
            print(f"   Doğruluk Payı mühürü: '{muhur}'")
            return etiketten_skor_cikart(muhur), True
        return 40.0, True
    except Exception as e:
        print(f"Doğruluk Payı scrape hatası: {e}")
        return None, False

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

    dogru_indeksi = None
    for i, sinif in enumerate(label_encoder.classes_):
        if str(sinif).lower() in ['doğru', 'dogru', 'true', 'gerçek', 'gercek']:
            dogru_indeksi = i
            break
    if dogru_indeksi is not None:
        model_skoru = round(float(olasiliklar[dogru_indeksi]) * 100, 2)
    else:
        model_skoru = round(float(np.max(olasiliklar)) * 100, 2)

    # --- ADIM C: TEYİT.ORG DOĞRUDAN ARŞİV TARAMASI (Newsdata'dan bağımsız!) ---
    api_skoru = None
    kaynak_metni = ""
    nlp_sadece_calissin = False

    teyit_skor, teyit_aciklama = teyit_org_ara(haber_basligi)
    if teyit_skor is not None:
        api_skoru = teyit_skor
        kaynak_metni = f"Canlı Analiz (Hibrit: NLP + {teyit_aciklama})"
        print(f"✅ Teyit.org skoru: {api_skoru} | {teyit_aciklama}")
    else:
        print("ℹ️ Teyit.org'da bulunamadı, Newsdata + medya taramasına geçiliyor...")

        # --- ADIM D: NEWSDATA API + MEDYA TARAMASI (Fallback) ---
        sonuclar = []
        try:
            url = f"https://newsdata.io/api/1/latest?apikey={NEWSDATA_API_KEY}&q={requests.utils.quote(haber_basligi)}&language=tr"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                sonuclar = response.json().get("results", [])
        except Exception as api_err:
            print(f"Newsdata API hatası: {api_err}")

        guvenilir_kaynaklar = [
            "aa.com.tr", "trthaber.com", "bbc.com/turkce", "sozcu.com.tr",
            "hurriyet.com.tr", "milliyet.com.tr", "haberturk.com",
            "ntv.com.tr", "cumhuriyet.com.tr"
        ]

        api_skoru = 20.0  # varsayılan

        if sonuclar:
            aranan_baslik = haber_basligi.lower()
            for sonuc in sonuclar:
                kaynak_linki = sonuc.get("link", "").lower()
                bulunan_baslik = sonuc.get("title", "").lower()
                ortak_kelimeler = set(bulunan_baslik.split()) & set(aranan_baslik.split())

                # Newsdata üzerinden gelen dogrulukpayi linki
                if "dogrulukpayi.com" in kaynak_linki:
                    skor, basarili = dogrulukpayi_tara(kaynak_linki)
                    if basarili:
                        api_skoru = skor
                        kaynak_metni = "Canlı Analiz (Hibrit: NLP + Doğruluk Payı Scraping)"
                    else:
                        nlp_sadece_calissin = True
                    break

                # Güvenilir medya kontrolü
                elif (
                    any(domain in kaynak_linki for domain in guvenilir_kaynaklar)
                    and len(ortak_kelimeler) >= 2
                ):
                    api_skoru = 90.0
                    kaynak_metni = "Canlı Analiz (Hibrit: NLP + Güvenilir Medya)"
                    break
            else:
                kaynak_metni = "Canlı Analiz (Hibrit: NLP + Genel Medya Tarama)"
        else:
            api_skoru = 5.0
            kaynak_metni = "Canlı Analiz (Hibrit: NLP + Medyada Bulunamadı)"

    # --- ADIM E: NİHAİ HİBRİT KARAR ---
    if nlp_sadece_calissin or api_skoru is None:
        final_skor = round(model_skoru, 2)
        kaynak_metni = "Canlı Analiz (Sadece Yapay Zeka - NLP)"
    else:
        final_skor = round((model_skoru * 0.50) + (api_skoru * 0.50), 2)

    if final_skor >= 70:
        karar = "GÜVENİLİR / DOĞRULANMIŞ HABER"
    elif final_skor >= 45:
        karar = "ŞÜPHELİ / KANIT YETERSİZ"
    else:
        karar = "YANLIŞ / DEZENFORMASYON"

    # api_skoru None ise ekranda göstermek için 0 yap
    api_skoru_goster = api_skoru if api_skoru is not None else 0.0

    # --- ADIM F: POSTGRESQL'E KAYDET ---
    if conn and cursor:
        try:
            cursor.execute(
                "INSERT INTO arama_gecmisi (haber_basligi, haber_detayi, model_skoru, api_skoru, final_dogruluk_yuzdesi, karar) VALUES (%s, %s, %s, %s, %s, %s)",
                (haber_basligi, haber_detayi, model_skoru, api_skoru_goster, final_skor, karar)
            )
            conn.commit()
        except Exception as save_err:
            print(f"Veritabanı kaydetme hatası: {save_err}")
        finally:
            cursor.close()
            conn.close()

    return render_template(
        "index.html", sonuc=True, baslik=haber_basligi, detay=haber_detayi,
        model_skoru=model_skoru, api_skoru=api_skoru_goster, final_skor=final_skor,
        karar=karar, kaynak=kaynak_metni
    )

if __name__ == "__main__":
    app.run(debug=True)
