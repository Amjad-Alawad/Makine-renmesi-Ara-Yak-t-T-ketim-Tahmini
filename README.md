# 🚗 Araç Yakıt Tüketimi ve Maliyet Analiz Sistemi (Fuel Consumption & Cost ML Pipeline)

Bu proje, Kanada Hükümeti'nin Açık Veri Portalı (MY2022 Fuel Consumption Ratings) veri seti kullanılarak geliştirilmiş uçtan uca bir **Makine Öğrenmesi (Machine Learning) ve Web Entegrasyonu (Flask)** projesidir. Proje, araç özelliklerine (motor hacmi, vites, yakıt tipi vb.) dayanarak ortalama yakıt tüketimini (L/100 km) tahmin eder ve güncel yakıt fiyatları üzerinden kilometre başına maliyet analizi sunar.

---

## 📊 Proje Mimarisi ve Aşamaları

1. **Veri Ön İşleme ve Temizlik (Data Preprocessing):**
   - Eksik değerlerin ve tekrarlayan kayıtların temizlenmesi[cite: 1].
   - Aykırı değerlerin (Outliers) **LOF (Local Outlier Factor)** ve istatistiksel yöntemlerle ayıklanması[cite: 1].
   - Hedef sızıntısına (Data Leakage) yol açabilecek sütunların (`FC_City`, `FC_Hwy`, `CO2` vb.) modelden arındırılması[cite: 1].

2. **Özellik Mühendisliği ve Seçimi (Feature Engineering & Selection):**
   - Çoklu doğrusal bağlantı (Multicollinearity) analizi için **VIF (Variance Inflation Factor)** ve korelasyon matrisleri kullanıldı[cite: 1].
   - Pearson Korelasyonu, Karşılıklı Bilgi (Mutual Information), Ağaç tabanlı önem skorları ve Lasso katsayıları birleştirilerek en kararlı **6 özellik** seçildi:
     - `Engine_Size` (Motor Hacmi)[cite: 1]
     - `Fuel_Type_E` (Yakıt Türü - Ethanol)[cite: 1]
     - `Vehicle_Class_Ord` (Araç Sınıfı)[cite: 1]
     - `Trans_Type_AV` (Şanzıman Tipi)[cite: 1]
     - `Gears` (Vites Sayısı)[cite: 1]
     - `Make_Freq` (Marka Frekansı)[cite: 1]

3. **Model Eğitimi ve Karşılaştırma:**
   - Linear Regression, SVR, Decision Tree, XGBoost, LightGBM ve **Random Forest** gibi modeller eğitildi[cite: 1].
   - **Validation RMSE** metriklerine göre en başarılı sonuç veren **Random Forest** modeli en iyi model (`best_model`) olarak seçildi ve kaydedildi[cite: 1].

4. **Web Uygulaması (Deployment & UI):**
   - Eğitilen model **Flask** altyapısı ile web ortamına taşındı.
   - Kullanıcı dostu, modern koyu tema (Dark Mode) arayüzü ile anlık simülasyon ve maliyet hesaplama imkanı sunuldu.

---

## 🛠️ Kullanılan Teknolojiler ve Kütüphaneler

* **Dil:** Python 3.11+
* **Veri Bilimi & ML:** Pandas, NumPy, Scikit-Learn, XGBoost, Joblib
* **Web Çatı (Backend):** Flask
* **Arayüz (Frontend):** HTML5, Bootstrap 5, FontAwesome

---

## 📂 Proje Klasör Yapısı

```text
Odev Sonuc/
│
├── app.py                 # Flask web sunucusu ve tahmin API'si
├── best_model.pkl         # Eğitilmiş en iyi model (Random Forest)
├── requirements.txt       # Gerekli Python kütüphaneleri
├── arac_tuketim_ml_pipeline.ipynb  # Jupyter Notebook (Eğitim ve Analiz)
└── templates/
    └── index.html         # Web arayüzü tasarımı
