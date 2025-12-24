# 🍽️ Ödev 2 – YOLOv8 Tabanlı Mutfak Eşyası Tespit Uygulaması

Bu proje, **YOLOv8** kullanılarak eğitilmiş bir derin öğrenme modeli ile **çatal, kaşık ve tabak** nesnelerini tespit eden ve **PyQt5 tabanlı grafik arayüz (GUI)** üzerinden çalışan bir masaüstü uygulamasıdır.

Proje iki ana aşamadan oluşmaktadır:

1. **YOLOv8 Model Eğitimi (Google Colab)**
2. **Eğitilen Model ile PyQt5 GUI Uygulaması**

---

## 📌 Proje Amacı

- Gerçek görüntüler üzerinde **mutfak eşyası tespiti** yapmak  
- YOLOv8 ile **özel veri seti** kullanarak model eğitmek  
- Kullanıcı dostu bir **grafik arayüz** ile:
  - Görüntü yükleme  
  - Nesne tespiti  
  - Fare ile bölge (ROI) seçimi  
  - Güven skoruna göre filtreleme  
  - Sonuçları listeleme  

---

## 🧠 Kullanılan Teknolojiler

- **Python 3.10+**
- **YOLOv8 (Ultralytics)**
- **PyTorch**
- **OpenCV**
- **PyQt5**
- **Matplotlib**
- **Roboflow (Bounding Box etiketleme)**

---

## 📂 Dataset Hazırlama Süreci

- Veri seti **Dataset1.zip** içerisinde yer almaktadır.
- İçerik:
  - `catal` (çatal)
  - `kasik` (kaşık)
  - `tabak` (tabak)
- Tüm görüntüler **Roboflow** platformu kullanılarak:
  - Tek tek **Bounding Box** ile etiketlenmiştir
  - **YOLOv8 formatında** dışa aktarılmıştır

Her sınıf için ayrı YOLO formatlı veri setleri oluşturulmuş, ardından **tek bir birleşik veri seti** haline getirilmiştir.

---

## 🗂️ Birleştirilmiş Dataset Yapısı


yolo_dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── data.yaml


YOLOv8 Model Eğitimi

Model eğitimi Google Colab ortamında gerçekleştirilmiştir.

Eğitim Ayarları

Model: yolov8n.pt

Epoch: 40

Image Size: 640

Batch Size: GPU varsa 16, yoksa 8

Early Stopping: patience = 20

Confidence Threshold: 0.4

Eğitim Komutu
model = YOLO('yolov8n.pt')
model.train(
    data='data.yaml',
    epochs=40,
    imgsz=640,
    batch=16,
    device=0
)

📊 Model Performans Metrikleri

Eğitim sonrası en iyi model (best.pt) doğrulama verisi üzerinde test edilmiştir.

Örnek metrikler:

mAP@50

mAP@50–95

Confusion Matrix

Loss ve Accuracy grafikleri

Elde edilen en iyi model dosyası:

best_utensil_colab.pt

💾 Modelin Uygulamaya Entegrasyonu

Eğitilen model dosyası:

best_utensil_colab.pt


GUI uygulamasının bulunduğu klasöre kopyalanır ve aşağıdaki satırda kullanılır:

self.model = YOLO("best_utensil_colab.pt")

🖥️ PyQt5 GUI Uygulaması Özellikleri
✔️ Temel Özellikler

Görüntü yükleme

Tüm görüntü üzerinde nesne tespiti

Confidence filtresi (≥ 0.4)

En yüksek güven skoruna sahip sınıfın seçilmesi

Sonuçların liste halinde gösterilmesi

✔️ Gelişmiş Özellikler

Fare ile dikdörtgen (ROI) çizimi

Seçili bölge için ayrı analiz

ROI için daha yüksek confidence eşiği (≥ 0.5)

Seçilen bölgede baskın nesnenin gösterilmesi

Görüntüyü kaydetme

🖱️ Kullanım Adımları

Uygulamayı başlat:

python gui_app.py


Resim Yükle butonu ile bir görüntü seç

Analizi Başlat ile tüm görüntüyü analiz et

Fare ile görüntü üzerinde alan çizerek bölgesel analiz yap

Sonuçları sağ taraftaki listede incele

İstersen sonucu Kaydet

📸 Örnek Çıktılar

Tespit edilen nesne adı

Güven skoru (confidence)

Seçili bölge sonucu

Örnek:

Doğru Tahmin: tabak (Güven: 0.87)
Seçili Bölge: kasik (92%)

Sonuç

Bu proje kapsamında:

YOLOv8 ile özel veri seti kullanılarak model eğitilmiş

Eğitilen model gerçek zamanlı GUI uygulamasına entegre edilmiş

ROI destekli, confidence filtreli ve kullanıcı dostu bir nesne tespit sistemi geliştirilmiştir
