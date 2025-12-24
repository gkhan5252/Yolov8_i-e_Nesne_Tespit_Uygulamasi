import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog,
    QHBoxLayout, QVBoxLayout, QWidget, QListWidget, QComboBox
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QRect, QPoint

# TensorFlow uyarılarını bastır
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


class ClickableLabel(QLabel):
    """Target Image üzerinde dikdörtgen çizebilmek için genişletilmiş QLabel."""

    def __init__(self):
        super().__init__()
        self.start_point = None
        self.end_point = None
        self.drawing = False
        self.rect_ready = False
        self.current_pixmap = None

    def setImage(self, pixmap):
        self.current_pixmap = pixmap
        self.setPixmap(pixmap)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_point = event.pos()
            self.end_point = event.pos()
            self.drawing = True
            self.rect_ready = False
            self.update()

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.end_point = event.pos()
            self.drawing = False
            self.rect_ready = True
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)

        if self.drawing or self.rect_ready:
            painter = QPainter(self)
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))

            rect = QRect(self.start_point, self.end_point)
            painter.drawRect(rect)


class YOLOGui(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("YOLO Gelişmiş Birleştirme ve Filtreleme")
        self.setGeometry(100, 100, 1280, 720)

        self.model = YOLO("best_utensil_colab.pt")
        self.image = None

        # Sol görüntü
        self.label_input = QLabel()
        self.label_input.setAlignment(Qt.AlignCenter)
        self.label_input.setStyleSheet("background-color: #000000;")

        # Sağ görüntü (fare ile çizim yapılabilen)
        self.label_result = ClickableLabel()
        self.label_result.setAlignment(Qt.AlignCenter)
        self.label_result.setStyleSheet("background-color: #000000;")

        # Nesne listesi
        self.list_widget = QListWidget()

        # Butonlar
        self.btn_load = QPushButton("Resim Yükle")
        self.btn_load.clicked.connect(self.load_image)

        self.btn_start = QPushButton("Analizi Başlat")
        self.btn_start.clicked.connect(self.analyze_image)

        self.btn_save = QPushButton("Kaydet")
        self.btn_save.clicked.connect(self.save_image)

        # Birleştirme dropdown
        self.combo_merge = QComboBox()
        self.combo_merge.addItems(["Birleştirme (Baskın Nesne)"])

        # Layout
        h_layout_img = QHBoxLayout()
        h_layout_img.addWidget(self.label_input, 3)
        h_layout_img.addWidget(self.label_result, 3)
        h_layout_img.addWidget(self.list_widget, 1)

        h_bottom = QHBoxLayout()
        h_bottom.addWidget(self.btn_load)
        h_bottom.addWidget(self.combo_merge)
        h_bottom.addWidget(self.btn_start)
        h_bottom.addWidget(self.btn_save)

        v_layout = QVBoxLayout()
        v_layout.addLayout(h_layout_img)
        v_layout.addLayout(h_bottom)

        container = QWidget()
        container.setLayout(v_layout)
        self.setCentralWidget(container)

        # Sağ tarafta dikdörtgen seçimi tamamlandığında kontrol
        self.label_result.mouseReleaseEvent = self.region_selected

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Resim Seç", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.image = cv2.imread(path)
            self.show_image(self.label_input, self.image)
            self.show_image(self.label_result, self.image)

    def analyze_image(self):
        if self.image is None:
            return

        img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        results = self.model(img_rgb, verbose=False)

        all_predictions = []

        for r in results:
            for box in r.boxes:
                # Confidence score kontrolü ekle
                confidence = float(box.conf)
                if confidence < 0.4:  # Çok düşük confidence'ı atla
                    continue

                cls_id = int(box.cls)
                class_name = self.model.names[cls_id]

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = img_rgb[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

                # Açık alan limitini yumuşat (daha fazla nesne algıla)
                if (mask > 0).mean() * 100 < 5:
                    continue

                all_predictions.append((class_name, confidence))

        self.list_widget.clear()

        if len(all_predictions) == 0:
            self.list_widget.addItem("Nesne bulunamadı.")
            return

        # En yüksek confidence'a sahip nesneyi seç
        final_class, max_conf = max(all_predictions, key=lambda x: x[1])
        self.list_widget.addItem(f"Doğru Tahmin: {final_class} (Güven: {max_conf:.2f})")

    def region_selected(self, event):
        ClickableLabel.mouseReleaseEvent(self.label_result, event)

        if not self.label_result.rect_ready or self.image is None:
            return

        # Kareyi al
        p1 = self.label_result.start_point
        p2 = self.label_result.end_point

        label_w = self.label_result.width()
        label_h = self.label_result.height()

        img_h, img_w, _ = self.image.shape

        # UI → gerçek görüntü oran dönüşümü
        scale_x = img_w / label_w
        scale_y = img_h / label_h

        x1 = int(min(p1.x(), p2.x()) * scale_x)
        y1 = int(min(p1.y(), p2.y()) * scale_y)
        x2 = int(max(p1.x(), p2.x()) * scale_x)
        y2 = int(max(p1.y(), p2.y()) * scale_y)

        # Sınırları kontrol et
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)

        crop = self.image[y1:y2, x1:x2]

        if crop.size == 0:
            self.list_widget.addItem("Seçili alan boş!")
            return

        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = self.model(img_rgb, verbose=False)

        preds = []
        for r in results:
            for box in r.boxes:
                confidence = float(box.conf)
                # Seçili bölge için daha yüksek confidence gerekli
                if confidence >= 0.5:
                    class_id = int(box.cls)
                    class_name = self.model.names[class_id]
                    preds.append((class_name, confidence, class_id))

        if len(preds) == 0:
            self.list_widget.addItem("Seçili Bölge: Nesne yok")
        else:
            # En yüksek confidence'a sahip nesneyi seç
            final_class, max_conf, _ = max(preds, key=lambda x: x[1])
            self.list_widget.addItem(f"Seçili Bölge: {final_class} ({max_conf:.0%})")

    def save_image(self):
        if self.image is None:
            return

        path, _ = QFileDialog.getSaveFileName(self, "Kaydet", "", "PNG Image (*.png);;JPEG Image (*.jpg)")
        if path:
            cv2.imwrite(path, self.image)

    def show_image(self, widget, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(widget.width(), widget.height(), Qt.KeepAspectRatio)
        widget.setImage(pixmap) if isinstance(widget, ClickableLabel) else widget.setPixmap(pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLOGui()
    window.show()
    sys.exit(app.exec_())
