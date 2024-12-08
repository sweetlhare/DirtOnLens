import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                           QWidget, QMessageBox, QProgressDialog)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import shutil
import cv2
import numpy as np

class ImageSorter(QMainWindow):
   def __init__(self):
       super().__init__()
       self.setWindowTitle('Image Sorter')
       self.setGeometry(100, 100, 800, 800)

       # Создаем центральный виджет и layout
       central_widget = QWidget()
       self.setCentralWidget(central_widget)
       layout = QVBoxLayout(central_widget)

       # Добавляем описание клавиш
       self.keys_label = QLabel()
       self.keys_label.setText(
           "Горячие клавиши:\n"
           "F - сохранить фото и маску в raindrop_image\n"
           "J - сохранить фото (без маски) в empty_image\n"
           "B - пропустить текущее фото"
       )
       self.keys_label.setStyleSheet("border: 1px solid gray; padding: 10px; background-color: #f0f0f0;")

       # Создаем остальные метки
       self.image_label = QLabel()
       self.mask_label = QLabel()
       self.counter_label = QLabel()
       self.debug_label = QLabel()
       self.filename_label = QLabel()
       self.dataset_type_label = QLabel()
       self.dataset_type_label.setStyleSheet("font-weight: bold; color: blue;")
       
       # Добавляем все виджеты в layout
       layout.addWidget(self.keys_label)
       layout.addWidget(self.dataset_type_label)
       layout.addWidget(self.image_label)
       layout.addWidget(self.mask_label)
       layout.addWidget(self.counter_label)
       layout.addWidget(self.filename_label)
       layout.addWidget(self.debug_label)

       # Инициализируем пути к папкам
       self.img_dir = "cv_open_dataset/open_img"
       self.mask_dir = "cv_open_dataset/open_msk"
       
       # Проверяем существование директорий
       if not os.path.exists(self.img_dir):
           self.debug_label.setText(f"Error: Directory not found: {self.img_dir}")
           self.image_files = []
           return
       if not os.path.exists(self.mask_dir):
           self.debug_label.setText(f"Error: Directory not found: {self.mask_dir}")
           self.image_files = []
           return

       # Создаем выходные директории
       os.makedirs("raindrop_image/img", exist_ok=True)
       os.makedirs("raindrop_image/msk", exist_ok=True)
       os.makedirs("empty_image/img", exist_ok=True)

       # Инициализируем списки для файлов
       self.empty_files = []
       self.mask_files = {}
       self.masked_files = []

       # Создаем и показываем прогресс-диалог
       progress = QProgressDialog("Анализ изображений...", None, 0, 100, self)
       progress.setWindowModality(Qt.WindowModal)
       progress.setMinimumDuration(0)
       progress.setValue(0)
       
       # Получаем список всех файлов
       all_files = sorted([f for f in os.listdir(self.img_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
       total_files = len(all_files)
       
       # Обрабатываем каждый файл
       for i, img_file in enumerate(all_files):
           progress.setValue(int((i / total_files) * 100))
           
           base_name = os.path.splitext(img_file)[0]
           mask_found = False
           
           # Проверяем возможные имена масок
           possible_mask_names = [f"{base_name}.png"]
           
           for mask_name in possible_mask_names:
               mask_path = os.path.join(self.mask_dir, mask_name)
               if os.path.exists(mask_path):
                   # Читаем маску и проверяем на пустоту
                   mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                   if mask is not None:
                       non_zero = cv2.countNonZero(mask)
                       total_pixels = mask.shape[0] * mask.shape[1]
                       if (non_zero / total_pixels) > 0.01:
                           self.mask_files[img_file] = mask_name
                           self.masked_files.append(img_file)
                           mask_found = True
                           break
           
           if not mask_found:
               self.empty_files.append(img_file)

       progress.setValue(100)

       # Начинаем с пустых изображений
       self.current_mode = 'empty'
       self.image_files = self.empty_files
       self.current_index = 0
       self.total_images = len(self.empty_files)
       self.dataset_type_label.setText("Текущий набор: Изображения без масок")
       
       # Показываем статистику
       self.debug_label.setText(f"Найдено изображений без масок: {len(self.empty_files)}\n"
                              f"Найдено изображений с масками: {len(self.masked_files)}")
       
       if self.image_files:
           self.update_display()
       else:
           self.switch_to_masked_images()

   def switch_to_masked_images(self):
       if self.masked_files:
           reply = QMessageBox.question(self, 'Переключение набора данных',
                                      'Вы закончили сортировку изображений без масок. Перейти к изображениям с масками?',
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
           
           if reply == QMessageBox.Yes:
               self.current_mode = 'masked'
               self.image_files = self.masked_files
               self.current_index = 0
               self.total_images = len(self.masked_files)
               self.dataset_type_label.setText("Текущий набор: Изображения с масками")
               self.update_display()
           else:
               self.close()
       else:
           QMessageBox.information(self, 'Завершение', 'Сортировка завершена.')
           self.close()

   def update_display(self):
       if not self.image_files or self.current_index >= self.total_images:
           if self.current_mode == 'empty':
               self.switch_to_masked_images()
           else:
               QMessageBox.information(self, 'Завершение', 'Сортировка завершена.')
               self.close()
           return

       current_image = self.image_files[self.current_index]
       img_path = os.path.join(self.img_dir, current_image)
       
       # Получаем соответствующую маску
       mask_name = self.mask_files.get(current_image)
       mask_path = os.path.join(self.mask_dir, mask_name) if mask_name else None

       # Показываем текущие имена файлов
       self.filename_label.setText(f"Image: {current_image}\nMask: {mask_name if mask_name else 'No mask found'}")

       # Загружаем и отображаем изображение
       if os.path.exists(img_path):
           img = cv2.imread(img_path)
           if img is not None:
               img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
               h, w, c = img.shape
               qimg = QImage(img.data, w, h, w * c, QImage.Format_RGB888)
               pixmap = QPixmap.fromImage(qimg)
               scaled_pixmap = pixmap.scaled(400, 300, Qt.KeepAspectRatio)
               self.image_label.setPixmap(scaled_pixmap)

       # Загружаем и отображаем маску
       if mask_path and os.path.exists(mask_path):
           mask = cv2.imread(mask_path)
           if mask is not None:
               mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
               h, w, c = mask.shape
               qimg = QImage(mask.data, w, h, w * c, QImage.Format_RGB888)
               pixmap = QPixmap.fromImage(qimg)
               scaled_pixmap = pixmap.scaled(400, 300, Qt.KeepAspectRatio)
               self.mask_label.setPixmap(scaled_pixmap)
       else:
           self.mask_label.clear()

       # Обновляем счетчик
       self.counter_label.setText(f'Image {self.current_index + 1} of {self.total_images}')

   def keyPressEvent(self, event):
       if not self.image_files or self.current_index >= self.total_images:
           return

       current_image = self.image_files[self.current_index]
       
       if event.key() == Qt.Key_F:
           # Сохраняем изображение и маску в raindrop_image
           shutil.copy(
               os.path.join(self.img_dir, current_image),
               os.path.join("raindrop_image/img", current_image)
           )
           
           mask_name = self.mask_files.get(current_image)
           if mask_name:
               shutil.copy(
                   os.path.join(self.mask_dir, mask_name),
                   os.path.join("raindrop_image/msk", mask_name)
               )
           self.current_index += 1
           
       elif event.key() == Qt.Key_J:
           # Сохраняем только изображение в empty_image
           shutil.copy(
               os.path.join(self.img_dir, current_image),
               os.path.join("empty_image/img", current_image)
           )
           self.current_index += 1
           
       elif event.key() == Qt.Key_B:
           # Просто пропускаем текущее изображение
           self.current_index += 1

       self.update_display()

if __name__ == '__main__':
   app = QApplication(sys.argv)
   window = ImageSorter()
   window.show()
   sys.exit(app.exec_())