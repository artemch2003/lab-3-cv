#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Контроллер для OpenCV интерфейса.
Управляет окнами OpenCV и обработкой событий.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from models.image_processor import ImageProcessor
from models.image_analyzer import ImageAnalyzer
from config.settings import *


class OpenCVController:
    """
    Контроллер для управления OpenCV интерфейсом.
    """
    
    def __init__(self):
        """Инициализация контроллера."""
        self.processor = ImageProcessor()
        self.analyzer = ImageAnalyzer(self.processor)
        self.last_mouse = (0, 0)
        self.save_counter = 1
        self.running = True
    
    def load_image(self, path: str) -> bool:
        """
        Загружает изображение.
        
        Args:
            path: Путь к изображению
            
        Returns:
            True если изображение загружено успешно
        """
        success = self.processor.load_image(path)
        if success:
            # Устанавливаем начальную позицию мыши в центр изображения
            height, width = self.processor.original_image.shape[:2]
            self.last_mouse = (width // 2, height // 2)
        return success
    
    def load_image_from_processor(self, processor):
        """
        Загружает изображение из существующего процессора.
        
        Args:
            processor: Экземпляр ImageProcessor
        """
        self.processor = processor
        self.analyzer = ImageAnalyzer(self.processor)
        if self.processor.original_image is not None:
            height, width = self.processor.original_image.shape[:2]
            self.last_mouse = (width // 2, height // 2)
    
    def create_windows(self) -> None:
        """Создает все необходимые окна OpenCV."""
        for window_name in WINDOW_NAMES.values():
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    def create_trackbars(self) -> None:
        """Создает все трекбары в окне Controls."""
        controls_window = WINDOW_NAMES["CONTROLS"]
        
        trackbar_configs = [
            ("Bright", self.processor.get_parameter("brightness"), TRACKBAR_RANGES["Bright"][1]),
            ("Contrast", self.processor.get_parameter("contrast"), TRACKBAR_RANGES["Contrast"][1]),
            ("R_off", self.processor.get_parameter("r_offset"), TRACKBAR_RANGES["R_off"][1]),
            ("G_off", self.processor.get_parameter("g_offset"), TRACKBAR_RANGES["G_off"][1]),
            ("B_off", self.processor.get_parameter("b_offset"), TRACKBAR_RANGES["B_off"][1]),
            ("Gamma_x10", self.processor.get_parameter("gamma_x10"), TRACKBAR_RANGES["Gamma_x10"][1]),
            ("SwapMode", self.processor.get_parameter("swap_mode"), TRACKBAR_RANGES["SwapMode"][1]),
            ("NegR", int(self.processor.get_parameter("negate_r")), TRACKBAR_RANGES["NegR"][1]),
            ("NegG", int(self.processor.get_parameter("negate_g")), TRACKBAR_RANGES["NegG"][1]),
            ("NegB", int(self.processor.get_parameter("negate_b")), TRACKBAR_RANGES["NegB"][1]),
            # High-pass
            ("HP_Enable", int(self.processor.get_parameter("hp_enable")), TRACKBAR_RANGES["HP_Enable"][1]),
            ("HP_Mode", int(self.processor.get_parameter("hp_blur_mode")), TRACKBAR_RANGES["HP_Mode"][1]),
            ("HP_Kernel", int(self.processor.get_parameter("hp_kernel")), TRACKBAR_RANGES["HP_Kernel"][1]),
            ("HP_Scale_x100", int(self.processor.get_parameter("hp_scale_x100")), TRACKBAR_RANGES["HP_Scale_x100"][1]),
        ]
        
        for name, initial_value, max_value in trackbar_configs:
            cv2.createTrackbar(name, controls_window, initial_value, max_value, self._on_trackbar)
    
    def setup_mouse_callback(self) -> None:
        """Настраивает обработчик событий мыши."""
        cv2.setMouseCallback(WINDOW_NAMES["IMAGE"], self._on_mouse, self)
    
    def update_from_trackbars(self) -> None:
        """Обновляет параметры из трекбаров."""
        controls_window = WINDOW_NAMES["CONTROLS"]
        
        self.processor.set_parameter("brightness", cv2.getTrackbarPos("Bright", controls_window))
        self.processor.set_parameter("contrast", cv2.getTrackbarPos("Contrast", controls_window))
        self.processor.set_parameter("r_offset", cv2.getTrackbarPos("R_off", controls_window))
        self.processor.set_parameter("g_offset", cv2.getTrackbarPos("G_off", controls_window))
        self.processor.set_parameter("b_offset", cv2.getTrackbarPos("B_off", controls_window))
        self.processor.set_parameter("gamma_x10", cv2.getTrackbarPos("Gamma_x10", controls_window))
        self.processor.set_parameter("swap_mode", cv2.getTrackbarPos("SwapMode", controls_window))
        self.processor.set_parameter("negate_r", cv2.getTrackbarPos("NegR", controls_window) > 0)
        self.processor.set_parameter("negate_g", cv2.getTrackbarPos("NegG", controls_window) > 0)
        self.processor.set_parameter("negate_b", cv2.getTrackbarPos("NegB", controls_window) > 0)
        # High-pass
        self.processor.set_parameter("hp_enable", cv2.getTrackbarPos("HP_Enable", controls_window) > 0)
        self.processor.set_parameter("hp_blur_mode", cv2.getTrackbarPos("HP_Mode", controls_window))
        # Приводим ядро к нечетному значению
        hp_kernel_raw = cv2.getTrackbarPos("HP_Kernel", controls_window)
        if hp_kernel_raw % 2 == 0:
            hp_kernel_raw = max(3, hp_kernel_raw - 1)
        self.processor.set_parameter("hp_kernel", max(3, min(25, hp_kernel_raw)))
        self.processor.set_parameter("hp_scale_x100", cv2.getTrackbarPos("HP_Scale_x100", controls_window))
    
    def handle_keyboard_input(self, key: int) -> None:
        """
        Обрабатывает ввод с клавиатуры.
        
        Args:
            key: Код нажатой клавиши
        """
        if key in (27, ord('q'), ord('Q')):
            self.running = False
        elif key in (ord('h'), ord('H')):
            current = self.processor.get_parameter("flip_horizontal")
            self.processor.set_parameter("flip_horizontal", not current)
        elif key in (ord('v'), ord('V')):
            current = self.processor.get_parameter("flip_vertical")
            self.processor.set_parameter("flip_vertical", not current)
        elif key in (ord('s'), ord('S')):
            self._save_current_image()
    
    def update_display(self) -> None:
        """Обновляет отображение всех окон."""
        if self.processor.original_image is None:
            return
        
        # Обновляем параметры и пересчитываем изображение
        self.update_from_trackbars()
        processed_image = self.processor.process_image()
        
        # Получаем текущие координаты мыши
        mouse_x, mouse_y = self.last_mouse
        
        # Создаем изображение с рамкой и статусом
        main_image = self.analyzer.draw_frame_and_status(
            processed_image, mouse_x, mouse_y, processed_image
        )
        
        # Создаем увеличенное окно
        zoom_image = self.analyzer.create_zoom_window(processed_image, mouse_x, mouse_y)
        
        # Создаем мозаику каналов и гистограммы
        channels_mosaic = self.analyzer.create_mosaic_channels(processed_image)
        histograms_image = self.analyzer.make_histogram_image(processed_image)
        
        # Отображаем все окна
        cv2.imshow(WINDOW_NAMES["IMAGE"], main_image)
        cv2.imshow(WINDOW_NAMES["ZOOM"], zoom_image)
        cv2.imshow(WINDOW_NAMES["CHANNELS"], channels_mosaic)
        cv2.imshow(WINDOW_NAMES["HISTOGRAMS"], histograms_image)
    
    def run_main_loop(self) -> None:
        """Запускает главный цикл приложения."""
        while self.running:
            self.update_display()
            
            # Обрабатываем ввод с клавиатуры
            key = cv2.waitKey(GUI_SETTINGS["update_interval"]) & 0xFF
            self.handle_keyboard_input(key)
        
        # Закрываем все окна
        cv2.destroyAllWindows()
    
    def _on_mouse(self, event: int, x: int, y: int, flags: int, userdata) -> None:
        """
        Обработчик событий мыши для основного окна изображения.
        
        Args:
            event: Тип события мыши
            x: Координата X мыши
            y: Координата Y мыши
            flags: Флаги события
            userdata: Данные пользователя (self)
        """
        if event == cv2.EVENT_MOUSEMOVE and self.processor.processed_image is not None:
            # Ограничиваем координаты границами изображения
            x_clipped = np.clip(x, 0, self.processor.processed_image.shape[1] - 1)
            y_clipped = np.clip(y, 0, self.processor.processed_image.shape[0] - 1)
            self.last_mouse = (x_clipped, y_clipped)
    
    def _on_trackbar(self, _: int = None) -> None:
        """
        Обработчик изменений трекбаров.
        
        Примечание: Фактически значения читаются пакетом в основном цикле,
        поэтому этот колбэк пустой.
        """
        pass
    
    def _save_current_image(self) -> None:
        """Сохраняет текущее обработанное изображение."""
        if self.processor.processed_image is not None:
            from utils.file_utils import get_save_filename
            output_filename = get_save_filename(self.save_counter)
            cv2.imwrite(output_filename, self.processor.processed_image)
            print(f"[OK] Сохранено: {output_filename}")
            self.save_counter += 1
