#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модель для обработки изображений.
Содержит всю бизнес-логику обработки изображений.
"""

import math
from typing import Tuple, Dict, Any
import cv2
import numpy as np
from config.settings import *


class ImageProcessor:
    """`
    Класс для обработки изображений с различными эффектами.
    """
    
    def __init__(self):
        """Инициализация процессора изображений."""
        self.original_image = None
        self.processed_image = None
        self.params_dirty = True
        
        # Параметры обработки
        self.params = {
            "brightness": 100,      # 0..200 (beta = val-100)
            "contrast": 100,        # 50..300 mapped; alpha = val/100
            "r_offset": 100,        # 0..200 (off = val-100)
            "g_offset": 100,
            "b_offset": 100,
            "gamma_x10": 10,        # 5..40  => 0.5..4.0
            "swap_mode": "BGR",     # BGR, BRG, GBR, GRB, RBG, RGB
            "negate_r": False,      # 0/1
            "negate_g": False,
            "negate_b": False,
            "flip_horizontal": False,
            "flip_vertical": False,
            # High-pass параметры
            "hp_enable": False,     # включение высокочастотного режима
            "hp_blur_mode": 0,      # 0=mean, 1=gaussian
            "hp_kernel": 3,         # нечетный размер ядра (3..25)
            "hp_scale_x100": 100,   # c в формуле, х100 (0..300)
            # Свёртка
            "conv_enable": False,           # включение свёртки
            "conv_normalize": True,         # делить на сумму ядра
            "conv_add128": False,           # прибавить 128 к результату
            "conv_kernel_size": 3,          # n (нечётное)
            "conv_kernel_text": "",        # текст ядра n×n через пробел/запятую/перевод строки
            "conv_preset": "Пользовательская",  # название пресета
            # Границы (Edges)
            "edges_enable": False,
            "edges_method": 0,          # 0=Sobel, 1=Prewitt
            "edges_thresh": 100,        # 0..255 порог по модулю градиента
            "edges_overlay": True,      # накладывать на изображение
            # Углы (Harris)
            "corners_enable": False,
            "corners_k_x1000": 4,       # k*1000 (по умолчанию 0.004)
            "corners_block": 5,         # размер окна усреднения (нечётный)
            "corners_thresh_x100": 10,  # 0..100 (% от max R)
            "corners_nms": 3,           # окно NMS (нечётный)
            "corners_overlay": True,    # рисовать поверх изображения
        }
    
    def load_image(self, path: str) -> bool:
        """
        Загружает изображение из файла.
        
        Args:
            path: Путь к файлу изображения
            
        Returns:
            True если изображение успешно загружено, False иначе
        """
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                return False
            
            if img.ndim == 2:
                # Серое изображение -> в 3 канала
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                # Отбрасываем альфа-канал
                img = img[:, :, :3]
            
            self.original_image = img
            self.processed_image = img.copy()
            self.params_dirty = True
            return True
            
        except Exception:
            return False
    
    def set_parameter(self, param_name: str, value: Any) -> None:
        """
        Устанавливает параметр обработки.
        
        Args:
            param_name: Имя параметра
            value: Значение параметра
        """
        if param_name in self.params:
            self.params[param_name] = value
            self.params_dirty = True
    
    def get_parameter(self, param_name: str) -> Any:
        """
        Получает значение параметра.
        
        Args:
            param_name: Имя параметра
            
        Returns:
            Значение параметра
        """
        return self.params.get(param_name, None)
    
    def process_image(self) -> np.ndarray:
        """
        Обрабатывает изображение согласно текущим параметрам.
        
        Returns:
            Обработанное изображение
        """
        if self.original_image is None or not self.params_dirty:
            return self.processed_image
        
        # Начинаем с исходного изображения
        processed = self.original_image.copy()
        
        # Извлекаем параметры
        brightness_offset = self.params["brightness"] - 100
        contrast_factor = self.params["contrast"] / 100.0
        red_offset = self.params["r_offset"] - 100
        green_offset = self.params["g_offset"] - 100
        blue_offset = self.params["b_offset"] - 100
        gamma_value = self.params["gamma_x10"] / 10.0
        swap_mode = self.params["swap_mode"]
        negate_red = self.params["negate_r"]
        negate_green = self.params["negate_g"]
        negate_blue = self.params["negate_b"]
        
        # Применяем эффекты в порядке:
        # 1) Контраст, яркость и сдвиги каналов
        processed = self._apply_brightness_contrast_per_channel(
            processed, brightness_offset, red_offset, green_offset, 
            blue_offset, contrast_factor
        )
        
        # 2) Гамма-коррекция
        processed = self._apply_gamma(processed, gamma_value)
        
        # 3) Инверсия выбранных каналов
        processed = self._invert_selected_channels(
            processed, negate_red, negate_green, negate_blue
        )
        
        # 4) Перестановка каналов
        processed = self._swap_channels(processed, swap_mode)
        
        # 5) Отражения
        processed = self._flip_image(
            processed, self.params["flip_horizontal"], self.params["flip_vertical"]
        )

        # 6) High-pass фильтр (опционально)
        if self.params.get("hp_enable", False):
            processed = self._apply_high_pass(
                processed,
                blur_mode=int(self.params.get("hp_blur_mode", 0)),
                kernel_size=int(self.params.get("hp_kernel", 3)),
                scale=float(self.params.get("hp_scale_x100", 100)) / 100.0,
            )

        # 7) Свёртка (опционально)
        if self.params.get("conv_enable", False):
            kernel = self._get_convolution_kernel(
                preset=str(self.params.get("conv_preset", "Пользовательская")),
                kernel_size=int(self.params.get("conv_kernel_size", 3)),
                kernel_text=str(self.params.get("conv_kernel_text", "")),
            )
            if kernel is not None:
                processed = self._apply_convolution(
                    processed,
                    kernel,
                    normalize=bool(self.params.get("conv_normalize", True)),
                    add128=bool(self.params.get("conv_add128", False)),
                )
        # 8) Выделение границ (опционально)
        if bool(self.params.get("edges_enable", False)):
            gray = self.to_gray_manual(processed)
            edges_mask = self._edges_detect(gray,
                                            method=int(self.params.get("edges_method", 0)),
                                            thresh=int(self.params.get("edges_thresh", 100)))
            if bool(self.params.get("edges_overlay", True)):
                # Накладываем границы цветом (циан)
                processed = self._overlay_mask_color(processed, edges_mask, color=(255, 255, 0))
            else:
                # Выводим карту границ как 3-канальное
                processed = cv2.cvtColor(edges_mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)

        # 9) Выделение углов (опционально)
        if bool(self.params.get("corners_enable", False)):
            gray = self.to_gray_manual(processed)
            corners_mask = self._harris_corners(gray,
                                                k=float(self.params.get("corners_k_x1000", 4)) / 1000.0,
                                                block_size=int(self.params.get("corners_block", 5)),
                                                thresh_ratio=float(self.params.get("corners_thresh_x100", 10)) / 100.0,
                                                nms_size=int(self.params.get("corners_nms", 3)))
            if bool(self.params.get("corners_overlay", True)):
                # Рисуем углы красным цветом
                processed = self._overlay_points(processed, corners_mask, color=(0, 0, 255))
            else:
                processed = cv2.cvtColor((corners_mask > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)

        self.processed_image = processed
        self.params_dirty = False
        return processed
    
    def to_gray_manual(self, bgr: np.ndarray) -> np.ndarray:
        """
        Конвертирует BGR изображение в серое, используя формулу (R+G+B)/3.
        
        Args:
            bgr: Изображение в формате BGR
            
        Returns:
            Серое изображение в формате uint8
        """
        red_channel = bgr[..., 2].astype(np.float32)
        green_channel = bgr[..., 1].astype(np.float32)
        blue_channel = bgr[..., 0].astype(np.float32)
        
        gray = (red_channel + green_channel + blue_channel) / 3.0
        gray = np.clip(gray, MIN_PIXEL_VALUE, MAX_PIXEL_VALUE).astype(np.uint8)
        
        return gray
    
    def intensity_at_pixel(self, bgr: np.ndarray, x: int, y: int) -> float:
        """
        Вычисляет интенсивность пикселя как среднее значение RGB каналов.
        
        Args:
            bgr: Изображение в формате BGR
            x: Координата X пикселя
            y: Координата Y пикселя
            
        Returns:
            Интенсивность пикселя (0.0 - 255.0)
        """
        blue, green, red = bgr[y, x]
        return (float(red) + float(green) + float(blue)) / 3.0
    
    def window_mean_std_intensity(
        self, bgr: np.ndarray, x: int, y: int, half: int = INNER_WINDOW_HALF
    ) -> Tuple[float, float]:
        """
        Вычисляет среднее и стандартное отклонение интенсивности в окне.
        
        Args:
            bgr: Изображение в формате BGR
            x: Центральная координата X окна
            y: Центральная координата Y окна
            half: Половина размера окна
            
        Returns:
            Кортеж (среднее, стандартное_отклонение)
        """
        height, width, _ = bgr.shape
        
        # Определяем границы окна с учетом границ изображения
        x_start = max(0, x - half)
        y_start = max(0, y - half)
        x_end = min(width, x + half + 1)
        y_end = min(height, y + half + 1)
        
        # Извлекаем окно
        window_patch = bgr[y_start:y_end, x_start:x_end, :]
        
        # Вычисляем интенсивность для каждого пикселя
        red_channel = window_patch[..., 2].astype(np.float64)
        green_channel = window_patch[..., 1].astype(np.float64)
        blue_channel = window_patch[..., 0].astype(np.float64)
        intensity = (red_channel + green_channel + blue_channel) / 3.0
        
        # Вычисляем статистики
        pixel_count = intensity.size
        sum_intensity = float(intensity.sum())
        sum_squared = float((intensity * intensity).sum())
        
        mean_intensity = sum_intensity / max(pixel_count, 1)
        variance = sum_squared / max(pixel_count, 1) - (mean_intensity * mean_intensity)
        std_deviation = math.sqrt(max(variance, 0.0))
        
        return mean_intensity, std_deviation
    
    def _apply_brightness_contrast_per_channel(
        self, bgr: np.ndarray, global_beta: int, r_off: int, 
        g_off: int, b_off: int, contrast_alpha: float
    ) -> np.ndarray:
        """Применяет контраст, яркость и сдвиги по каналам."""
        result = bgr.astype(np.float32)
        
        # Применяем контраст и глобальную яркость
        result = (result - 128.0) * contrast_alpha + 128.0 + float(global_beta)
        
        # Применяем смещения по каналам
        result[..., 2] += float(r_off)  # Красный канал
        result[..., 1] += float(g_off)  # Зеленый канал
        result[..., 0] += float(b_off)  # Синий канал
        
        return np.clip(result, MIN_PIXEL_VALUE, MAX_PIXEL_VALUE).astype(np.uint8)
    
    def _apply_gamma(self, bgr: np.ndarray, gamma: float) -> np.ndarray:
        """Применяет гамма-коррекцию к изображению."""
        if abs(gamma - 1.0) < 1e-6:
            return bgr
        
        # Создаем lookup table для гамма-коррекции
        inverse_gamma = 1.0 / max(gamma, 1e-6)
        pixel_values = np.arange(HISTOGRAM_BINS, dtype=np.float32) / MAX_PIXEL_VALUE
        lookup_table = np.clip(
            (pixel_values ** inverse_gamma) * MAX_PIXEL_VALUE, 
            MIN_PIXEL_VALUE, 
            MAX_PIXEL_VALUE
        ).astype(np.uint8)
        
        return cv2.LUT(bgr, lookup_table)
    
    def _invert_selected_channels(
        self, bgr: np.ndarray, neg_r: bool, neg_g: bool, neg_b: bool
    ) -> np.ndarray:
        """Инвертирует выбранные цветовые каналы."""
        result = bgr.copy()
        
        if neg_b:
            result[..., 0] = MAX_PIXEL_VALUE - result[..., 0]  # Синий канал
        if neg_g:
            result[..., 1] = MAX_PIXEL_VALUE - result[..., 1]  # Зеленый канал
        if neg_r:
            result[..., 2] = MAX_PIXEL_VALUE - result[..., 2]  # Красный канал
        
        return result
    
    def _swap_channels(self, bgr: np.ndarray, mode) -> np.ndarray:
        """Переставляет цветовые каналы согласно заданному режиму."""
        blue_channel, green_channel, red_channel = cv2.split(bgr)
        
        # Поддержка как строковых, так и числовых значений
        if isinstance(mode, str):
            channel_combinations = {
                "BGR": [blue_channel, green_channel, red_channel],
                "BRG": [blue_channel, red_channel, green_channel],
                "GBR": [green_channel, blue_channel, red_channel],
                "GRB": [green_channel, red_channel, blue_channel],
                "RBG": [red_channel, blue_channel, green_channel],
                "RGB": [red_channel, green_channel, blue_channel],
            }
        else:
            # Обратная совместимость с числовыми значениями
            channel_combinations = {
                0: [blue_channel, green_channel, red_channel],   # BGR
                1: [blue_channel, red_channel, green_channel],   # BRG
                2: [green_channel, blue_channel, red_channel],   # GBR
                3: [green_channel, red_channel, blue_channel],   # GRB
                4: [red_channel, blue_channel, green_channel],   # RBG
                5: [red_channel, green_channel, blue_channel],   # RGB
            }
        
        if mode in channel_combinations:
            return cv2.merge(channel_combinations[mode])
        
        return bgr
    
    def _flip_image(self, bgr: np.ndarray, flip_h: bool, flip_v: bool) -> np.ndarray:
        """Применяет отражения изображения по горизонтали и/или вертикали."""
        result = bgr
        
        if flip_h and flip_v:
            # Отражение по обеим осям
            result = cv2.flip(result, -1)
        elif flip_h:
            # Отражение по горизонтали
            result = cv2.flip(result, 1)
        elif flip_v:
            # Отражение по вертикали
            result = cv2.flip(result, 0)
        
        return result

    def _apply_high_pass(self, bgr: np.ndarray, blur_mode: int, kernel_size: int, scale: float) -> np.ndarray:
        """Применяет высокочастотный фильтр: HP = SRC - BLUR * c.

        Размытие реализовано вручную: усреднение или гауссово сверткой без сторонних библиотек.
        """
        # Приводим к допустимым значениям
        kernel_size = max(1, int(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = min(kernel_size, 25)
        scale = float(scale)

        src = bgr.astype(np.float32)
        if blur_mode == 1:
            blurred = self._gaussian_blur_manual(src, kernel_size)
        else:
            blurred = self._mean_blur_manual(src, kernel_size)

        high_pass = src - blurred * scale
        high_pass = np.clip(high_pass, MIN_PIXEL_VALUE, MAX_PIXEL_VALUE)
        return high_pass.astype(np.uint8)

    def _mean_blur_manual(self, img_f32: np.ndarray, k: int) -> np.ndarray:
        """Ручное усредняющее размытие сверткой с ядром из единиц / (k*k)."""
        kernel_area = float(k * k)
        # Разбиваем на каналы для ускорения
        channels = cv2.split(img_f32)
        blurred_channels = [self._box_blur_channel(ch, k, kernel_area) for ch in channels]
        return cv2.merge(blurred_channels)

    def _box_blur_channel(self, ch: np.ndarray, k: int, kernel_area: float) -> np.ndarray:
        pad = k // 2
        # Паддинг отражением, чтобы не терять края
        padded = np.pad(ch, ((pad, pad), (pad, pad)), mode='reflect')
        h, w = ch.shape
        out = np.empty_like(ch)

        # Интегральное изображение для O(1) усреднения окна
        integral = padded.cumsum(axis=0).cumsum(axis=1)
        # вспомогательная функция суммы по окну через интегральное изображение
        def rect_sum(ii: np.ndarray, y0: int, x0: int, y1: int, x1: int) -> float:
            total = ii[y1, x1]
            if y0 > 0:
                total -= ii[y0 - 1, x1]
            if x0 > 0:
                total -= ii[y1, x0 - 1]
            if y0 > 0 and x0 > 0:
                total += ii[y0 - 1, x0 - 1]
            return float(total)

        for y in range(h):
            y0 = y
            y1 = y + k - 1
            for x in range(w):
                x0 = x
                x1 = x + k - 1
                s = rect_sum(integral, y0, x0, y1, x1)
                out[y, x] = s / kernel_area
        return out

    def _gaussian_blur_manual(self, img_f32: np.ndarray, k: int) -> np.ndarray:
        """Ручное гауссово размытие: separable 1D ядро, sigma ~ k/6."""
        sigma = max(0.1, k / 6.0)
        kernel_1d = self._gaussian_kernel_1d(k, sigma)
        # Разделим по каналам
        channels = cv2.split(img_f32)
        blurred_channels = []
        for ch in channels:
            tmp = self._convolve_1d(ch, kernel_1d, axis=1)
            res = self._convolve_1d(tmp, kernel_1d, axis=0)
            blurred_channels.append(res)
        return cv2.merge(blurred_channels)

    def _gaussian_kernel_1d(self, k: int, sigma: float) -> np.ndarray:
        radius = k // 2
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
        kernel /= np.sum(kernel)
        return kernel.astype(np.float32)

    def _convolve_1d(self, ch: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
        pad = len(kernel) // 2
        if axis == 1:  # по x (строчно)
            padded = np.pad(ch, ((0, 0), (pad, pad)), mode='reflect')
            h, w = ch.shape
            out = np.empty_like(ch)
            for y in range(h):
                row = padded[y]
                # свертка строки
                for x in range(w):
                    window = row[x:x + 2 * pad + 1]
                    out[y, x] = float(np.dot(window, kernel))
            return out
        else:  # по y (столбцово)
            padded = np.pad(ch, ((pad, pad), (0, 0)), mode='reflect')
            h, w = ch.shape
            out = np.empty_like(ch)
            for x in range(w):
                col = padded[:, x]
                for y in range(h):
                    window = col[y:y + 2 * pad + 1]
                    out[y, x] = float(np.dot(window, kernel))
            return out

    # ---- Свёртка NxN ----
    def _get_convolution_kernel(self, preset: str, kernel_size: int, kernel_text: str) -> np.ndarray | None:
        """Возвращает ядро свёртки из пресета или парсит пользовательское.

        Правила:
        - Если выбран пресет, игнорируем текст (кроме "Пользовательская").
        - Размер ядра приводим к нечётному в диапазоне [1, 25].
        - Парсер принимает числа через пробелы, запятые и переводы строк.
        """
        # Пресеты
        presets = self._preset_kernels()
        if preset in presets and preset != "Пользовательская":
            return presets[preset].astype(np.float32)

        # Пользовательское ядро
        k = max(1, int(kernel_size))
        if k % 2 == 0:
            k += 1
        k = min(k, 25)

        values = self._parse_kernel_text(kernel_text)
        if values is None or len(values) == 0:
            return None

        # Если размер не задан явно, пытаемся вывести из количества значений
        n_vals = len(values)
        n = int(round(math.sqrt(n_vals)))
        if n * n == n_vals:
            k = n

        if k * k != n_vals:
            return None

        kernel = np.array(values, dtype=np.float32).reshape((k, k))
        return kernel

    def _parse_kernel_text(self, text: str):
        """Парсит текст ядра в список чисел (float). Допускает разделители: пробел, запятая, перевод строки, точка с запятой."""
        if text is None:
            return None
        raw = text.replace("\n", " ").replace(",", " ").replace(";", " ")
        tokens = [t for t in raw.split() if len(t) > 0]
        values = []
        for tok in tokens:
            try:
                values.append(float(tok))
            except Exception:
                return None
        return values

    def _preset_kernels(self) -> Dict[str, np.ndarray]:
        """Набор стандартных ядер свёртки."""
        return {
            "Пользовательская": np.array([[0]], dtype=np.float32),
            "Identity 3x3": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32),
            "Box 3x3": np.ones((3, 3), dtype=np.float32),
            "Gaussian 3x3": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32),
            "Sobel X": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
            "Sobel Y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
            "Prewitt X": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32),
            "Prewitt Y": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32),
            "Laplacian 4": np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32),
            "Laplacian 8": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32),
        }

    def _apply_convolution(self, bgr: np.ndarray, kernel: np.ndarray, normalize: bool, add128: bool) -> np.ndarray:
        """Применяет свёртку ядром к каждому каналу BGR вручную.

        Вычисления ведутся в float32. Паддинг: reflect. Нормализация: деление на сумму ядра (если сумма != 0).
        Опция +128: после свёртки добавляется 128 к каждому значению.
        """
        k = np.array(kernel, dtype=np.float32)
        # Нормализация ядра (деление на сумму) если требуется и сумма != 0
        if normalize:
            s = float(k.sum())
            if abs(s) > 1e-12:
                k = k / s

        channels = cv2.split(bgr.astype(np.float32))
        out_channels = [self._convolve_channel(ch, k, add128) for ch in channels]
        out = cv2.merge(out_channels)
        out = np.clip(out, MIN_PIXEL_VALUE, MAX_PIXEL_VALUE).astype(np.uint8)
        return out

    def _convolve_channel(self, ch: np.ndarray, kernel: np.ndarray, add128: bool) -> np.ndarray:
        """Свёртка одного канала ch ядром kernel через окна и tensordot."""
        kh, kw = kernel.shape
        pad_y = kh // 2
        pad_x = kw // 2
        padded = np.pad(ch, ((pad_y, pad_y), (pad_x, pad_x)), mode='reflect')
        h, w = ch.shape

        # Используем представление всех окон через stride_tricks (без доп. памяти)
        shape = (h, w, kh, kw)
        strides = (padded.strides[0], padded.strides[1], padded.strides[0], padded.strides[1])
        windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides, writeable=False)

        # Свёртка = корреляция с перевёрнутым ядром
        kernel_flipped = kernel[::-1, ::-1]
        result = np.tensordot(windows, kernel_flipped, axes=([2, 3], [0, 1]))
        if add128:
            result = result + 128.0
        return result.astype(np.float32)

    # ---- Выделение границ ----
    def _edges_detect(self, gray_u8: np.ndarray, method: int, thresh: int) -> np.ndarray:
        """Возвращает бинарную маску границ для серого изображения.

        method: 0=Sobel, 1=Prewitt
        thresh: порог на |grad| (0..255)
        """
        gray = gray_u8.astype(np.float32)
        if method == 1:
            # Prewitt
            kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
            ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        else:
            # Sobel
            kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
            ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

        gx = self._convolve_channel(gray, kx, add128=False)
        gy = self._convolve_channel(gray, ky, add128=False)
        mag = np.sqrt(gx * gx + gy * gy)
        mag = np.clip(mag, 0.0, 255.0)
        mask = (mag >= float(thresh))
        return mask.astype(np.uint8)

    def _overlay_mask_color(self, bgr: np.ndarray, mask_u8: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
        """Накладывает бинарную маску на изображение указанным цветом."""
        result = bgr.copy()
        yx = np.where(mask_u8 > 0)
        if len(yx[0]) > 0:
            # color: (B, G, R)
            result[yx[0], yx[1], :] = color
        return result

    # ---- Выделение углов (Harris) ----
    def _harris_corners(self, gray_u8: np.ndarray, k: float, block_size: int, thresh_ratio: float, nms_size: int) -> np.ndarray:
        """Возвращает бинарную маску углов по методу Харриса (без OpenCV cornerHarris)."""
        gray = gray_u8.astype(np.float32)
        # Градиенты через Sobel
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        ix = self._convolve_channel(gray, kx, add128=False)
        iy = self._convolve_channel(gray, ky, add128=False)

        ix2 = ix * ix
        iy2 = iy * iy
        ixy = ix * iy

        # Усреднение в окне block_size (box blur)
        bs = max(1, int(block_size))
        if bs % 2 == 0:
            bs += 1
        area = float(bs * bs)
        sxx = self._box_blur_channel(ix2, bs, area)
        syy = self._box_blur_channel(iy2, bs, area)
        sxy = self._box_blur_channel(ixy, bs, area)

        # Отклик Харриса
        det = sxx * syy - sxy * sxy
        trace = sxx + syy
        r = det - k * (trace * trace)

        r_max = float(np.max(r)) if r.size > 0 else 0.0
        if r_max <= 0.0:
            return np.zeros_like(gray_u8, dtype=np.uint8)
        thr = r_max * max(0.0, float(thresh_ratio))
        cand = (r >= thr)

        # NMS (не максимум подавляем) через сравнение с локальным максимумом
        nms = max(1, int(nms_size))
        if nms % 2 == 0:
            nms += 1
        pad = nms // 2
        padded = np.pad(r, ((pad, pad), (pad, pad)), mode='constant', constant_values=-np.inf)
        h, w = r.shape
        # Окна через as_strided
        shape = (h, w, nms, nms)
        strides = (padded.strides[0], padded.strides[1], padded.strides[0], padded.strides[1])
        windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides, writeable=False)
        local_max = windows.max(axis=(2, 3))
        is_max = (r == local_max)
        corners = (cand & is_max)
        return corners.astype(np.uint8)

    def _overlay_points(self, bgr: np.ndarray, points_mask_u8: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
        """Рисует маленькие крестики в местах, где points_mask_u8 > 0."""
        result = bgr.copy()
        ys, xs = np.where(points_mask_u8 > 0)
        h, w = result.shape[:2]
        for y, x in zip(ys, xs):
            # горизонтальная линия
            x0 = max(0, x - 2)
            x1 = min(w - 1, x + 2)
            result[y, x0:x1 + 1] = color
            # вертикальная линия
            y0 = max(0, y - 2)
            y1 = min(h - 1, y + 2)
            result[y0:y1 + 1, x] = color
        return result
