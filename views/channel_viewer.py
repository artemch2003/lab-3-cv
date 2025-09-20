#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Окно для отображения каналов изображения в полный размер.
Поддерживает переключение между каналами по клику.
"""

import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
from typing import Optional, Dict, Any


class ChannelViewer:
    """
    Окно для отображения каналов изображения в полный размер.
    """
    
    def __init__(self, parent=None):
        """
        Инициализация окна просмотра каналов.
        
        Args:
            parent: Родительское окно (опционально)
        """
        self.parent = parent
        self.root = None
        self.current_channel = 0  # 0: Gray, 1: Red, 2: Green, 3: Blue
        self.channel_images = {}  # Словарь для хранения каналов
        self.display_image = None
        
        # Создаем окно
        self._create_window()
        self._setup_bindings()
    
    def _create_window(self):
        """Создает окно просмотра каналов."""
        self.root = tk.Toplevel(self.parent) if self.parent else tk.Tk()
        self.root.title("Просмотр каналов изображения")
        self.root.geometry("800x600")
        self.root.minsize(400, 300)
        
        # Главный фрейм
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Заголовок с информацией
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.channel_label = ttk.Label(
            info_frame, 
            text="Серый канал (Gray)", 
            font=("Arial", 12, "bold")
        )
        self.channel_label.pack(side=tk.LEFT)
        
        self.instructions_label = ttk.Label(
            info_frame, 
            text="Кликните для переключения каналов: Gray → R → G → B → Gray",
            font=("Arial", 10)
        )
        self.instructions_label.pack(side=tk.RIGHT)
        
        # Canvas для отображения изображения
        self.image_canvas = tk.Canvas(
            main_frame, 
            bg="gray", 
            cursor="crosshair"
        )
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Привязываем клик для переключения каналов
        self.image_canvas.bind("<Button-1>", self._on_canvas_click)
        
        # Кнопки управления
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(
            control_frame, 
            text="Серый", 
            command=lambda: self._switch_channel(0)
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            control_frame, 
            text="Красный", 
            command=lambda: self._switch_channel(1)
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            control_frame, 
            text="Зеленый", 
            command=lambda: self._switch_channel(2)
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            control_frame, 
            text="Синий", 
            command=lambda: self._switch_channel(3)
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            control_frame, 
            text="Закрыть", 
            command=self._close_window
        ).pack(side=tk.RIGHT)
    
    def _setup_bindings(self):
        """Настраивает привязки клавиш."""
        self.root.bind("<Escape>", lambda e: self._close_window())
        self.root.bind("<space>", lambda e: self._next_channel())
        self.root.bind("<Left>", lambda e: self._previous_channel())
        self.root.bind("<Right>", lambda e: self._next_channel())
        
        # Цифровые клавиши для быстрого переключения
        self.root.bind("<Key-1>", lambda e: self._switch_channel(0))
        self.root.bind("<Key-2>", lambda e: self._switch_channel(1))
        self.root.bind("<Key-3>", lambda e: self._switch_channel(2))
        self.root.bind("<Key-4>", lambda e: self._switch_channel(3))
    
    def _on_canvas_click(self, event):
        """Обработчик клика по canvas - переключает канал."""
        self._next_channel()
    
    def _next_channel(self):
        """Переключает на следующий канал."""
        self.current_channel = (self.current_channel + 1) % 4
        self._update_display()
    
    def _previous_channel(self):
        """Переключает на предыдущий канал."""
        self.current_channel = (self.current_channel - 1) % 4
        self._update_display()
    
    def _switch_channel(self, channel_index: int):
        """
        Переключает на указанный канал.
        
        Args:
            channel_index: Индекс канала (0: Gray, 1: Red, 2: Green, 3: Blue)
        """
        if 0 <= channel_index <= 3:
            self.current_channel = channel_index
            self._update_display()
    
    def _update_display(self):
        """Обновляет отображение текущего канала."""
        if not self.channel_images:
            return
        
        # Получаем названия каналов
        channel_names = ["Gray", "Red", "Green", "Blue"]
        channel_keys = ["gray", "red", "green", "blue"]
        
        # Обновляем заголовок
        self.channel_label.config(text=f"{channel_names[self.current_channel]} канал")
        
        # Получаем текущий канал
        current_key = channel_keys[self.current_channel]
        if current_key not in self.channel_images:
            return
        
        channel_image = self.channel_images[current_key]
        
        # Конвертируем в PIL Image
        if len(channel_image.shape) == 2:
            # Серый канал
            pil_image = Image.fromarray(channel_image, mode='L')
        else:
            # Цветной канал
            pil_image = Image.fromarray(channel_image)
        
        # Масштабируем изображение для отображения
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            # Вычисляем масштаб с сохранением пропорций
            img_width, img_height = pil_image.size
            scale = min(canvas_width / img_width, canvas_height / img_height)
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Конвертируем в PhotoImage
        self.display_image = ImageTk.PhotoImage(pil_image)
        
        # Очищаем canvas и отображаем изображение
        self.image_canvas.delete("all")
        self.image_canvas.create_image(
            canvas_width // 2, canvas_height // 2,
            image=self.display_image, anchor=tk.CENTER
        )
    
    def load_image(self, image: np.ndarray):
        """
        Загружает изображение и создает каналы.
        
        Args:
            image: Изображение в формате BGR
        """
        if image is None:
            return
        
        # Создаем каналы
        self._create_channels(image)
        
        # Обновляем отображение
        self._update_display()
    
    def _create_channels(self, bgr_image: np.ndarray):
        """
        Создает каналы из BGR изображения.
        
        Args:
            bgr_image: Изображение в формате BGR
        """
        # Разделяем на каналы
        blue_channel, green_channel, red_channel = cv2.split(bgr_image)
        
        # Создаем серый канал
        gray_channel = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        
        # Сохраняем каналы
        self.channel_images = {
            "gray": gray_channel,
            "red": red_channel,
            "green": green_channel,
            "blue": blue_channel
        }
    
    def _close_window(self):
        """Закрывает окно."""
        if self.parent:
            self.root.destroy()
        else:
            self.root.quit()
    
    def show(self):
        """Показывает окно."""
        if self.root:
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()
    
    def hide(self):
        """Скрывает окно."""
        if self.root:
            self.root.withdraw()
    
    def is_visible(self) -> bool:
        """Проверяет, видимо ли окно."""
        if self.root:
            return self.root.winfo_viewable()
        return False


def create_channel_viewer(parent=None) -> ChannelViewer:
    """
    Создает и возвращает экземпляр ChannelViewer.
    
    Args:
        parent: Родительское окно (опционально)
        
    Returns:
        Экземпляр ChannelViewer
    """
    return ChannelViewer(parent)
