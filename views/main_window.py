#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Главное окно приложения с современным интерфейсом.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from typing import Optional, Callable
from config.settings import GUI_SETTINGS


class ScrollableFrame(ttk.Frame):
    """Прокручиваемая рамка с полосой прокрутки."""
    
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        
        # Создаем canvas и scrollbar
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        # Настраиваем прокрутку
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Размещаем элементы
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Привязываем события мыши для прокрутки
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", self._on_mousewheel)
        self.canvas.bind("<Button-5>", self._on_mousewheel)
        
        # Фокус для работы с клавиатурой
        self.canvas.bind("<1>", lambda event: self.canvas.focus_set())
    
    def _on_mousewheel(self, event):
        """Обработчик прокрутки колесом мыши."""
        if event.delta:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        elif event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")
    
    def get_scrollable_frame(self):
        """Возвращает прокручиваемую рамку для размещения виджетов."""
        return self.scrollable_frame


class MainWindow:
    """
    Главное окно приложения с современным интерфейсом.
    """
    
    def __init__(self):
        """Инициализация главного окна."""
        self.root = tk.Tk()
        self.root.title(GUI_SETTINGS["window_title"])
        self.root.geometry("1200x800")
        self.root.minsize(*GUI_SETTINGS["min_window_size"])
        
        # Переменные для хранения изображений
        self.original_image = None
        self.processed_image = None
        self.display_image = None
        self.zoom_image = None
        self.channels_image = None
        self.histograms_image = None
        
        # Переменные для дополнительных изображений в правой панели
        self.channels_display_image = None
        self.histograms_display_image = None
        
        # Параметры обработки
        self.params = {
            "brightness": tk.IntVar(value=100),
            "contrast": tk.IntVar(value=100),
            "r_offset": tk.IntVar(value=100),
            "g_offset": tk.IntVar(value=100),
            "b_offset": tk.IntVar(value=100),
            "gamma_x10": tk.IntVar(value=10),
            "swap_mode": tk.StringVar(value="BGR"),
            "negate_r": tk.BooleanVar(value=False),
            "negate_g": tk.BooleanVar(value=False),
            "negate_b": tk.BooleanVar(value=False),
            "flip_horizontal": tk.BooleanVar(value=False),
            "flip_vertical": tk.BooleanVar(value=False),
        }
        
        # Позиция мыши
        self.mouse_pos = (0, 0)
        
        # Callback для обновления изображения
        self.update_callback: Optional[Callable] = None
        
        self._create_widgets()
        self._setup_bindings()
    
    def _create_widgets(self):
        """Создает виджеты интерфейса."""
        # Главный фрейм
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Создаем панели
        self._create_menu_bar()
        self._create_left_panel(main_frame)
        self._create_center_panel(main_frame)
        self._create_right_panel(main_frame)
        self._create_status_bar()
    
    def _create_menu_bar(self):
        """Создает меню приложения."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # Файл
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Файл", menu=file_menu)
        file_menu.add_command(label="Открыть изображение...", command=self._open_image)
        file_menu.add_separator()
        file_menu.add_command(label="Сохранить", command=self._save_image)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.root.quit)
        
        # Вид
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Вид", menu=view_menu)
        view_menu.add_command(label="Сбросить параметры", command=self._reset_parameters)
        view_menu.add_command(label="Дополнительные окна", command=self._show_additional_windows)
    
    def _create_left_panel(self, parent):
        """Создает левую панель с параметрами."""
        left_frame = ttk.LabelFrame(parent, text="Параметры обработки", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Яркость
        ttk.Label(left_frame, text="Яркость:").pack(anchor=tk.W)
        brightness_scale = ttk.Scale(
            left_frame, from_=0, to=200, variable=self.params["brightness"],
            orient=tk.HORIZONTAL, command=self._on_parameter_change
        )
        brightness_scale.pack(fill=tk.X, pady=(0, 10))
        
        # Контраст
        ttk.Label(left_frame, text="Контраст:").pack(anchor=tk.W)
        contrast_scale = ttk.Scale(
            left_frame, from_=50, to=300, variable=self.params["contrast"],
            orient=tk.HORIZONTAL, command=self._on_parameter_change
        )
        contrast_scale.pack(fill=tk.X, pady=(0, 10))
        
        # Сдвиги каналов
        ttk.Label(left_frame, text="Сдвиги каналов:").pack(anchor=tk.W)
        
        ttk.Label(left_frame, text="R:").pack(anchor=tk.W)
        r_scale = ttk.Scale(
            left_frame, from_=0, to=200, variable=self.params["r_offset"],
            orient=tk.HORIZONTAL, command=self._on_parameter_change
        )
        r_scale.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(left_frame, text="G:").pack(anchor=tk.W)
        g_scale = ttk.Scale(
            left_frame, from_=0, to=200, variable=self.params["g_offset"],
            orient=tk.HORIZONTAL, command=self._on_parameter_change
        )
        g_scale.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(left_frame, text="B:").pack(anchor=tk.W)
        b_scale = ttk.Scale(
            left_frame, from_=0, to=200, variable=self.params["b_offset"],
            orient=tk.HORIZONTAL, command=self._on_parameter_change
        )
        b_scale.pack(fill=tk.X, pady=(0, 10))
        
        # Гамма
        ttk.Label(left_frame, text="Гамма (×10):").pack(anchor=tk.W)
        gamma_scale = ttk.Scale(
            left_frame, from_=5, to=40, variable=self.params["gamma_x10"],
            orient=tk.HORIZONTAL, command=self._on_parameter_change
        )
        gamma_scale.pack(fill=tk.X, pady=(0, 10))
        
        # Перестановка каналов
        ttk.Label(left_frame, text="Перестановка каналов:").pack(anchor=tk.W)
        swap_combo = ttk.Combobox(
            left_frame, textvariable=self.params["swap_mode"],
            values=["BGR", "BRG", "GBR", "GRB", "RBG", "RGB"],
            state="readonly", width=10
        )
        swap_combo.pack(fill=tk.X, pady=(0, 10))
        swap_combo.bind("<<ComboboxSelected>>", self._on_parameter_change)
        
        # Инверсия каналов
        ttk.Label(left_frame, text="Инверсия каналов:").pack(anchor=tk.W)
        
        neg_r_check = ttk.Checkbutton(
            left_frame, text="R", variable=self.params["negate_r"],
            command=self._on_parameter_change
        )
        neg_r_check.pack(anchor=tk.W)
        
        neg_g_check = ttk.Checkbutton(
            left_frame, text="G", variable=self.params["negate_g"],
            command=self._on_parameter_change
        )
        neg_g_check.pack(anchor=tk.W)
        
        neg_b_check = ttk.Checkbutton(
            left_frame, text="B", variable=self.params["negate_b"],
            command=self._on_parameter_change
        )
        neg_b_check.pack(anchor=tk.W, pady=(0, 10))
        
        # Отражения
        ttk.Label(left_frame, text="Отражения:").pack(anchor=tk.W)
        
        flip_h_check = ttk.Checkbutton(
            left_frame, text="Горизонтальное", variable=self.params["flip_horizontal"],
            command=self._on_parameter_change
        )
        flip_h_check.pack(anchor=tk.W)
        
        flip_v_check = ttk.Checkbutton(
            left_frame, text="Вертикальное", variable=self.params["flip_vertical"],
            command=self._on_parameter_change
        )
        flip_v_check.pack(anchor=tk.W)
    
    def _create_center_panel(self, parent):
        """Создает центральную панель с изображением."""
        center_frame = ttk.Frame(parent)
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Фрейм для изображения
        image_frame = ttk.LabelFrame(center_frame, text="Изображение", padding=5)
        image_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas для изображения
        self.image_canvas = tk.Canvas(
            image_frame, bg="gray", width=640, height=480,
            cursor="crosshair"
        )
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Привязываем события мыши
        self.image_canvas.bind("<Motion>", self._on_mouse_move)
        self.image_canvas.bind("<Button-1>", self._on_mouse_click)
    
    def _create_right_panel(self, parent):
        """Создает правую панель с информацией и дополнительными окнами."""
        right_frame = ttk.LabelFrame(parent, text="Информация и дополнительные окна", padding=5)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # Создаем прокручиваемую область
        scrollable_area = ScrollableFrame(right_frame)
        scrollable_area.pack(fill=tk.BOTH, expand=True)
        
        # Получаем прокручиваемую рамку для размещения виджетов
        content_frame = scrollable_area.get_scrollable_frame()
        
        # Информация о пикселе
        pixel_frame = ttk.LabelFrame(content_frame, text="Информация о пикселе", padding=5)
        pixel_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.pixel_info = tk.Text(pixel_frame, height=6, width=30, wrap=tk.WORD)
        self.pixel_info.pack(fill=tk.BOTH, expand=True)
        
        # Статистики окна
        stats_frame = ttk.LabelFrame(content_frame, text="Статистики окна 11×11", padding=5)
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.stats_info = tk.Text(stats_frame, height=4, width=30, wrap=tk.WORD)
        self.stats_info.pack(fill=tk.BOTH, expand=True)
        
        # Zoom окно
        zoom_frame = ttk.LabelFrame(content_frame, text="Zoom 11×11 x8", padding=5)
        zoom_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.zoom_canvas = tk.Canvas(zoom_frame, bg="gray", width=88, height=88)
        self.zoom_canvas.pack()
        
        # Каналы изображения
        channels_frame = ttk.LabelFrame(content_frame, text="Каналы изображения", padding=5)
        channels_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.channels_canvas = tk.Canvas(channels_frame, bg="gray", width=300, height=200)
        self.channels_canvas.pack()
        
        # Гистограммы
        histograms_frame = ttk.LabelFrame(content_frame, text="Гистограммы", padding=5)
        histograms_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.histograms_canvas = tk.Canvas(histograms_frame, bg="gray", width=300, height=200)
        self.histograms_canvas.pack()
        
        # Кнопки
        buttons_frame = ttk.Frame(content_frame)
        buttons_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(buttons_frame, text="Открыть", command=self._open_image).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(buttons_frame, text="Сохранить", command=self._save_image).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(buttons_frame, text="Сбросить", command=self._reset_parameters).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(buttons_frame, text="Обновить дополнительные окна", command=self._update_additional_windows).pack(fill=tk.X)
    
    def _create_status_bar(self):
        """Создает строку состояния."""
        self.status_bar = ttk.Label(
            self.root, text="Готов к работе", relief=tk.SUNKEN, anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _setup_bindings(self):
        """Настраивает привязки клавиш."""
        self.root.bind("<Control-o>", lambda e: self._open_image())
        self.root.bind("<Control-s>", lambda e: self._save_image())
        self.root.bind("<F5>", lambda e: self._reset_parameters())
        self.root.bind("<Escape>", lambda e: self.root.quit())
    
    def _on_parameter_change(self, event=None):
        """Обработчик изменения параметров."""
        if self.update_callback:
            self.update_callback()
    
    def _on_mouse_move(self, event):
        """Обработчик движения мыши."""
        if self.display_image:
            # Преобразуем координаты canvas в координаты изображения
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                # Получаем размеры изображения на canvas
                img_width = self.display_image.width()
                img_height = self.display_image.height()
                
                # Вычисляем масштаб
                scale_x = img_width / canvas_width
                scale_y = img_height / canvas_height
                
                # Преобразуем координаты
                x = int(event.x * scale_x)
                y = int(event.y * scale_y)
                
                self.mouse_pos = (x, y)
                
                if self.update_callback:
                    self.update_callback()
    
    def _on_mouse_click(self, event):
        """Обработчик клика мыши."""
        self._on_mouse_move(event)
    
    def _open_image(self):
        """Открывает диалог выбора изображения."""
        filetypes = [
            ("Изображения", "*.bmp *.png *.tiff *.tif *.jpg *.jpeg"),
            ("Все файлы", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=filetypes
        )
        
        if filename:
            self.status_bar.config(text=f"Загружается: {filename}")
            if self.update_callback:
                self.update_callback(filename)
    
    def _save_image(self):
        """Сохраняет текущее изображение."""
        if self.processed_image is not None:
            filetypes = [
                ("PNG", "*.png"),
                ("JPEG", "*.jpg"),
                ("BMP", "*.bmp"),
                ("TIFF", "*.tiff")
            ]
            
            filename = filedialog.asksaveasfilename(
                title="Сохранить изображение",
                defaultextension=".png",
                filetypes=filetypes
            )
            
            if filename:
                cv2.imwrite(filename, self.processed_image)
                self.status_bar.config(text=f"Сохранено: {filename}")
                messagebox.showinfo("Успех", f"Изображение сохранено:\n{filename}")
    
    def _reset_parameters(self):
        """Сбрасывает все параметры к значениям по умолчанию."""
        self.params["brightness"].set(100)
        self.params["contrast"].set(100)
        self.params["r_offset"].set(100)
        self.params["g_offset"].set(100)
        self.params["b_offset"].set(100)
        self.params["gamma_x10"].set(10)
        self.params["swap_mode"].set("BGR")
        self.params["negate_r"].set(False)
        self.params["negate_g"].set(False)
        self.params["negate_b"].set(False)
        self.params["flip_horizontal"].set(False)
        self.params["flip_vertical"].set(False)
        
        if self.update_callback:
            self.update_callback()
    
    def _show_additional_windows(self):
        """Показывает дополнительные окна с каналами и гистограммами."""
        if self.update_callback:
            self.update_callback(show_additional=True)
    
    def _update_additional_windows(self):
        """Обновляет дополнительные окна в правой панели."""
        if self.update_callback:
            self.update_callback(update_additional=True)
    
    def set_update_callback(self, callback: Callable):
        """Устанавливает callback для обновления изображения."""
        self.update_callback = callback
    
    def update_image_display(self, image: np.ndarray):
        """Обновляет отображение изображения."""
        if image is None:
            return
        
        # Конвертируем BGR в RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Создаем PIL изображение
        pil_image = Image.fromarray(rgb_image)
        
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
    
    def update_zoom_display(self, zoom_image: np.ndarray):
        """Обновляет отображение zoom окна."""
        if zoom_image is None:
            return
        
        # Конвертируем BGR в RGB
        rgb_image = cv2.cvtColor(zoom_image, cv2.COLOR_BGR2RGB)
        
        # Создаем PIL изображение
        pil_image = Image.fromarray(rgb_image)
        
        # Масштабируем до размера canvas (88x88)
        pil_image = pil_image.resize((88, 88), Image.Resampling.LANCZOS)
        
        # Конвертируем в PhotoImage
        self.zoom_image = ImageTk.PhotoImage(pil_image)
        
        # Очищаем canvas и отображаем изображение
        self.zoom_canvas.delete("all")
        self.zoom_canvas.create_image(44, 44, image=self.zoom_image, anchor=tk.CENTER)
    
    def update_pixel_info(self, info: str):
        """Обновляет информацию о пикселе."""
        self.pixel_info.delete(1.0, tk.END)
        self.pixel_info.insert(1.0, info)
    
    def update_stats_info(self, info: str):
        """Обновляет статистическую информацию."""
        self.stats_info.delete(1.0, tk.END)
        self.stats_info.insert(1.0, info)
    
    def update_channels_display(self, channels_image: np.ndarray):
        """Обновляет отображение каналов в правой панели."""
        if channels_image is None:
            return
        
        # Конвертируем BGR в RGB
        rgb_image = cv2.cvtColor(channels_image, cv2.COLOR_BGR2RGB)
        
        # Создаем PIL изображение
        pil_image = Image.fromarray(rgb_image)
        
        # Масштабируем изображение для отображения (300x200)
        pil_image = pil_image.resize((300, 200), Image.Resampling.LANCZOS)
        
        # Конвертируем в PhotoImage
        self.channels_display_image = ImageTk.PhotoImage(pil_image)
        
        # Очищаем canvas и отображаем изображение
        self.channels_canvas.delete("all")
        self.channels_canvas.create_image(150, 100, image=self.channels_display_image, anchor=tk.CENTER)
    
    def update_histograms_display(self, histograms_image: np.ndarray):
        """Обновляет отображение гистограмм в правой панели."""
        if histograms_image is None:
            return
        
        # Конвертируем BGR в RGB
        rgb_image = cv2.cvtColor(histograms_image, cv2.COLOR_BGR2RGB)
        
        # Создаем PIL изображение
        pil_image = Image.fromarray(rgb_image)
        
        # Масштабируем изображение для отображения (300x200)
        pil_image = pil_image.resize((300, 200), Image.Resampling.LANCZOS)
        
        # Конвертируем в PhotoImage
        self.histograms_display_image = ImageTk.PhotoImage(pil_image)
        
        # Очищаем canvas и отображаем изображение
        self.histograms_canvas.delete("all")
        self.histograms_canvas.create_image(150, 100, image=self.histograms_display_image, anchor=tk.CENTER)
    
    def update_status(self, message: str):
        """Обновляет строку состояния."""
        self.status_bar.config(text=message)
    
    def get_parameters(self) -> dict:
        """Возвращает текущие параметры."""
        return {name: var.get() for name, var in self.params.items()}
    
    def run(self):
        """Запускает главный цикл приложения."""
        self.root.mainloop()
