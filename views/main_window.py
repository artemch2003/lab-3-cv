#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Главное окно приложения с современным интерфейсом.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import cv2
import numpy as np
from PIL import Image, ImageTk
from typing import Optional, Callable
from config.settings import GUI_SETTINGS
from views.channel_viewer import ChannelViewer


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
        
        # Окно просмотра каналов
        self.channel_viewer = None
        
        # Режим отображения каналов
        self.channel_mode = False  # False - обычное изображение, True - каналы
        self.current_channel = 0  # 0: Gray, 1: Red, 2: Green, 3: Blue
        self.channel_images = {}  # Словарь для хранения каналов
        
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
            # High-pass
            "hp_enable": tk.BooleanVar(value=False),
            "hp_blur_mode": tk.IntVar(value=0),  # 0 mean, 1 gaussian
            "hp_kernel": tk.IntVar(value=3),
            "hp_scale_x100": tk.IntVar(value=100),
            # Convolution
            "conv_enable": tk.BooleanVar(value=False),
            "conv_normalize": tk.BooleanVar(value=True),
            "conv_add128": tk.BooleanVar(value=False),
            "conv_kernel_size": tk.IntVar(value=3),
            "conv_kernel_text": tk.StringVar(value=""),
            "conv_preset": tk.StringVar(value="Пользовательская"),
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
        view_menu.add_separator()
        view_menu.add_command(label="Просмотр каналов", command=self._show_channel_viewer)
    
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
        flip_v_check.pack(anchor=tk.W, pady=(0, 10))

        # High-pass блок
        hp_frame = ttk.LabelFrame(left_frame, text="High-pass фильтр", padding=8)
        hp_frame.pack(fill=tk.X, pady=(0, 10))

        hp_enable_check = ttk.Checkbutton(
            hp_frame, text="Включить High-pass", variable=self.params["hp_enable"],
            command=self._on_parameter_change
        )
        hp_enable_check.pack(anchor=tk.W)

        ttk.Label(hp_frame, text="Тип размытия:").pack(anchor=tk.W)
        hp_mode_combo = ttk.Combobox(
            hp_frame, state="readonly", width=14,
            values=["Усреднение", "Гаусс"],
        )
        # Привязка комбобокса к IntVar через manual sync
        hp_mode_combo.current(self.params["hp_blur_mode"].get())
        def _on_hp_mode_change(event=None):
            self.params["hp_blur_mode"].set(hp_mode_combo.current())
            self._on_parameter_change()
        hp_mode_combo.bind("<<ComboboxSelected>>", _on_hp_mode_change)
        hp_mode_combo.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(hp_frame, text="Размер ядра (нечетн.):").pack(anchor=tk.W)
        hp_kernel_scale = ttk.Scale(
            hp_frame, from_=3, to=25, orient=tk.HORIZONTAL,
            variable=self.params["hp_kernel"], command=self._on_parameter_change
        )
        hp_kernel_scale.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(hp_frame, text="Коэфф. c (x100):").pack(anchor=tk.W)
        hp_scale = ttk.Scale(
            hp_frame, from_=0, to=300, orient=tk.HORIZONTAL,
            variable=self.params["hp_scale_x100"], command=self._on_parameter_change
        )
        hp_scale.pack(fill=tk.X)

        # Свёртка (конволюция)
        conv_frame = ttk.LabelFrame(left_frame, text="Свёртка (конволюция)", padding=8)
        conv_frame.pack(fill=tk.X, pady=(0, 10))

        conv_enable_check = ttk.Checkbutton(
            conv_frame, text="Включить свёртку", variable=self.params["conv_enable"],
            command=self._on_parameter_change
        )
        conv_enable_check.pack(anchor=tk.W)

        ttk.Label(conv_frame, text="Пресет ядра:").pack(anchor=tk.W)
        conv_presets = [
            "Пользовательская",
            "Identity 3x3",
            "Box 3x3",
            "Gaussian 3x3",
            "Sobel X",
            "Sobel Y",
            "Prewitt X",
            "Prewitt Y",
            "Laplacian 4",
            "Laplacian 8",
        ]
        conv_preset_combo = ttk.Combobox(
            conv_frame, state="readonly", width=18,
            values=conv_presets, textvariable=self.params["conv_preset"]
        )
        conv_preset_combo.pack(fill=tk.X, pady=(0, 6))
        conv_preset_combo.bind("<<ComboboxSelected>>", self._on_parameter_change)

        ttk.Label(conv_frame, text="Размер ядра (n×n, нечетн.):").pack(anchor=tk.W)
        conv_kernel_scale = ttk.Scale(
            conv_frame, from_=1, to=25, orient=tk.HORIZONTAL,
            variable=self.params["conv_kernel_size"], command=self._on_parameter_change
        )
        conv_kernel_scale.pack(fill=tk.X, pady=(0, 6))

        conv_norm_check = ttk.Checkbutton(
            conv_frame, text="Нормализация (делить на сумму ядра)",
            variable=self.params["conv_normalize"], command=self._on_parameter_change
        )
        conv_norm_check.pack(anchor=tk.W)

        conv_add128_check = ttk.Checkbutton(
            conv_frame, text="+128 после свёртки",
            variable=self.params["conv_add128"], command=self._on_parameter_change
        )
        conv_add128_check.pack(anchor=tk.W)

        ttk.Label(conv_frame, text="Матрица ядра (через пробел/запятую/перенос строки):").pack(anchor=tk.W, pady=(6, 0))
        self.conv_kernel_text_widget = scrolledtext.ScrolledText(conv_frame, height=6, wrap=tk.WORD)
        self.conv_kernel_text_widget.pack(fill=tk.BOTH, expand=False)
        # Инициализируем содержимое из переменной
        self.conv_kernel_text_widget.insert("1.0", self.params["conv_kernel_text"].get())

        def _on_conv_text_change(event=None):
            # Синхронизация содержимого Text -> StringVar и триггер перерасчёта
            self.params["conv_kernel_text"].set(self.conv_kernel_text_widget.get("1.0", tk.END))
            self._on_parameter_change()

        self.conv_kernel_text_widget.bind("<KeyRelease>", _on_conv_text_change)
        
        # Режим каналов
        ttk.Label(left_frame, text="Режим просмотра:").pack(anchor=tk.W)
        
        self.channel_mode_var = tk.BooleanVar(value=False)
        channel_mode_check = ttk.Checkbutton(
            left_frame, text="Просмотр каналов", variable=self.channel_mode_var,
            command=self._toggle_channel_mode
        )
        channel_mode_check.pack(anchor=tk.W)
        
        # Информация о текущем канале
        self.channel_info_label = ttk.Label(
            left_frame, text="Серый канал", font=("Arial", 10, "bold")
        )
        self.channel_info_label.pack(anchor=tk.W, pady=(5, 0))
    
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
        ttk.Button(buttons_frame, text="Обновить дополнительные окна", command=self._update_additional_windows).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(buttons_frame, text="Просмотр каналов", command=self._show_channel_viewer).pack(fill=tk.X)
    
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
        
        # Горячие клавиши для переключения каналов
        self.root.bind("<space>", lambda e: self._next_channel() if self.channel_mode else None)
        self.root.bind("<Right>", lambda e: self._next_channel() if self.channel_mode else None)
        self.root.bind("<Left>", lambda e: self._previous_channel() if self.channel_mode else None)
        
        # Цифровые клавиши для быстрого переключения каналов
        self.root.bind("<Key-1>", lambda e: self._switch_to_channel(0) if self.channel_mode else None)
        self.root.bind("<Key-2>", lambda e: self._switch_to_channel(1) if self.channel_mode else None)
        self.root.bind("<Key-3>", lambda e: self._switch_to_channel(2) if self.channel_mode else None)
        self.root.bind("<Key-4>", lambda e: self._switch_to_channel(3) if self.channel_mode else None)
    
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
        # Если включен режим каналов, переключаем канал
        if self.channel_mode:
            self._next_channel()
        else:
            # Обычное поведение - обновляем позицию мыши
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
        # High-pass
        self.params["hp_enable"].set(False)
        self.params["hp_blur_mode"].set(0)
        self.params["hp_kernel"].set(3)
        self.params["hp_scale_x100"].set(100)
        # Convolution
        self.params["conv_enable"].set(False)
        self.params["conv_normalize"].set(True)
        self.params["conv_add128"].set(False)
        self.params["conv_kernel_size"].set(3)
        self.params["conv_kernel_text"].set("")
        self.params["conv_preset"].set("Пользовательская")
        
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
    
    def _show_channel_viewer(self):
        """Показывает окно просмотра каналов."""
        if self.processed_image is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение")
            return
        
        # Создаем окно просмотра каналов если его еще нет
        if self.channel_viewer is None:
            self.channel_viewer = ChannelViewer(self.root)
        
        # Загружаем изображение в окно просмотра
        self.channel_viewer.load_image(self.processed_image)
        
        # Показываем окно
        self.channel_viewer.show()
    
    def _toggle_channel_mode(self):
        """Переключает режим просмотра каналов."""
        self.channel_mode = self.channel_mode_var.get()
        
        if self.channel_mode and self.processed_image is not None:
            # Создаем каналы из текущего изображения
            self._create_channels(self.processed_image)
            self._update_channel_display()
        elif not self.channel_mode:
            # Возвращаемся к обычному отображению
            if self.update_callback:
                self.update_callback()
    
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
    
    def _next_channel(self):
        """Переключает на следующий канал."""
        if not self.channel_mode or not self.channel_images:
            return
        
        self.current_channel = (self.current_channel + 1) % 4
        self._update_channel_display()
    
    def _update_channel_display(self):
        """Обновляет отображение текущего канала."""
        if not self.channel_mode or not self.channel_images:
            return
        
        # Получаем названия каналов
        channel_names = ["Серый", "Красный", "Зеленый", "Синий"]
        channel_keys = ["gray", "red", "green", "blue"]
        
        # Обновляем информацию о канале
        self.channel_info_label.config(text=f"{channel_names[self.current_channel]} канал")
        
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
    
    def set_update_callback(self, callback: Callable):
        """Устанавливает callback для обновления изображения."""
        self.update_callback = callback
    
    def update_image_display(self, image: np.ndarray):
        """Обновляет отображение изображения."""
        if image is None:
            return
        
        # Если включен режим каналов, обновляем каналы и отображаем текущий
        if self.channel_mode:
            self._create_channels(image)
            self._update_channel_display()
            return
        
        # Обычное отображение изображения
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
    
    def _previous_channel(self):
        """Переключает на предыдущий канал."""
        if not self.channel_mode or not self.channel_images:
            return
        
        self.current_channel = (self.current_channel - 1) % 4
        self._update_channel_display()
    
    def _switch_to_channel(self, channel_index: int):
        """
        Переключает на указанный канал.
        
        Args:
            channel_index: Индекс канала (0: Gray, 1: Red, 2: Green, 3: Blue)
        """
        if not self.channel_mode or not self.channel_images:
            return
        
        if 0 <= channel_index <= 3:
            self.current_channel = channel_index
            self._update_channel_display()
    
    def run(self):
        """Запускает главный цикл приложения."""
        self.root.mainloop()
