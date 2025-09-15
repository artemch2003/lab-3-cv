#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Главный файл приложения для обработки изображений.

Современное приложение с архитектурой MVC для интерактивной обработки RGB-изображений.
Поддерживает как GUI интерфейс (tkinter), так и классический OpenCV интерфейс.

Использование:
    python main.py                    # Запуск GUI интерфейса
    python main.py --opencv           # Запуск OpenCV интерфейса
    python main.py image.png          # Запуск GUI с загрузкой изображения
    python main.py --opencv image.png # Запуск OpenCV с загрузкой изображения
"""

import sys
import argparse
from controllers.gui_controller import GUIController
from controllers.opencv_controller import OpenCVController
from utils.file_utils import get_image_path_from_args, get_image_path_from_user, validate_image_path


def parse_arguments():
    """
    Парсит аргументы командной строки.
    
    Returns:
        Объект с аргументами
    """
    parser = argparse.ArgumentParser(
        description="Приложение для обработки изображений",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python main.py                    # GUI интерфейс
  python main.py --opencv           # OpenCV интерфейс
  python main.py image.png          # GUI с изображением
  python main.py --opencv image.png # OpenCV с изображением
        """
    )
    
    parser.add_argument(
        "--opencv", 
        action="store_true", 
        help="Использовать OpenCV интерфейс вместо GUI"
    )
    
    parser.add_argument(
        "image_path", 
        nargs="?", 
        help="Путь к изображению для загрузки"
    )
    
    return parser.parse_args()


def run_gui_interface(image_path=None):
    """
    Запускает GUI интерфейс.
    
    Args:
        image_path: Путь к изображению для загрузки
    """
    print("Запуск GUI интерфейса...")
    
    controller = GUIController()
    
    # Загружаем изображение если указан путь
    if image_path:
        if validate_image_path(image_path):
            controller._load_image(image_path)
        else:
            print(f"Ошибка: Неверный формат файла: {image_path}")
    
    # Запускаем приложение
    controller.run()


def run_opencv_interface(image_path=None):
    """
    Запускает OpenCV интерфейс.
    
    Args:
        image_path: Путь к изображению для загрузки
    """
    print("Запуск OpenCV интерфейса...")
    
    # Получаем путь к изображению
    if not image_path:
        image_path = get_image_path_from_args()
        if not image_path:
            image_path = get_image_path_from_user()
    
    # Проверяем путь
    if not validate_image_path(image_path):
        print(f"Ошибка: Неверный формат файла: {image_path}")
        return
    
    # Создаем контроллер и загружаем изображение
    controller = OpenCVController()
    
    if not controller.load_image(image_path):
        print(f"Ошибка: Не удалось загрузить изображение: {image_path}")
        return
    
    print(f"Загружено изображение: {image_path}")
    print("Управление:")
    print("  S - сохранить изображение")
    print("  H - горизонтальное отражение")
    print("  V - вертикальное отражение")
    print("  Q/ESC - выход")
    
    # Создаем интерфейс
    controller.create_windows()
    controller.create_trackbars()
    controller.setup_mouse_callback()
    
    # Запускаем главный цикл
    controller.run_main_loop()


def main():
    """Главная функция приложения."""
    try:
        # Парсим аргументы
        args = parse_arguments()
        
        # Выбираем интерфейс
        if args.opencv:
            run_opencv_interface(args.image_path)
        else:
            run_gui_interface(args.image_path)
            
    except KeyboardInterrupt:
        print("\nПриложение прервано пользователем")
    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
