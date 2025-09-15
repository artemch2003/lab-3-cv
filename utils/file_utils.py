#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Утилиты для работы с файлами.
"""

import os
import sys
from typing import List, Optional
from config.settings import SUPPORTED_FORMATS


def get_image_path_from_args() -> Optional[str]:
    """
    Получает путь к изображению из аргументов командной строки.
    
    Returns:
        Путь к файлу изображения или None
    """
    if len(sys.argv) >= 2:
        return sys.argv[1]
    return None


def get_image_path_from_user() -> str:
    """
    Получает путь к изображению от пользователя.
    
    Returns:
        Путь к файлу изображения
    """
    print("Укажите путь к изображению (BMP/PNG/TIFF без потерь):")
    return input("> ").strip().strip('"')


def validate_image_path(path: str) -> bool:
    """
    Проверяет, является ли путь валидным файлом изображения.
    
    Args:
        path: Путь к файлу
        
    Returns:
        True если файл валиден, False иначе
    """
    if not os.path.exists(path):
        return False
    
    _, ext = os.path.splitext(path.lower())
    return ext in SUPPORTED_FORMATS


def get_save_filename(counter: int) -> str:
    """
    Генерирует имя файла для сохранения.
    
    Args:
        counter: Счетчик сохранений
        
    Returns:
        Имя файла для сохранения
    """
    return f"output_{counter}.png"


def get_supported_formats_string() -> str:
    """
    Возвращает строку с поддерживаемыми форматами.
    
    Returns:
        Строка с форматами
    """
    return ", ".join(SUPPORTED_FORMATS)
