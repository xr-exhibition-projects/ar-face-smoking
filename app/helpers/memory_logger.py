"""
Утилита для логирования использования памяти GPU и других метрик производительности.
"""
import logging
import os
from pathlib import Path
from datetime import datetime
import torch

# Настройка логгера
def setup_memory_logger(log_dir: Path = None) -> logging.Logger:
    """
    Настраивает логгер для записи информации о памяти и производительности.
    
    Args:
        log_dir: Директория для сохранения логов. Если None, используется текущая директория.
    
    Returns:
        Настроенный логгер.
    """
    if log_dir is None:
        log_dir = Path.cwd() / "logs"
    else:
        log_dir = Path(log_dir)
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Создаем имя файла с датой и временем
    log_file = log_dir / f"memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Настраиваем логгер
    logger = logging.getLogger("MemoryLogger")
    logger.setLevel(logging.INFO)
    
    # Удаляем существующие обработчики, чтобы избежать дублирования
    logger.handlers.clear()
    
    # Обработчик для файла - записываем все логи (INFO и DEBUG)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Обработчик для консоли - только важные события (WARNING и выше)
    # Детальные логи (GPU Memory, Cache Clear) идут только в файл
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Только WARNING и ERROR в консоль
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Memory logger initialized. Log file: {log_file}")
    return logger

def log_gpu_memory(logger: logging.Logger, context: str = "", log_to_console: bool = False):
    """
    Логирует текущее использование памяти GPU.
    
    Args:
        logger: Логгер для записи.
        context: Контекст, в котором вызывается логирование (например, "after_frame_processing").
        log_to_console: Если True, логирует также в консоль (по умолчанию только в файл).
    """
    if torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
            max_reserved = torch.cuda.max_memory_reserved() / 1024**3  # GB
            
            message = (
                f"[GPU Memory] {context} - "
                f"Allocated: {allocated:.2f} GB, "
                f"Reserved: {reserved:.2f} GB, "
                f"Max Allocated: {max_allocated:.2f} GB, "
                f"Max Reserved: {max_reserved:.2f} GB"
            )
            # Всегда логируем в файл, в консоль только если явно указано
            if log_to_console:
                logger.info(message)
            else:
                # Используем DEBUG уровень для детальных логов, чтобы они не попадали в консоль
                logger.debug(message)
        except Exception as e:
            logger.warning(f"[GPU Memory] Failed to get GPU memory info: {e}")
    else:
        if log_to_console:
            logger.info(f"[GPU Memory] {context} - CUDA not available")
        else:
            logger.debug(f"[GPU Memory] {context} - CUDA not available")

def log_buffer_size(logger: logging.Logger, buffer_name: str, size: int, max_size: int = None, log_to_console: bool = False):
    """
    Логирует размер буфера кадров.
    
    Args:
        logger: Логгер для записи.
        buffer_name: Имя буфера (например, "frames_to_display").
        size: Текущий размер буфера.
        max_size: Максимальный размер буфера (опционально).
        log_to_console: Если True, логирует также в консоль (по умолчанию только в файл).
    """
    if max_size:
        percentage = (size / max_size) * 100
        message = f"[Buffer] {buffer_name}: {size}/{max_size} ({percentage:.1f}%)"
    else:
        message = f"[Buffer] {buffer_name}: {size}"
    
    # Всегда логируем в файл, в консоль только если явно указано или если буфер переполнен
    if log_to_console or (max_size and size >= max_size):
        logger.info(message)
    else:
        # Используем DEBUG уровень для детальных логов, чтобы они не попадали в консоль
        logger.debug(message)

def log_cache_clear(logger: logging.Logger, context: str = "", log_to_console: bool = False):
    """
    Логирует очистку GPU кэша.
    
    Args:
        logger: Логгер для записи.
        context: Контекст очистки кэша.
        log_to_console: Если True, логирует также в консоль (по умолчанию только в файл).
    """
    message = f"[Cache Clear] {context}"
    # Всегда логируем в файл, в консоль только если явно указано
    if log_to_console:
        logger.info(message)
    else:
        # Используем DEBUG уровень для детальных логов, чтобы они не попадали в консоль
        logger.debug(message)

def log_frame_processing_time(logger: logging.Logger, frame_number: int, processing_time: float, log_to_console: bool = False):
    """
    Логирует время обработки кадра.
    
    Args:
        logger: Логгер для записи.
        frame_number: Номер кадра.
        processing_time: Время обработки в секундах.
        log_to_console: Если True, логирует также в консоль (по умолчанию только в файл).
    """
    message = f"[Frame Processing] Frame {frame_number}: {processing_time*1000:.2f} ms"
    # Всегда логируем в файл, в консоль только если явно указано
    if log_to_console:
        logger.info(message)
    else:
        # Используем DEBUG уровень для детальных логов, чтобы они не попадали в консоль
        logger.debug(message)

# Глобальный логгер (инициализируется при первом использовании)
_memory_logger = None

def get_memory_logger() -> logging.Logger:
    """Возвращает глобальный логгер памяти."""
    global _memory_logger
    if _memory_logger is None:
        _memory_logger = setup_memory_logger()
    return _memory_logger

