import os
import csv
import shutil
from config import Config

def prepare_emopia_dataset():
    """
    Подготавливает датасет EMOPIA:
    1. Копирует MIDI файлы в рабочую директорию
    2. Создает файл метаданных с информацией об эмоциях
    3. Обрабатывает файл с лейблами
    
    Возвращает:
        bool: True если подготовка успешна, иначе False
    """
    print("Подготовка датасета...")
    Config.setup_dirs()

    # Чтение эмоциональных лейблов
    emotion_map = {}
    label_path = os.path.join(Config.emopia_dir, "label.csv")
    if not os.path.exists(label_path):
        print(f"Label file not found: {label_path}")
        return False

    with open(label_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            song_id = row['ID']
            emotion = int(row['4Q']) - 1  # Конвертация 1-4 -> 0-3
            emotion_map[song_id] = emotion

    # Создание метаданных
    with open(Config.metadata_path, 'w', newline='') as csvfile:
        fieldnames = ['midi_file', 'emotion', 'original_path']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        source_dir = os.path.join(Config.emopia_dir, "midis")
        if not os.path.exists(source_dir):
            print(f"MIDI directory not found: {source_dir}")
            return False

        # Обработка MIDI файлов
        for filename in os.listdir(source_dir):
            if filename.endswith(".mid"):
                song_id = filename.replace('.mid', '')
                emotion = emotion_map.get(song_id, -1)
                if emotion == -1:
                    continue

                src_path = os.path.join(source_dir, filename)
                dest_path = os.path.join(Config.midi_dir, filename)
                shutil.copy2(src_path, dest_path)

                writer.writerow({
                    'midi_file': filename,
                    'emotion': emotion,
                    'original_path': src_path
                })

    print(f"Подготовлено {len(os.listdir(Config.midi_dir))} MIDI файлов")
    return True
