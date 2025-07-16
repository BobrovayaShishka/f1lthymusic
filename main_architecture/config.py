import os

class Config:
    """
    Конфигурация системы для обработки данных, обучения модели и генерации музыки.
    Все пути и гиперпараметры заданы здесь для централизованного управления.
    """
    # Пути данных
    emopia_dir = "EMOPIA_1.0/EMOPIA_1.0"
    processed_dir = "emopia_processed"
    midi_dir = os.path.join(processed_dir, "midis")
    metadata_path = os.path.join(processed_dir, "metadata.csv")

    # Параметры токенизации
    max_sequence_length = 2048
    vocab_size = 3500
    emotion_classes = 4  # HA/HV, HA/LV, LA/HV, LA/LV
    positions_per_bar = 16

    # Параметры модели Transformer
    d_model = 512
    nhead = 8
    num_layers = 6
    dim_feedforward = 2048
    dropout = 0.2

    # Параметры обучения
    batch_size = 16
    learning_rate = 1e-4
    weight_decay = 1e-5
    epochs = 50
    early_stopping_patience = 10
    train_ratio = 0.8
    validation_ratio = 0.1
    test_ratio = 0.1

    # Параметры генерации
    generation_temperature = 0.85
    max_generation_length = 2048
    target_duration = 20.0  # секунды

    # Сохранение результатов
    save_dir = "emopia_results"
    model_name = "emotion_music_transformer"

    @classmethod
    def setup_dirs(cls):
        """Создает необходимые директории для проекта"""
        os.makedirs(cls.processed_dir, exist_ok=True)
        os.makedirs(cls.midi_dir, exist_ok=True)
        os.makedirs(cls.save_dir, exist_ok=True)
