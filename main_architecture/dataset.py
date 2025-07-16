import os
import csv
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from config import Config
from tokenizer import MIDITokenizer

class EMOPIADataset(Dataset):
    """
    Загрузчик датасета EMOPIA
    Преобразует MIDI файлы в тензоры для обучения модели
    
    Особенности:
    - Автоматическая пакетизация
    - Создание масок внимания
    - Поддержка паддинга
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.samples = []
        self.load_metadata()

    def load_metadata(self):
        """Загружает метаданные и токенизирует MIDI файлы"""
        if not os.path.exists(Config.metadata_path):
            print(f"Не найдены файлы metadata: {Config.metadata_path}")
            return

        metadata = []
        with open(Config.metadata_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata.append((row['midi_file'], int(row['emotion'])))

        print("Токенизация MIDI файлов...")
        for midi_file, emotion in tqdm(metadata):
            midi_path = os.path.join(Config.midi_dir, midi_file)
            token_ids = self.tokenizer.tokenize_midi(midi_path, emotion)
            if token_ids:
                self.samples.append((token_ids, emotion))

        print(f"Загружено {len(self.samples)} valid экземпляров")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        token_ids, emotion = self.samples[idx]
        
        # Применение паддинга
        padded_ids = np.zeros(Config.max_sequence_length, dtype=np.int64)
        padded_ids[:len(token_ids)] = token_ids
        
        # Создание маски внимания
        attention_mask = np.zeros(Config.max_sequence_length, dtype=np.int64)
        attention_mask[:len(token_ids)] = 1

        return {
            'input_ids': torch.tensor(padded_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'emotion': torch.tensor(emotion, dtype=torch.long)
        }
