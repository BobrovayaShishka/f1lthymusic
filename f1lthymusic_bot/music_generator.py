import os
import pretty_midi
import random
import time
import logging
import torch
import numpy as np
from config import TEMP_DIR, SOUND_FONT

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Определение устройства (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Используется устройство: {device}")


# Архитектура нейросети 
class MusicGenerationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Эмбеддинги
        self.pitch_embedding = torch.nn.Embedding(37, 128)
        self.duration_embedding = torch.nn.Embedding(4, 32)

        # Сверточные слои
        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv1d(160, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Conv1d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Conv1d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU()
        )

        # LSTM слои
        self.lstm = torch.nn.LSTM(256, 512, num_layers=2, batch_first=True)

        # Выходные слои
        self.fc_pitch = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 37)  # 37 классов высоты тона
        )

        self.fc_duration = torch.nn.Sequential(
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 4)  # 4 класса длительности
        )

    def forward(self, x):
        pitch, duration = x[..., 0], x[..., 1]
        pitch_emb = self.pitch_embedding(pitch)
        duration_emb = self.duration_embedding(duration)
        x = torch.cat([pitch_emb, duration_emb], dim=-1)

        # Конволюции
        x = x.permute(0, 2, 1)
        x = self.conv_block(x)
        x = x.permute(0, 2, 1)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Выходы
        pitch_out = self.fc_pitch(lstm_out)
        duration_out = self.fc_duration(lstm_out)

        return pitch_out, duration_out


# Загрузка модели
def load_model():
    model = MusicGenerationModel().to(device)
    model_path = "model_for_tg.pth"

    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device)

            # Адаптация ключей для совместимости
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace("module.", "").replace("_", ".")
                new_state_dict[new_key] = value

            model.load_state_dict(new_state_dict, strict=False)
            model.eval()
            logger.info("✅ Модель успешно загружена")
            return model
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            return None
    else:
        logger.error(f"Файл модели не найден: {model_path}")
        return None


# Загружаем модель при старте
model = load_model()


def generate_music(mood: str) -> tuple:
    """Генерация музыки с использованием нейросети"""
    try:
        logger.info(f"Начало генерации для настроения: {mood}")
        start_time = time.time()
        os.makedirs(TEMP_DIR, exist_ok=True)
        midi_path = os.path.join(TEMP_DIR, f"{mood}_{random.randint(1000, 9999)}.mid")
        audio_path = midi_path.replace('.mid', '.wav')

        if model is None:
            raise RuntimeError("Модель не загружена")

        # Начальная последовательность (случайная)
        input_seq = torch.zeros((1, 64, 2), dtype=torch.long).to(device)
        input_seq[..., 0] = torch.randint(0, 37, (1, 64))  # pitch
        input_seq[..., 1] = torch.randint(0, 4, (1, 64))  # duration

        # Авторегрессионная генерация
        generated_notes = []
        for i in range(64):
            with torch.no_grad():
                pitch_logits, duration_logits = model(input_seq)

            # Выбор следующих значений с учетом температуры
            temperature = 0.7
            pitch_probs = torch.softmax(pitch_logits[0, i] / temperature, dim=-1)
            next_pitch = torch.multinomial(pitch_probs, 1).item()

            duration_probs = torch.softmax(duration_logits[0, i] / temperature, dim=-1)
            next_duration = torch.multinomial(duration_probs, 1).item()

            # Сохранение сгенерированной ноты
            generated_notes.append({
                "pitch": next_pitch + 60,  # C4
                "duration": [0.25, 0.5, 1.0, 2.0][next_duration],
                "start": i * 0.5
            })

            # Обновление входной последовательности
            if i < 63:
                input_seq[0, i + 1, 0] = next_pitch
                input_seq[0, i + 1, 1] = next_duration

        # Применение настроения
        processed_notes = apply_mood_rules(generated_notes, mood)

        # Создание MIDI
        midi = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0)

        # Установка темпа по настроению
        tempo_map = {
            "sad": 76, "calm": 92,
            "happy": 142, "energetic": 160
        }
        time_scale = 60.0 / tempo_map[mood]

        for note in processed_notes:
            start_time = note["start"] * time_scale
            duration = note["duration"] * time_scale

            midi_note = pretty_midi.Note(
                velocity=random.randint(70, 100),
                pitch=note["pitch"],
                start=start_time,
                end=start_time + duration
            )
            piano.notes.append(midi_note)

        midi.instruments.append(piano)
        midi.write(midi_path)

        # Конвертация в WAV
        try:
            if os.path.exists(SOUND_FONT):
                audio_data = midi.fluidsynth(fs=44100, sf2_path=SOUND_FONT)
                import soundfile as sf
                sf.write(audio_path, audio_data, 44100)
                return audio_path, midi_path
            else:
                logger.warning(f"SoundFont не найден, возвращаем только MIDI: {SOUND_FONT}")
                return None, midi_path
        except Exception as e:
            logger.error(f"Ошибка конвертации в WAV: {e}")
            return None, midi_path

    except Exception as e:
        logger.exception(f"Ошибка генерации: {e}")
        return None, None


def apply_mood_rules(notes, mood):
    """Применение правил настроения к сгенерированным нотам"""
    new_notes = []

    for note in notes:
        new_note = note.copy()

        if mood == "energetic":
            new_note['duration'] = min(new_note['duration'], 0.25)
            if random.random() > 0.7:
                new_note['start'] += 0.05

        elif mood == "sad":
            new_note['duration'] *= 1.5
            if random.random() > 0.6:
                new_note['pitch'] = max(48, new_note['pitch'] - 3)

        elif mood == "happy":
            if random.random() > 0.6:
                new_note['pitch'] = min(84, new_note['pitch'] + 5)
            if new_note['duration'] > 0.25:
                new_note['duration'] = 0.25

        elif mood == "calm":
            new_note['duration'] *= 1.8
            if len(new_notes) > 0:
                prev_pitch = new_notes[-1]['pitch']
                if abs(new_note['pitch'] - prev_pitch) > 8:
                    new_note['pitch'] = (new_note['pitch'] + prev_pitch) // 2

        new_notes.append(new_note)

    return new_notes
