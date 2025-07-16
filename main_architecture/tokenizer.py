import numpy as np
import pretty_midi
from config import Config

class MIDITokenizer:
    """
    Токенизатор для преобразования MIDI в последовательность токенов
    по схеме Compound Word Representation.
    
    Особенности:
    - Поддержка нот, пауз, длительностей и эмоциональных меток
    - Квантование длительностей
    - Обработка музыкальной структуры (такты, позиции)
    """
    def __init__(self):
        # Инициализация словаря токенов
        self.special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
        self.bar_token = 'Bar'
        self.position_tokens = [f'Pos_{i}' for i in range(Config.positions_per_bar)]
        self.pitch_tokens = [f'Pitch_{i}' for i in range(21, 109)]  # MIDI ноты
        self.duration_tokens = [f'Dur_{d}' for d in [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4]]
        self.rest_tokens = [f'Rest_{d}' for d in [0.25, 0.5, 1, 2]]
        self.tempo_token = 'Tempo'
        self.emotion_tokens = [f'Emo_{i}' for i in range(Config.emotion_classes)]

        # Построение словаря
        self.vocab = (
            self.special_tokens +
            [self.bar_token] +
            self.position_tokens +
            self.pitch_tokens +
            self.duration_tokens +
            self.rest_tokens +
            [self.tempo_token] +
            self.emotion_tokens
        )[:Config.vocab_size]
        
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}
        self.duration_to_token = {d: f'Dur_{d}' for d in [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4]}
        self.rest_to_token = {d: f'Rest_{d}' for d in [0.25, 0.5, 1, 2]}

    def tokenize_midi(self, midi_path, emotion=None):
        """
        Преобразует MIDI-файл в последовательность токенов
        
        Аргументы:
            midi_path: путь к MIDI файлу
            emotion: метка эмоции (0-3)
            
        Возвращает:
            list: последовательность идентификаторов токенов
        """
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            tokens = []
            
            # Добавление эмоционального токена
            if emotion is not None and f'Emo_{emotion}' in self.token_to_id:
                tokens.append(f'Emo_{emotion}')
                
            tokens.append(self.tempo_token)

            # Извлечение и сортировка нот
            notes = []
            for instrument in midi_data.instruments:
                if not instrument.is_drum:
                    notes.extend(instrument.notes)
            notes.sort(key=lambda x: x.start)
            
            if not notes:
                return None

            # Определение музыкальной структуры
            time_signature = midi_data.time_signature_changes[0] if midi_data.time_signature_changes else None
            beats_per_bar = time_signature.numerator if time_signature else 4
            beat_times = midi_data.get_beats()
            
            if len(beat_times) < 2:
                return None
                
            bar_times = beat_times[::beats_per_bar]
            if not bar_times:
                return None

            # Построение токенов
            current_bar_idx = -1
            current_time = 0.0
            for note in notes:
                bar_idx = np.searchsorted(bar_times, note.start, side='right') - 1
                if bar_idx < 0:
                    continue
                    
                bar_time = bar_times[bar_idx]
                
                # Обработка пауз
                if note.start > current_time:
                    rest_duration = note.start - current_time
                    rest_in_beats = rest_duration / (beat_times[1] - beat_times[0])
                    quantized_rest = self.quantize_duration(rest_in_beats, is_rest=True)
                    if quantized_rest > 0:
                        tokens.append(f'Rest_{quantized_rest}')
                        current_time += quantized_rest * (beat_times[1] - beat_times[0])
                
                # Добавление токенов такта
                if bar_idx != current_bar_idx:
                    tokens.append(self.bar_token)
                    current_bar_idx = bar_idx
                
                # Токены позиции, высоты тона и длительности
                position_in_bar = (note.start - bar_time) / (beat_times[1] - beat_times[0])
                position_idx = min(Config.positions_per_bar-1, 
                                 int(position_in_bar * Config.positions_per_bar/beats_per_bar))
                tokens.append(f'Pos_{position_idx}')
                tokens.append(f'Pitch_{note.pitch}')
                
                duration = note.end - note.start
                duration_in_beats = duration / (beat_times[1] - beat_times[0])
                quantized_duration = self.quantize_duration(duration_in_beats)
                tokens.append(f'Dur_{quantized_duration}')
                
                current_time = note.end

            # Форматирование последовательности
            tokens = tokens[:Config.max_sequence_length-2]
            tokens = ['<sos>'] + tokens + ['<eos>']
            return [self.token_to_id.get(token, self.token_to_id['<unk>']) for token in tokens]
            
        except Exception as e:
            print(f"Error processing {midi_path}: {str(e)}")
            return None

    def quantize_duration(self, duration, is_rest=False):
        """
        Квантует длительность к ближайшему стандартному значению
        
        Аргументы:
            duration: исходная длительность в долях такта
            is_rest: флаг для пауз
            
        Возвращает:
            float: квантованное значение
        """
        durations = [0.25, 0.5, 1, 2] if is_rest else [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4]
        return min(durations, key=lambda x: abs(x - duration))
