import os
import numpy as np
import pretty_midi
import matplotlib.pyplot as plt
from config import Config

def tokens_to_midi(token_ids, tokenizer, emotion_name):
    """
    Конвертирует последовательность токенов в MIDI файл
    
    Аргументы:
        token_ids: список идентификаторов токенов
        tokenizer: инстанс токенизатора
        emotion_name: название эмоции для имени файла
        
    Возвращает:
        pretty_midi.PrettyMIDI: сгенерированный MIDI объект
    """
    tokens = [tokenizer.id_to_token.get(idx, '<unk>') for idx in token_ids]
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
    
    current_bar = 0
    current_time = 0.0
    tempo = 120
    seconds_per_beat = 60.0 / tempo
    note_start = 0.0
    current_pitch = None
    
    for token in tokens:
        if token in ['<sos>', '<eos>', '<unk>']:
            continue
            
        if token == 'Bar':
            current_bar += 1
        elif token.startswith('Pos_'):
            position = int(token.split('_')[1])
            beats_in_bar = position * (4 / Config.positions_per_bar)
            note_start = (current_bar * 4 + beats_in_bar) * seconds_per_beat
        elif token == 'Tempo':
            tempo = np.random.randint(90, 151)
            seconds_per_beat = 60.0 / tempo
        elif token.startswith('Pitch_'):
            current_pitch = int(token.split('_')[1])
        elif token.startswith('Dur_'):
            if current_pitch is not None:
                duration = float(token.split('_')[1])
                note_end = note_start + duration * seconds_per_beat
                piano.notes.append(pretty_midi.Note(
                    velocity=100,
                    pitch=current_pitch,
                    start=note_start,
                    end=note_end
                ))
                current_pitch = None
        elif token.startswith('Rest_'):
            rest_duration = float(token.split('_')[1])
            note_start += rest_duration * seconds_per_beat
    
    midi.instruments.append(piano)
    output_path = os.path.join(Config.save_dir, f"generated_{emotion_name.replace('/', '_')}.mid")
    midi.write(output_path)
    return midi

def analyze_generated_music():
    """Анализ сгенерированных MIDI файлов"""
    emotion_map = {
        'HA_HV': 'HA/HV',
        'HA_LV': 'HA/LV',
        'LA_HV': 'LA/HV',
        'LA_LV': 'LA/LV'
    }
    stats = {name: {'count': 0, 'notes': 0, 'duration': 0.0, 'tempo': []} 
             for name in emotion_map.values()}
    
    # Сбор статистики
    for filename in os.listdir(Config.save_dir):
        if filename.startswith("generated_") and filename.endswith(".mid"):
            parts = filename.split('_')
            if len(parts) < 2:
                continue
                
            emotion_key = parts[1].replace('.mid', '')
            emotion_name = emotion_map.get(emotion_key)
            if not emotion_name:
                continue
                
            try:
                midi = pretty_midi.PrettyMIDI(os.path.join(Config.save_dir, filename))
                stats[emotion_name]['count'] += 1
                stats[emotion_name]['duration'] += midi.get_end_time()
                stats[emotion_name]['tempo'].append(midi.estimate_tempo())
                
                for instrument in midi.instruments:
                    stats[emotion_name]['notes'] += len(instrument.notes)
            except:
                continue
    
    # Визуализация
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    emotion_names = list(emotion_map.values())
    
    # Количество файлов
    axs[0, 0].bar(emotion_names, [stats[e]['count'] for e in emotion_names])
    axs[0, 0].set_title('Generated Files')
    
    # Среднее количество нот
    axs[0, 1].bar(emotion_names, 
                 [stats[e]['notes'] / max(1, stats[e]['count']) for e in emotion_names])
    axs[0, 1].set_title('Average Notes')
    
    # Средняя длительность
    axs[1, 0].bar(emotion_names, 
                 [stats[e]['duration'] / max(1, stats[e]['count']) for e in emotion_names])
    axs[1, 0].set_title('Average Duration (sec)')
    
    # Средний темп
    axs[1, 1].bar(emotion_names, 
                 [np.mean(stats[e]['tempo']) if stats[e]['tempo'] else 0 for e in emotion_names])
    axs[1, 1].set_title('Average Tempo (BPM)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.save_dir, "generation_analysis.png"))
    plt.close()
