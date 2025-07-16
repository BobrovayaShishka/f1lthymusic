import numpy as np
from collections import Counter
from scipy.stats import entropy
import music21 as m21

def calculate_basic_metrics(notes):
    """Вычисление базовых метрик для последовательности нот"""
    pitches = []
    durations = []
    
    for note_str in notes:
        parts = note_str.split('_')
        if len(parts) < 2: 
            continue
        pitches.append(int(parts[0]))
        durations.append(float(parts[1]))
    
    if not pitches or not durations:
        return {
            'pitch_range': 0,
            'avg_duration': 0,
            'pitch_entropy': 0,
            'rhythm_complexity': 0
        }
    
    # Размах высот
    pitch_range = max(pitches) - min(pitches)
    
    # Средняя длительность
    avg_duration = np.mean(durations)
    
    # Энтропия высот (разнообразие)
    pitch_counts = Counter(pitches)
    total = sum(pitch_counts.values())
    pitch_probs = [count / total for count in pitch_counts.values()]
    pitch_entropy = -sum(p * np.log2(p) for p in pitch_probs if p > 0)
    
    # Ритмическая сложность (стандартное отклонение длительностей)
    rhythm_complexity = np.std(durations)
    
    return {
        'pitch_range': pitch_range,
        'avg_duration': avg_duration,
        'pitch_entropy': pitch_entropy,
        'rhythm_complexity': rhythm_complexity
    }

def calculate_harmony_score(notes, key='C'):
    """Оценка гармонической согласованности"""
    major_scale = [0, 2, 4, 5, 7, 9, 11]
    pitches = []
    
    for note_str in notes:
        parts = note_str.split('_')
        if len(parts) < 2: 
            continue
        pitch = int(parts[0])
        pitches.append(pitch % 12)
    
    if not pitches:
        return 0.0
    
    # Определяем тональность
    key_pitch = key.upper()
    if key_pitch in ['C', 'D', 'E', 'F', 'G', 'A', 'B']:
        base_pitch = {'C':0, 'D':2, 'E':4, 'F':5, 'G':7, 'A':9, 'B':11}[key_pitch]
        scale = [(base_pitch + interval) % 12 for interval in major_scale]
    else:
        scale = major_scale
    
    in_scale_count = sum(p in scale for p in pitches)
    return in_scale_count / len(pitches)

def calculate_rhythm_consistency(notes):
    """Оценка ритмической согласованности через автокорреляцию"""
    durations = []
    for note_str in notes:
        parts = note_str.split('_')
        if len(parts) < 2: 
            continue
        durations.append(float(parts[1]))
    
    if len(durations) < 2:
        return 0.0
    
    # Нормализация
    durations = np.array(durations)
    durations = (durations - np.mean(durations)) / (np.std(durations) + 1e-8)
    
    # Автокорреляция
    autocorr = np.correlate(durations, durations, mode='full')
    autocorr = autocorr[len(durations)-1:]
    autocorr /= autocorr[0]
    
    # Средняя автокорреляция для лагов 1-4
    return np.mean(autocorr[1:5]) if len(autocorr) > 4 else 0.0

def calculate_novelty(generated_notes, train_notes, n=4):
    """Вычисление новизны сгенерированной последовательности"""
    if not generated_notes or not train_notes:
        return 0.0
    
    # Создаем n-граммы
    gen_ngrams = set()
    train_ngrams = set()
    
    # Для тренировочных данных
    for i in range(len(train_notes) - n + 1):
        train_ngrams.add(tuple(train_notes[i:i+n]))
    
    # Для сгенерированных данных
    for i in range(len(generated_notes) - n + 1):
        gen_ngrams.add(tuple(generated_notes[i:i+n]))
    
    # Вычисляем процент уникальных n-грамм
    novel_ngrams = gen_ngrams - train_ngrams
    return len(novel_ngrams) / len(gen_ngrams) if gen_ngrams else 0.0

def calculate_kl_divergence(train_notes, generated_notes):
    """KL-дивергенция распределений высот нот"""
    train_pitches = []
    gen_pitches = []
    
    for note_str in train_notes:
        parts = note_str.split('_')
        if len(parts) >= 2:
            train_pitches.append(int(parts[0]))
    
    for note_str in generated_notes:
        parts = note_str.split('_')
        if len(parts) >= 2:
            gen_pitches.append(int(parts[0]))
    
    if not train_pitches or not gen_pitches:
        return float('inf')
    
    # Создаем гистограммы
    bins = np.linspace(0, 127, 128)
    train_hist, _ = np.histogram(train_pitches, bins=bins, density=True)
    gen_hist, _ = np.histogram(gen_pitches, bins=bins, density=True)
    
    # Добавляем эпсилон
    train_hist += 1e-10
    gen_hist += 1e-10
    
    # Вычисляем KL-дивергенцию
    return entropy(train_hist, gen_hist)

def evaluate_music(generated_notes, train_notes):
    """Комплексная оценка сгенерированной музыки"""
    if not generated_notes:
        return {}
    
    metrics = calculate_basic_metrics(generated_notes)
    metrics['harmony_score'] = calculate_harmony_score(generated_notes)
    metrics['rhythm_consistency'] = calculate_rhythm_consistency(generated_notes)
    metrics['novelty'] = calculate_novelty(generated_notes, train_notes)
    metrics['kl_divergence'] = calculate_kl_divergence(train_notes, generated_notes)
    
    # Составная оценка
    metrics['overall_score'] = (
        metrics['pitch_entropy'] * 0.2 +
        metrics['harmony_score'] * 0.3 +
        metrics['rhythm_consistency'] * 0.2 +
        metrics['novelty'] * 0.2 +
        (1 / (1 + metrics['kl_divergence'])) * 0.1
    )
    
    return metrics
