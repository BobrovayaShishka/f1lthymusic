import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pretty_midi
import matplotlib.pyplot as plt
import os
import time
import tarfile
from sklearn.model_selection import train_test_split
from IPython.display import Audio, display
import subprocess
from tqdm import tqdm
from models import GRUModel, DeepGRUModel, LSTMModel, TransformerModel, CNNLSTMModel

# Конфигурация
SEQ_LENGTH = 32
HIDDEN_SIZE = 128
NUM_LAYERS = 1
BATCH_SIZE = 64
EPOCHS = 30
TEMPO = 140
LEARNING_RATE = 0.001
EMBED_DIM = 64
MODELS = ['GRU', 'DeepGRU', 'LSTM', 'Transformer', 'CNN_LSTM']

# Установка зависимостей
def install_dependencies():
    print("Установка зависимостей...")
    subprocess.run(['sudo', 'apt-get', 'install', '-y', 'fluidsynth'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(['pip', 'install', 'midi2audio', 'tqdm'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.exists('game_boy_soundfont.sf2'):
        subprocess.run(['wget', '-q', 'https://files.phat.zone/game_boy_soundfont.sf2'])
    print("Все зависимости установлены")

# Распаковка датасета
def extract_nes_midi_dataset(max_files=None):
    archive_path = 'nesmdb_midi.tar.gz'
    if not os.path.exists(archive_path):
        raise FileNotFoundError("Загрузите архив nesmdb_midi.tar.gz в Colab")
    
    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall()
    
    midi_paths = []
    for root, _, files in os.walk('nesmdb_midi'):
        for file in files:
            if file.endswith('.mid'):
                midi_paths.append(os.path.join(root, file))
                if max_files and len(midi_paths) >= max_files:
                    return midi_paths
    return midi_paths

# Обработка MIDI
def quantize_duration(duration):
    durations = [0.125, 0.25, 0.5, 1.0]
    return min(durations, key=lambda x: abs(x - duration))

def load_midi_data(file_paths):
    notes = []
    for path in tqdm(file_paths, desc="Обработка MIDI"):
        try:
            midi = pretty_midi.PrettyMIDI(path)
            for instrument in midi.instruments:
                if not instrument.is_drum:
                    for note in instrument.notes:
                        if 48 <= note.pitch <= 84:
                            duration = quantize_duration(note.end - note.start)
                            notes.append(f"{note.pitch}_{duration}")
        except:
            continue
    return notes

def create_dataset(notes, seq_length=32):
    unique_notes = sorted(set(notes))
    note_to_int = {note: i for i, note in enumerate(unique_notes)}
    int_to_note = {i: note for note, i in note_to_int.items()}
    
    sequences = []
    targets = []
    
    for i in range(len(notes) - seq_length):
        sequences.append([note_to_int[note] for note in notes[i:i+seq_length]])
        targets.append(note_to_int[notes[i+seq_length]])
    
    return np.array(sequences), np.array(targets), note_to_int, int_to_note, unique_notes

# Генерация музыки
def generate_music(model, start_seq, note_to_int, int_to_note, length=100, temperature=0.7):
    model.eval()
    generated = list(start_seq)
    hidden = None
    
    for _ in range(length):
        input_seq = torch.tensor([generated[-SEQ_LENGTH:]]).long()
        
        with torch.no_grad():
            if isinstance(model, TransformerModel):
                output, _ = model(input_seq)
            else:
                output, hidden = model(input_seq, hidden)
                
            logits = output / temperature
            probs = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()
            
        generated.append(next_idx)
    
    return [int_to_note[idx] for idx in generated]

# Преобразование в MIDI и WAV
def notes_to_midi(notes, output_path="music.mid"):
    midi = pretty_midi.PrettyMIDI(initial_tempo=TEMPO)
    square_wave = pretty_midi.Instrument(program=80)
    
    time = 0.0
    for note_str in notes:
        parts = note_str.split('_')
        if len(parts) < 2: continue
        pitch, duration = int(parts[0]), float(parts[1])
        
        note = pretty_midi.Note(
            velocity=100,
            pitch=pitch,
            start=time,
            end=time + duration
        )
        square_wave.notes.append(note)
        time += duration
    
    midi.instruments.append(square_wave)
    midi.write(output_path)
    return output_path

def convert_to_8bit_sound(midi_path):
    from midi2audio import FluidSynth
    fs = FluidSynth('game_boy_soundfont.sf2')
    wav_path = midi_path.replace('.mid', '.wav')
    fs.midi_to_audio(midi_path, wav_path)
    return wav_path

# Обучение модели
def train_model(model, train_loader, val_loader, optimizer, device, epochs=EPOCHS):
    criterion = nn.CrossEntropyLoss()
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        for batch_seq, batch_target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_seq, batch_target = batch_seq.to(device), batch_target.to(device)
            
            optimizer.zero_grad()
            output, _ = model(batch_seq)
            loss = criterion(output, batch_target)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_seq, batch_target in val_loader:
                batch_seq, batch_target = batch_seq.to(device), batch_target.to(device)
                output, _ = model(batch_seq)
                loss = criterion(output, batch_target)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"best_{model.__class__.__name__}.pth")
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
    
    return train_losses, val_losses

# Основной блок
if __name__ == "__main__":
    install_dependencies()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")
    
    midi_paths = extract_nes_midi_dataset(max_files=500)
    print(f"Найдено {len(midi_paths)} MIDI-файлов")
    
    notes = load_midi_data(midi_paths)
    print(f"Всего нот: {len(notes)}")
    
    sequences, targets, note_to_int, int_to_note, unique_notes = create_dataset(notes, SEQ_LENGTH)
    vocab_size = len(note_to_int)
    print(f"Размер словаря: {vocab_size}")
    
    X_train, X_val, y_train, y_val = train_test_split(sequences, targets, test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    results = {}
    
    for model_name in MODELS:
        print(f"\n=== Обучение модели: {model_name} ===")
        
        if model_name == 'GRU':
            model = GRUModel(vocab_size, EMBED_DIM, HIDDEN_SIZE).to(device)
        elif model_name == 'DeepGRU':
            model = DeepGRUModel(vocab_size, EMBED_DIM, HIDDEN_SIZE).to(device)
        elif model_name == 'LSTM':
            model = LSTMModel(vocab_size, EMBED_DIM, HIDDEN_SIZE).to(device)
        elif model_name == 'Transformer':
            model = TransformerModel(vocab_size, EMBED_DIM).to(device)
        elif model_name == 'CNN_LSTM':
            model = CNNLSTMModel(vocab_size, EMBED_DIM, HIDDEN_SIZE).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        start_time = time.time()
        train_losses, val_losses = train_model(model, train_loader, val_loader, optimizer, device)
        train_time = time.time() - start_time
        
        model.load_state_dict(torch.load(f"best_{model.__class__.__name__}.pth"))
        start_idx = np.random.randint(0, len(sequences)-1)
        start_seq = sequences[start_idx].tolist()
        
        gen_start = time.time()
        generated_notes = generate_music(
            model.cpu(), 
            start_seq, 
            note_to_int, 
            int_to_note,
            length=100,
            temperature=0.7
        )
        gen_time = time.time() - gen_start
        
        midi_file = f"generated_{model_name}.mid"
        notes_to_midi(generated_notes, midi_file)
        wav_file = convert_to_8bit_sound(midi_file)
        
        results[model_name] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_time': train_time,
            'gen_time': gen_time,
            'wav_file': wav_file,
            'notes': generated_notes[:20]
        }
    
    # Визуализация результатов
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    for model_name in MODELS:
        plt.plot(results[model_name]['val_losses'], label=f"{model_name} (val)")
    plt.title('Сравнение потерь на валидации')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    train_times = [results[m]['train_time'] for m in MODELS]
    gen_times = [results[m]['gen_time'] for m in MODELS]
    
    x = np.arange(len(MODELS))
    width = 0.35
    
    plt.bar(x - width/2, train_times, width, label='Обучение')
    plt.bar(x + width/2, gen_times, width, label='Генерация')
    plt.title('Время выполнения')
    plt.xlabel('Модель')
    plt.ylabel('Секунды')
    plt.xticks(x, MODELS)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()
    
    print("\nСравнение моделей:")
    for model_name in MODELS:
        print(f"\n{model_name}:")
        print(f"Время обучения: {results[model_name]['train_time']:.1f} сек")
        print(f"Время генерации: {results[model_name]['gen_time']:.2f} сек")
        print(f"Первые 10 нот: {results[model_name]['notes'][:10]}")
        print(f"Аудио: {results[model_name]['wav_file']}")
        display(Audio(results[model_name]['wav_file']))
    
    subprocess.run(['zip', '-r', 'all_results.zip'] + [f"generated_{m}.mid" for m in MODELS] + 
                   [f"generated_{m}.wav" for m in MODELS] + ['model_comparison.png'])
    
    from google.colab import files
    files.download('all_results.zip')
    print("Все результаты доступны для скачивания")
