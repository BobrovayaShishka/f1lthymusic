import torch
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from models import GRUModel, DeepGRUModel, LSTMModel, TransformerModel, CNNLSTMModel
from metrics import evaluate_music
from visualization import visualize_metrics
import torch.optim as optim
import torch.nn as nn

# Конфигурация
SEQ_LENGTH = 32
HIDDEN_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 20
EMBED_DIM = 64
MODELS = ['GRU', 'DeepGRU', 'LSTM', 'Transformer', 'CNN_LSTM']

def generate_music(model, start_seq, note_to_int, int_to_note, length=100, temperature=0.7):
    """Функция генерации музыки (аналогичная graphics.py)"""
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

if __name__ == "__main__":
    # Инициализация
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")
    
    # Подготовка данных (используем функции из graphics.py)
    from graphics import extract_nes_midi_dataset, load_midi_data, create_dataset
    
    midi_paths = extract_nes_midi_dataset(max_files=200)
    print(f"Найдено {len(midi_paths)} MIDI-файлов")
    
    notes = load_midi_data(midi_paths)
    print(f"Всего нот: {len(notes)}")
    
    sequences, targets, note_to_int, int_to_note, unique_notes, all_notes = create_dataset(notes, SEQ_LENGTH)
    vocab_size = len(note_to_int)
    print(f"Размер словаря: {vocab_size}")
    
    # Разделение данных
    X_train, X_val, y_train, y_val = train_test_split(sequences, targets, test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Словарь для хранения результатов
    results = {}
    
    # Обучение и оценка моделей
    for model_name in MODELS:
        print(f"\n=== Обучение модели: {model_name} ===")
        
        # Инициализация модели
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
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Упрощенное обучение
        best_val_loss = float('inf')
        for epoch in range(EPOCHS):
            model.train()
            for batch_seq, batch_target in train_loader:
                batch_seq, batch_target = batch_seq.to(device), batch_target.to(device)
                optimizer.zero_grad()
                output, _ = model(batch_seq)
                loss = criterion(output, batch_target)
                loss.backward()
                optimizer.step()
            
            # Валидация
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_seq, batch_target in val_loader:
                    batch_seq, batch_target = batch_seq.to(device), batch_target.to(device)
                    output, _ = model(batch_seq)
                    loss = criterion(output, batch_target)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f"best_{model_name}.pth")
            
            print(f"Epoch {epoch+1}/{EPOCHS}: Val Loss={avg_val_loss:.4f}")
        
        # Генерация музыки
        model.load_state_dict(torch.load(f"best_{model_name}.pth"))
        start_idx = np.random.randint(0, len(sequences)-1)
        start_seq = sequences[start_idx].tolist()
        
        generated_notes = generate_music(
            model.cpu(), 
            start_seq, 
            note_to_int, 
            int_to_note,
            length=100,
            temperature=0.7
        )
        
        # Оценка сгенерированной музыки
        metrics = evaluate_music(generated_notes, all_notes)
        
        # Сохранение результатов
        results[model_name] = metrics
        print(f"\nМетрики для {model_name}:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
    
    # Визуализация результатов
    visualize_metrics(results)
    
    # Сравнение моделей по составной оценке
    print("\nИтоговое сравнение моделей:")
    sorted_models = sorted(results.items(), key=lambda x: x[1]['overall_score'], reverse=True)
    for model, metrics in sorted_models:
        print(f"{model}: Overall Score = {metrics['overall_score']:.4f}")
    
    # Скачивание результатов
    from google.colab import files
    files.download('metrics_comparison.png')
    files.download('metrics_heatmap.png')
    print("Графики сравнения доступны для скачивания")
