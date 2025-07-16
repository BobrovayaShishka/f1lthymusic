import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from config import Config

def create_data_loaders(dataset, tokenizer):
    """
    Создает DataLoader'ы для тренировки, валидации и тестирования
    
    Возвращает:
        dict: словарь с DataLoader'ами
    """
    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=Config.test_ratio,
        random_state=42
    )
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=Config.validation_ratio/(1-Config.test_ratio),
        random_state=42
    )
    
    return {
        'train': DataLoader(
            Subset(dataset, train_idx),
            batch_size=Config.batch_size,
            shuffle=True,
            num_workers=4
        ),
        'val': DataLoader(
            Subset(dataset, val_idx),
            batch_size=Config.batch_size,
            shuffle=False
        ),
        'test': DataLoader(
            Subset(dataset, test_idx),
            batch_size=Config.batch_size,
            shuffle=False
        )
    }

def train_model(model, dataloaders, device):
    """
    Процедура обучения модели с ранней остановкой
    
    Аргументы:
        model: модель для обучения
        dataloaders: словарь с DataLoader'ами
        device: устройство для вычислений
        
    Возвращает:
        dict: история обучения (loss, accuracy)
    """
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.learning_rate,
        weight_decay=Config.weight_decay
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(Config.epochs):
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            for batch in tqdm(dataloaders[phase], desc=f"Epoch {epoch+1}/{Config.epochs} [{phase}]"):
                # Подготовка данных
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                emotion = batch['emotion'].to(device)
                
                # Сдвиг таргетов
                targets = input_ids[:, 1:].contiguous()
                outputs = model(
                    input_ids=input_ids[:, :-1],
                    attention_mask=attention_mask[:, :-1],
                    emotion_labels=emotion
                )
                
                # Расчет потерь
                loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                
                # Оптимизация на фазе обучения
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                # Статистика
                running_loss += loss.item() * input_ids.size(0)
                _, preds = torch.max(outputs, dim=-1)
                valid_mask = (targets != 0)
                running_corrects += torch.sum((preds == targets) * valid_mask)
                total_samples += torch.sum(valid_mask).item()
            
            # Агрегация метрик
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / total_samples
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())
            
            # Обработка валидационной фазы
            if phase == 'val':
                scheduler.step(epoch_loss)
                
                # Ранняя остановка
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    epochs_without_improvement = 0
                    torch.save(model.state_dict(), os.path.join(Config.save_dir, f"{Config.model_name}_best.pt"))
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= Config.early_stopping_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        return history
                        
        # Сохранение модели после эпохи
        torch.save(model.state_dict(), 
                  os.path.join(Config.save_dir, f"{Config.model_name}_epoch_{epoch+1}.pt"))
    
    return history

def evaluate_model(model, test_loader, device, tokenizer):
    """
    Оценка модели на тестовом наборе
    
    Возвращает:
        tuple: (test_loss, test_accuracy)
    """
    model.eval()
    all_targets = []
    all_preds = []
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # Вычисление предсказаний
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            emotion = batch['emotion'].to(device)
            targets = input_ids[:, 1:].contiguous()
            
            outputs = model(
                input_ids=input_ids[:, :-1],
                attention_mask=attention_mask[:, :-1],
                emotion_labels=emotion
            )
            
            # Расчет потерь
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            test_loss += loss.item() * input_ids.size(0)
            
            # Сбор предсказаний
            _, preds = torch.max(outputs, dim=-1)
            valid_mask = targets != 0
            all_preds.extend(preds[valid_mask].cpu().numpy())
            all_targets.extend(targets[valid_mask].cpu().numpy())
    
    # Расчет метрик
    test_loss /= len(test_loader.dataset)
    accuracy = np.sum(np.array(all_preds) == np.array(all_targets)) / len(all_targets)
    
    # Визуализация матрицы ошибок
    common_tokens = np.unique(all_targets)
    if len(common_tokens) > 100:
        token_counts = np.bincount(all_targets)
        common_tokens = np.argsort(token_counts)[-100:]
    
    cm = confusion_matrix(
        [t for t in all_targets if t in common_tokens],
        [p for p, t in zip(all_preds, all_targets) if t in common_tokens],
        labels=common_tokens
    )
    
    plt.figure(figsize=(20, 20))
    ConfusionMatrixDisplay(cm, display_labels=common_tokens).plot(cmap='viridis', values_format='d')
    plt.title("Confusion Matrix (Top 100 Tokens)")
    plt.savefig(os.path.join(Config.save_dir, "confusion_matrix.png"))
    plt.close()
    
    return test_loss, accuracy

def plot_training_history(history):
    """Визуализация истории обучения"""
    plt.figure(figsize=(12, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.save_dir, "training_history.png"))
    plt.close()
