import torch
from config import Config
from data_preparation import prepare_emopia_dataset
from tokenizer import MIDITokenizer
from dataset import EMOPIADataset
from model import EmotionConditionalTransformer
from utils import create_data_loaders, train_model, evaluate_model, plot_training_history
from generate import tokens_to_midi, analyze_generated_music

def main():
    # Инициализация
    Config.setup_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Подготовка данных
    if not os.path.exists(Config.metadata_path):
        if not prepare_emopia_dataset():
            return
    
    # Токенизация
    tokenizer = MIDITokenizer()
    dataset = EMOPIADataset(tokenizer)
    if len(dataset) == 0:
        print("Данные не загружены")
        return
    
    # Создание DataLoader'ов
    dataloaders = create_data_loaders(dataset, tokenizer)
    
    # Инициализация модели
    model = EmotionConditionalTransformer(
        vocab_size=len(tokenizer.vocab),
        emotion_vocab_size=Config.emotion_classes
    ).to(device)
    
    # Обучение
    history = train_model(model, dataloaders, device)
    plot_training_history(history)
    
    # Загрузка лучшей модели
    best_model_path = os.path.join(Config.save_dir, f"{Config.model_name}_best.pt")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    # Оценка
    test_loss, test_acc = evaluate_model(model, dataloaders['test'], device, tokenizer)
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
    
    # Генерация музыки
    emotion_names = ['HA/HV', 'HA/LV', 'LA/HV', 'LA/LV']
    for emotion_idx, emotion_name in enumerate(emotion_names):
        print(f"Генерирование {emotion_name}...")
        token_ids = model.generate(emotion_idx, tokenizer, device)
        tokens_to_midi(token_ids, tokenizer, emotion_name)
    
    # Анализ результатов
    analyze_generated_music()
    print("Все операции успешно завершены!")

if __name__ == "__main__":
    main()
