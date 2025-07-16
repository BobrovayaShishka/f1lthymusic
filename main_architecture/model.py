import math
import torch
import torch.nn as nn
from config import Config

class PositionalEncoding(nn.Module):
    """
    Позиционное кодирование для Transformer
    Добавляет информацию о позиции токенов с помощью синусоид
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EmotionConditionalTransformer(nn.Module):
    """
    Transformer с эмоциональным кондиционированием для генерации музыки
    
    Особенности:
    - Встраивание эмоциональных меток в скрытое пространство
    - Механизм внимания с несколькими головами
    - Генерация музыки с контролем длительности
    """
    def __init__(self, vocab_size, emotion_vocab_size):
        super().__init__()
        self.d_model = Config.d_model
        
        # Эмбеддинги
        self.token_embedding = nn.Embedding(vocab_size, Config.d_model)
        self.positional_encoding = PositionalEncoding(Config.d_model)
        self.dropout = nn.Dropout(Config.dropout)
        
        # Эмоциональное кондиционирование
        self.emotion_embedding = nn.Embedding(emotion_vocab_size, Config.d_model)
        self.emotion_projection = nn.Sequential(
            nn.Linear(Config.d_model, Config.d_model),
            nn.ReLU(),
            nn.Linear(Config.d_model, Config.d_model)
        )
        
        # Transformer Encoder
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=Config.d_model,
            nhead=Config.nhead,
            dim_feedforward=Config.dim_feedforward,
            dropout=Config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, Config.num_layers)
        
        # Нормализация и выходной слой
        self.layer_norm = nn.LayerNorm(Config.d_model)
        self.output_layer = nn.Linear(Config.d_model, vocab_size)
        
        self.init_weights()

    def init_weights(self):
        """Инициализация весов модели"""
        init_range = 0.1
        self.token_embedding.weight.data.uniform_(-init_range, init_range)
        self.emotion_embedding.weight.data.uniform_(-init_range, init_range)
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-init_range, init_range)
        
        for layer in self.emotion_projection:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.zero_()

    def forward(self, input_ids, attention_mask, emotion_labels):
        """
        Прямой проход модели
        
        Аргументы:
            input_ids: тензор с идентификаторами токенов
            attention_mask: маска внимания
            emotion_labels: метки эмоций
            
        Возвращает:
            logits: выходные логиты модели
        """
        # Эмбеддинг токенов
        token_emb = self.token_embedding(input_ids)
        
        # Эмоциональное кондиционирование
        emotion_emb = self.emotion_embedding(emotion_labels)
        emotion_emb = self.emotion_projection(emotion_emb).unsqueeze(1)
        emotion_emb_expanded = emotion_emb.expand(-1, token_emb.size(1), -1)
        
        # Комбинирование признаков
        x = token_emb + emotion_emb_expanded
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Transformer
        src_key_padding_mask = (attention_mask == 0)
        output = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        output = self.layer_norm(output)
        
        return self.output_layer(output)

    def generate(self, emotion_label, tokenizer, device, max_length=None, temperature=1.0):
        """
        Генерация музыки для заданной эмоции
        
        Аргументы:
            emotion_label: метка эмоции (0-3)
            tokenizer: инстанс токенизатора
            device: устройство для вычислений (CPU/GPU)
            
        Возвращает:
            numpy.ndarray: сгенерированные идентификаторы токенов
        """
        self.eval()
        max_length = max_length or Config.max_generation_length
        
        # Начальные токены
        sos_token = tokenizer.token_to_id['<sos>']
        eos_token = tokenizer.token_to_id['<eos>']
        input_ids = torch.tensor([[sos_token]], device=device)
        emotion_labels = torch.tensor([emotion_label], device=device)
        
        # Параметры генерации
        current_bar = 0
        current_time = 0.0
        tempo = 120  # начальный темп
        seconds_per_beat = 60.0 / tempo
        generated_duration = 0.0

        with torch.no_grad():
            for _ in range(max_length):
                # Предсказание следующего токена
                logits = self(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    emotion_labels=emotion_labels
                )[:, -1, :]
                
                # Применение температуры
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
                token_id = next_token.item()
                token_str = tokenizer.id_to_token.get(token_id, '<unk>')
                
                # Обработка музыкальных событий
                if token_str == 'Bar':
                    current_bar += 1
                elif token_str.startswith('Pos_'):
                    pass  # Позиция уже обрабатывается в нотных событиях
                elif token_str.startswith('Dur_'):
                    duration_val = float(token_str.split('_')[1])
                    generated_duration += duration_val * seconds_per_beat
                elif token_str.startswith('Rest_'):
                    rest_val = float(token_str.split('_')[1])
                    generated_duration += rest_val * seconds_per_beat
                elif token_str == 'Tempo':
                    tempo = np.random.randint(90, 151)
                    seconds_per_beat = 60.0 / tempo
                
                # Добавление токена
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Проверка завершения
                if generated_duration >= Config.target_duration or token_id == eos_token:
                    if token_id != eos_token:
                        input_ids = torch.cat([input_ids, torch.tensor([[eos_token]], device=device)], dim=1)
                    break

        return input_ids[0].cpu().numpy()
