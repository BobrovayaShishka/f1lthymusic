import torch
import torch.nn as nn


class MusicGenerationModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Эмбеддинги
        self.pitch_embedding = nn.Embedding(37, 192)
        self.duration_embedding = nn.Embedding(4, 64)

        # Сверточный блок
        self.conv_block = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        # LSTM блок (4 слоя)
        self.lstm = nn.LSTM(512, 512, num_layers=4, batch_first=True)

        # Слой нормализации
        self.layer_norm = nn.LayerNorm(512)

        # Residual connection (линейный слой)
        self.residual = nn.Linear(256, 512) 

        # Выходные слои
        self.fc_pitch = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 37)
        )

        self.fc_duration = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
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

        # Нормализация и residual connection
        norm_out = self.layer_norm(lstm_out)

        # Измененный residual connection
        residual_input = norm_out[:, :, :256]  # Берем только первые 256 каналов
        res_out = self.residual(residual_input) + lstm_out

        # Выходы
        pitch_out = self.fc_pitch(res_out)
        duration_out = self.fc_duration(res_out)

        return pitch_out, duration_out
