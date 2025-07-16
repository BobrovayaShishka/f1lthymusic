import matplotlib.pyplot as plt
import seaborn as sns

def visualize_metrics(results):
    """Визуализация метрик для всех моделей"""
    metrics_names = [
        'pitch_entropy', 'harmony_score', 'rhythm_consistency', 
        'novelty', 'kl_divergence', 'overall_score'
    ]
    model_names = list(results.keys())
    
    # Создаем grid для графиков
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Рисуем метрики для каждой модели
    for i, metric in enumerate(metrics_names):
        ax = axes[i]
        values = [results[model][metric] for model in model_names]
        
        # Для KL-дивергенции инвертируем значения
        if metric == 'kl_divergence':
            values = [-v for v in values]
            metric_name = "Inverse KL Divergence"
        else:
            metric_name = metric.replace('_', ' ').title()
        
        bars = ax.bar(model_names, values, color=sns.color_palette("viridis", len(model_names)))
        ax.set_title(metric_name)
        ax.set_ylabel("Score")
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Добавляем значения на столбцы
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png')
    plt.show()
    
    # Тепловая карта
    plt.figure(figsize=(12, 8))
    metrics_data = []
    display_names = [
        'Pitch Entropy', 'Harmony Score', 'Rhythm Consistency',
        'Novelty', 'Inverse KL Div', 'Overall Score'
    ]
    
    for model in model_names:
        row = []
        for metric in metrics_names:
            value = results[model][metric]
            # Инвертируем KL для тепловой карты
            row.append(-value if metric == 'kl_divergence' else value)
        metrics_data.append(row)
    
    sns.heatmap(
        metrics_data, 
        annot=True, 
        fmt=".2f", 
        cmap="viridis",
        xticklabels=display_names,
        yticklabels=model_names
    )
    plt.title("Model Metrics Comparison")
    plt.savefig('metrics_heatmap.png')
    plt.show()
