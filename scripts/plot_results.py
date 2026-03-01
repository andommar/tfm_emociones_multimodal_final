import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_comparison():
    results = {
        'Arquitectura': ['Audio Unimodal\n(CNN)', 'Video Unimodal\n(ResNet+LSTM)', 'Fusión Multimodal\n(Propuesta)'],
        'Accuracy (%)': [45.83, 49.58, 72.08] # de momento se introduce los valores manualmente
    }
    df_results = pd.DataFrame(results)

    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(10, 7))

    colors = ['#bdc3c7', '#95a5a6', '#2ecc71'] 
    
    ax = sns.barplot(x='Arquitectura', y='Accuracy (%)', data=df_results, palette=colors)

    plt.title('Comparativa de Rendimiento en Conjunto de Test (RAVDESS)', fontsize=18, fontweight='bold', pad=20)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.xlabel('')  
    plt.ylim(0, 100)

    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 10), 
                    textcoords='offset points',
                    fontsize=16, fontweight='bold', color='#2c3e50')

    plt.axhline(y=12.5, color='#e74c3c', linestyle='--', linewidth=2, label='Umbral de probabilidad por azar (12.5%)')
    plt.legend(loc='upper left', frameon=True)

    plt.tight_layout()
    
    os.makedirs("reports", exist_ok=True)
    plt.savefig('reports/comparative_results.png', dpi=300)
    print("Gráfico exportado a 'reports/comparative_results.png'")

if __name__ == "__main__":
    plot_comparison()