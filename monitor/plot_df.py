import os

import pandas as pd

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pandas.plotting import table
import seaborn as sns

# 設置支持中文的字型
rcParams['font.family'] = 'SimSun'  # 微軟黑體
rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題

def plot_heatmap(data, output_path, title=None):
    """
    Plot a heatmap from a DataFrame and save as an image.

    Args:
        data (pd.DataFrame): The DataFrame to visualize.
        output_path (str): Path to save the heatmap image.
        title (str, optional): Title of the heatmap.
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        data,
        annot=True,  # 顯示數值
        fmt=".2f",   # 數值格式
        cmap="YlGnBu",  # 色彩樣式
        cbar_kws={'label': '數值範圍'}
    )

    # 添加標題（如果有）
    if title:
        plt.title(title, fontsize=16)

    # 保存圖片
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_table_as_image(data, output_path, title=None):
    """
    Save a DataFrame as an image.

    Args:
        data (pd.DataFrame): The DataFrame to save as an image.
        output_path (str): Path to save the output image.
        title (str, optional): Title to display above the table.
    """
    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')  # Hide axis

    # Optionally add a title
    if title:
        ax.set_title(title, fontdict={'fontsize': 14}, loc='center', pad=20)

    # Plot the table
    tbl = table(ax, data, loc='center', cellLoc='center', colWidths=[0.3] * len(data.columns))
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.2)


    # Save the figure as an image
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def json_to_table(layers_data, summary_data):
    """
    Combine layers_data and summary_data into a single DataFrame.

    Args:
        layers_data (dict): Layer data in JSON format.
        summary_data (dict): Summary data in JSON format.

    Returns:
        pd.DataFrame: Combined table with summary as the last row.
    """
    # Convert layers_data to DataFrame
    layers_df = pd.DataFrame(layers_data).T
    layers_df.index.name = 'Layer'

    # Convert summary_data to a single-row DataFrame
    summary_df = pd.DataFrame(summary_data, index=['Summary'])

    # Concatenate layers and summary
    combined_df = pd.concat([layers_df, summary_df])

    return combined_df