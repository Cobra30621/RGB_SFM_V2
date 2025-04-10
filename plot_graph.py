import numpy as np
from matplotlib import pyplot as plt


def plot_map(rm, grid_size=None, rowspan=None, colspan=None, path=None, **kwargs):
    """
    繪製濾波器圖像的網格視覺化。

    參數:
        rm (torch.Tensor 或 numpy.ndarray): 五維張量，shape = (num_filter_rows, num_filter_cols, filter_height, filter_width, num_channels)

            - num_filter_rows: 濾波器排列的行數（可視為 filter map 的排版高度）
            - num_filter_cols: 濾波器排列的列數（可視為 filter map 的排版寬度）
            - filter_height:   每個濾波器圖像的實際高度（像素）
            - filter_width:    每個濾波器圖像的實際寬度（像素）
            - num_channels:    通道數（1 表示灰階，3 表示 RGB 彩色）

        grid_size (tuple): matplotlib subplot 的總 grid 尺寸。
        rowspan (int): 單個圖佔用的 grid row 數。
        colspan (int): 單個圖佔用的 grid col 數。
        path (str): 若指定路徑，則儲存圖片至該路徑；否則顯示圖片。
        **kwargs: 傳給 imshow() 的額外參數，如 cmap。

    回傳:
        fig: matplotlib 的 figure 物件。
    """

    # 新增：計算 rm 的全局最大值和最小值
    global_max = rm.max()
    global_min = rm.min()

    rows, cols, e_h, e_w, _ = rm.shape
    rowspan = rowspan or max(e_h // min(e_h, e_w), 1)
    colspan = colspan or max(e_w // min(e_h, e_w), 1)
    grid_size = grid_size or (rows * rowspan, cols * colspan)

    char_space = cols * 0.4
    fig = plt.figure(figsize=(grid_size[1] + char_space, grid_size[0]), facecolor="white")

    # 繪製圖像
    for row in range(rows):
        for col in range(cols):
            ax = plt.subplot2grid(grid_size, (row * rowspan, col * colspan), rowspan=rowspan, colspan=colspan)
            im = ax.imshow(rm[row][col], vmin=global_min, vmax=global_max, **kwargs)  # 使用全局最大值和最小值
            ax.axis('off')

    # 新增：調整 subplot 位置，讓每個 subplot 靠左
    plt.subplots_adjust(left=0, right=0.7, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1)  # 調整邊距

    # 添加色彩對應數值的長條圖，與整張圖片等高
    cbar_ax = fig.add_axes([0.75, 0.1, 0.02, 0.8])  # 自定義長條圖的位置和大小
    cbar = plt.colorbar(im, cax=cbar_ax)  # 使用自定義的長條圖軸
    cbar.ax.tick_params(labelsize=2 * rows)  # 調整長條圖文字大小

    if path:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close(fig)

    return fig  # 回傳繪製的圖像


def plot_combine_images(figs, save_path=None, spacing=0.05, fixed_width=5, fixed_height=5, show=False, title="Combined Images"):
    num_images = len(figs)
    fig_width = num_images * fixed_width + (num_images - 1) * spacing
    fig_height = fixed_height + 2

    # 創建畫布，啟用 constrained_layout
    fig, axes = plt.subplots(
        1, num_images,
        figsize=(fig_width, fig_height),
        gridspec_kw={'width_ratios': [1] * num_images, 'wspace': spacing / fixed_width},
        constrained_layout=True
    )

    # 確保 axes 是列表
    if num_images == 1:
        axes = [axes]

    for i, ((key, fig_source), ax) in enumerate(zip(figs.items(), axes)):
        # 將 `fig_source` 等比例縮小 90%
        fig_source.set_size_inches(fig_source.get_size_inches() * 0.8)
        ax.imshow(fig_source.canvas.buffer_rgba())
        ax.axis('off')

        ax.set_title(key, fontsize=20, pad=10)

    # 新增整張大圖片的標題
    fig.suptitle(title, fontsize=24, y=0.96)

    if save_path:
        plt.savefig(save_path, dpi=300)
        if not show:
            plt.close(fig)
        else:
            plt.show()
    else:
        plt.show()

    return fig


def plot_combine_images_vertical(figs, save_path=None, spacing=0.05, fixed_width=5, fixed_height=1, show=False):
    num_images = len(figs)
    fig_width = fixed_width
    fig_height = num_images * fixed_height + (num_images - 1) * spacing
    print(f"fig_width: {fig_width}, {fig_height}")

    # 創建畫布，啟用 constrained_layout
    fig, axes = plt.subplots(
        num_images, 1,
        figsize=(fig_width, fig_height),
        gridspec_kw={'height_ratios': [1] * num_images, 'hspace': spacing / fixed_height},
        constrained_layout=True
    )

    # 確保 axes 是列表
    if num_images == 1:
        axes = [axes]

    for i, ((key, fig_source), ax) in enumerate(zip(figs.items(), axes)):
        # 將 `fig_source` 等比例縮小 80%
        fig_source.set_size_inches(fig_source.get_size_inches() * 0.8)
        ax.imshow(fig_source.canvas.buffer_rgba())
        ax.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=150)
        if not show:
            plt.close(fig)
        else:
            plt.show()
    else:
        plt.show()

    return fig


def plot_heatmap(CI_values, save_path, width=15, height=15):
    """
    Generate and save a heatmap from the given values.

    Parameters:
    CI_values (list of list of float): The matrix values for the heatmap.
    save_path (str): The path to save the heatmap image.
    width (int): The width of the heatmap.
    height (int): The height of the heatmap.
    """
    # Convert the input list to a NumPy array and reshape to the specified width and height
    reshaped_CI_values = np.array(CI_values).reshape(height, width)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(9, 9))  # Enlarge by 1.5 times

    # Display the heatmap
    cax = ax.matshow(reshaped_CI_values, cmap='viridis')

    # Add color bar
    fig.colorbar(cax)

    # Set up ticks and labels
    ax.set_xticks(np.arange(width))
    ax.set_yticks(np.arange(height))
    ax.set_xticklabels([''] * width)
    ax.set_yticklabels([''] * height)

    # Render the values in the matrix
    for i in range(height):
        for j in range(width):
            ax.text(j, i, f"{reshaped_CI_values[i][j]:.2f}", va='center', ha='center', color='white', fontsize=7)

    # Save the figure
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    return fig
