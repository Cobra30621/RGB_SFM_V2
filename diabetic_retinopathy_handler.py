import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt


def display_image_comparison(save_path=None, **images_dict):
    """
    顯示多張圖片的比較，使用變數名稱作為標題
    Args:
        save_path: 圖片保存路徑，如果提供則保存圖片
        **images_dict: 圖片變數，例如 original=img1, enhanced=img2 等
    """
    image_list = list(images_dict.values())
    title_list = list(images_dict.keys())
    num_images = len(image_list)

    fig, axes = plt.subplots(1, num_images, figsize=(3 * num_images, 6))

    for idx in range(num_images):
        img = image_list[idx]

        # 將 PyTorch tensor 轉換為 numpy 並調整通道
        if torch.is_tensor(img):
            if img.shape[0] == 1:  # 灰階圖片
                img = img.squeeze(0).cpu().numpy()
            else:  # RGB 圖片
                img = img.permute(1, 2, 0).cpu().numpy()
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)

        # 使用 cmap='gray' 顯示灰階圖片
        if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
            axes[idx].imshow(img, cmap='gray')
        else:
            axes[idx].imshow(img)
            
        axes[idx].set_title(title_list[idx])
        axes[idx].axis('off')

    plt.tight_layout()
    
    # 如果提供保存路徑，則保存圖片
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()





def preprocess_retinal_tensor_image(image_tensor, target_radius=224, final_size=(225, 225)):
    """
    Retina 圖像的預處理，保留為 PyTorch tensor 格式 (CxHxW)

    Args:
        image_tensor: 輸入圖像 tensor (CxHxW)
        target_radius: 縮放後圖像的半徑基準大小
        final_size: 最終輸出的圖片大小 (H_final, W_final)

    Returns:
        Tuple of processed tensors: (scaled, blurred, contrast_enhanced, final_output)
    """
    if not isinstance(image_tensor, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor")

    device = image_tensor.device

    # CxHxW → HxWxC → numpy array
    img_np = image_tensor.permute(1, 2, 0).detach().cpu().numpy()
    if img_np.dtype != np.uint8:
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

    # Step 1: 根據亮度估計半徑並縮放
    row_brightness = img_np[img_np.shape[0] // 2, :, :].sum(axis=1)
    estimated_radius = (row_brightness > row_brightness.mean() / 10).sum() / 2
    scale_factor = target_radius / estimated_radius
    scaled_img = cv2.resize(img_np, (0, 0), fx=scale_factor, fy=scale_factor)

    # Step 2: 高斯模糊
    blurred_img = cv2.GaussianBlur(scaled_img, (0, 0), target_radius / 30)

    # Step 3: 增強對比
    contrast_img = cv2.addWeighted(scaled_img, 4, blurred_img, -4, 128)

    # Step 4: 遮罩處理（中心圓形區域保留）
    mask = np.zeros_like(scaled_img, dtype=np.uint8)
    center = (scaled_img.shape[1] // 2, scaled_img.shape[0] // 2)
    radius = int(target_radius * 0.9)
    cv2.circle(mask, center, radius, (1, 1, 1), thickness=-1)
    final_img = contrast_img * mask + 128 * (1 - mask)

    # Step 5: 重新縮放至指定大小
    H_final, W_final = final_size
    final_img_resized = cv2.resize(final_img, (W_final, H_final))

    def to_tensor(np_img):
        tensor = torch.tensor(np_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return tensor.to(device)

    return (
        to_tensor(scaled_img),
        to_tensor(blurred_img),
        to_tensor(contrast_img),
        to_tensor(final_img_resized)
    )

def preprocess_retinal_tensor_batch(image_tensors, target_radius=224, final_size=(255, 255)):
    """
    Retina 圖像的預處理，適用於 batch 處理，保留為 PyTorch tensor 格式 (B, C, H, W)

    Args:
        image_tensors: 輸入圖像 batch tensor (B, C, H, W)
        target_radius: 縮放後圖像的半徑基準大小
        final_size: 最終輸出的圖片大小 (H_final, W_final)

    Returns:
        Processed final output tensor (B, C, H, W)
    """
    if not isinstance(image_tensors, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor")

    device = image_tensors.device
    batch_size = image_tensors.shape[0]

    final_images = []

    for i in range(batch_size):
        img_tensor = image_tensors[i]
        img_np = img_tensor.permute(1, 2, 0).detach().cpu().numpy()
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

        # Step 1: 根據亮度估計半徑並縮放
        row_brightness = img_np[img_np.shape[0] // 2, :, :].sum(axis=1)
        estimated_radius = (row_brightness > row_brightness.mean() / 10).sum() / 2
        scale_factor = target_radius / estimated_radius
        scaled_img = cv2.resize(img_np, (0, 0), fx=scale_factor, fy=scale_factor)

        # Step 2: 高斯模糊
        blurred_img = cv2.GaussianBlur(scaled_img, (0, 0), target_radius / 30)

        # Step 3: 增強對比
        contrast_img = cv2.addWeighted(scaled_img, 4, blurred_img, -4, 128)

        # Step 4: 遮罩處理（中心圓形區域保留）
        mask = np.zeros_like(scaled_img, dtype=np.uint8)
        center = (scaled_img.shape[1] // 2, scaled_img.shape[0] // 2)
        radius = int(target_radius * 0.9)
        cv2.circle(mask, center, radius, (1, 1, 1), thickness=-1)
        final_img = contrast_img * mask + 128 * (1 - mask)

        # Step 5: 重新縮放至指定大小
        H_final, W_final = final_size
        final_img_resized = cv2.resize(final_img, (W_final, H_final))

        def to_tensor(np_img):
            tensor = torch.tensor(np_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
            return tensor.to(device)

        final_images.append(to_tensor(final_img_resized))

    return torch.stack(final_images)


