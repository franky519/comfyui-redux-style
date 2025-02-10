import torch
import numpy as np
from PIL import Image, ImageDraw

class StyleModelGridVisualizer:
    """
    在输入图像上绘制 27x27 的网格线，用于可视化 Style Model 的 patches 划分。
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "line_width": ("INT", {"default": 1, "min": 1, "max": 5, "step": 1}),
            }
        }

    CATEGORY = "style_model"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("grid_image",)

    FUNCTION = "draw_grid"

    def _draw_grid_on_image(self, image_tensor, line_width):
        """在图像上绘制网格线"""
        # 将 tensor 转换为 PIL Image
        image_np = image_tensor.squeeze(0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)
        
        # 创建可绘制对象
        draw = ImageDraw.Draw(pil_image)
        
        # 获取图像尺寸
        width, height = pil_image.size
        
        # 计算每个网格的大小
        cell_width = width / 27
        cell_height = height / 27
        
        # 使用红色作为网格线颜色
        color = (255, 0, 0)
        
        # 绘制垂直线
        for i in range(1, 27):
            x = i * cell_width
            draw.line([(x, 0), (x, height)], fill=color, width=line_width)
        
        # 绘制水平线
        for i in range(1, 27):
            y = i * cell_height
            draw.line([(0, y), (width, y)], fill=color, width=line_width)
        
        # 将 PIL Image 转回 tensor
        image_np = np.array(pil_image) / 255.0
        return torch.from_numpy(image_np).unsqueeze(0)

    def draw_grid(self, image, line_width):
        """
        在输入图像上绘制网格。
        
        参数:
            image (torch.Tensor): 输入图像 tensor，形状为 [B, H, W, C]
            line_width (int): 网格线宽度
            
        返回:
            torch.Tensor: 带有网格线的图像
        """
        results = []
        batch_size = image.shape[0]
        
        for b in range(batch_size):
            one_image = image[b:b+1]  # 保持 4D tensor 形状
            grid_image = self._draw_grid_on_image(one_image, line_width)
            results.append(grid_image)
        
        return (torch.cat(results, dim=0),) 