import torch

class StyleModelConditioner:
    """
    应用风格模型到条件信息的节点。
    
    该节点将 CLIP Vision 输出与条件信息结合，通过风格模型生成新的条件信息。
    可以通过 strength 参数控制风格的强度，支持 multiply 和 attn_bias 两种强度应用方式。
    支持通过 mask 指定图像中的有效区域。
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "style_model": ("STYLE_MODEL",),
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.001
                }),
                "strength_type": (["multiply", "attn_bias"],),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("styled_conditioning",)
    FUNCTION = "apply_style"
    CATEGORY = "style_model"

    def processMask(self, mask, img_size=384, patch_size=14):
        """
        处理mask，保持与原始图像的对齐关系。
        
        参数:
            mask (torch.Tensor): 输入mask
            img_size (int): 目标图像大小
            patch_size (int): patch的大小
            
        返回:
            torch.Tensor: 处理后的mask
        """
        if mask is None:
            return None
        
        # 标准化输入维度
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)  # 添加 batch 和 channel 维度
        elif len(mask.shape) == 3:
            mask = mask.unsqueeze(1)  # 添加 channel 维度
        
        # 直接调整到目标尺寸，保持对齐关系
        mask = torch.nn.functional.interpolate(
            mask,
            size=(img_size, img_size),
            mode="nearest"  # 使用最近邻插值以保持mask的二值性
        )
        
        # 转换为patches
        toks = img_size // patch_size
        mask = torch.nn.MaxPool2d(
            kernel_size=(patch_size, patch_size),
            stride=patch_size
        )(mask).view(-1, toks, toks, 1)
        
        return mask

    def _create_attention_mask(self, txt_shape, n_cond, mask_ref_size, attn_bias, device, existing_mask=None):
        """
        创建或更新注意力掩码。
        
        参数:
            txt_shape (tuple): 文本张量的形状
            n_cond (int): 条件向量的长度
            mask_ref_size (tuple): 掩码参考尺寸
            attn_bias (torch.Tensor): 注意力偏置值
            device: 计算设备
            existing_mask (torch.Tensor, optional): 现有的掩码
            
        返回:
            torch.Tensor: 更新后的注意力掩码
        """
        n_txt = txt_shape[1]
        n_ref = mask_ref_size[0] * mask_ref_size[1]
        
        if existing_mask is None:
            existing_mask = torch.zeros((txt_shape[0], n_txt + n_ref, n_txt + n_ref), dtype=torch.float16)
        elif existing_mask.dtype == torch.bool:
            existing_mask = torch.log(existing_mask.to(dtype=torch.float16))
            
        new_mask = torch.zeros((txt_shape[0], n_txt + n_cond + n_ref, n_txt + n_cond + n_ref), dtype=torch.float16)
        
        # 复制现有掩码的四个象限
        new_mask[:, :n_txt, :n_txt] = existing_mask[:, :n_txt, :n_txt]
        new_mask[:, :n_txt, n_txt+n_cond:] = existing_mask[:, :n_txt, n_txt:]
        new_mask[:, n_txt+n_cond:, :n_txt] = existing_mask[:, n_txt:, :n_txt]
        new_mask[:, n_txt+n_cond:, n_txt+n_cond:] = existing_mask[:, n_txt:, n_txt:]
        
        # 填充新的注意力偏置
        new_mask[:, :n_txt, n_txt:n_txt+n_cond] = attn_bias
        new_mask[:, n_txt+n_cond:, n_txt:n_txt+n_cond] = attn_bias
        
        return new_mask.to(device)

    def apply_style(self, conditioning, style_model, clip_vision_output, strength, strength_type, mask=None):
        """
        将风格模型应用到条件信息。
        
        参数:
            conditioning (list): 原始条件信息列表
            style_model: 风格模型
            clip_vision_output: CLIP vision 模型的输出
            strength (float): 风格强度
            strength_type (str): 强度应用方式，可选 'multiply' 或 'attn_bias'
            mask (torch.Tensor, optional): 图像mask，用于指定有效区域
            
        返回:
            tuple: 包含更新后条件信息的元组
        """
        # 获取风格条件并展平
        cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
        
        # 处理 mask
        if mask is not None:
            # 将 mask 转换为 patches
            patch_mask = self.processMask(mask)
            if patch_mask is not None:
                # 重塑 cond 以应用 mask
                b, t, h = cond.shape
                m = int(t ** 0.5)  # 计算特征图的边长
                cond = cond.view(b, m, m, h)
                # 应用 mask
                cond = cond * patch_mask
                # 重新展平
                cond = cond.reshape(b, t, h)
        
        if strength_type == "multiply":
            cond *= strength

        n_cond = cond.shape[1]
        styled_conditioning = []
        
        for txt, keys in conditioning:
            keys = keys.copy()
            
            # 处理注意力掩码
            if "attention_mask" in keys or (strength_type == "attn_bias" and strength != 1.0):
                attn_bias = torch.log(torch.Tensor([strength if strength_type == "attn_bias" else 1.0]))
                mask_ref_size = keys.get("attention_mask_img_shape", (1, 1))
                
                # 创建或更新注意力掩码
                new_mask = self._create_attention_mask(
                    txt.shape,
                    n_cond,
                    mask_ref_size,
                    attn_bias,
                    txt.device,
                    keys.get("attention_mask", None)
                )
                
                keys["attention_mask"] = new_mask
                keys["attention_mask_img_shape"] = mask_ref_size

            styled_conditioning.append([torch.cat((txt, cond), dim=1), keys])

        return (styled_conditioning,)