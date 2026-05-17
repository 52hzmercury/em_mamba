
# ... existing code ...
import torch
import torch.nn as nn
# ... existing code ...
from monai.networks.nets import SwinUNETR
import torch.nn.functional as F
# ... existing code ...

class SwinUnet(nn.Module):
    def __init__(self, img_size=(128, 128, 32), in_channels=1, num_classes=2, feature_size=48, use_checkpoint=False):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        # 使用 MONAI 的 3D SwinUNETR 实现
        self.swin_unet = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=num_classes,
            feature_size=feature_size,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=use_checkpoint,
        )

    def forward(self, x):
        logits = self.swin_unet(x)
        return logits

    # def forward(self, x):
    #     orig_d = x.size(4)
    #     required_d = getattr(self.swin_unet, "img_size", (0, 0, 32))[2]
    #     pad_left = pad_right = 0
    #     if orig_d != required_d:
    #         pad_total = required_d - orig_d
    #         pad_left = pad_total // 2
    #         pad_right = pad_total - pad_left
    #         # pad 顺序为 (W_left, W_right, H_left, H_right, D_left, D_right)
    #         x = F.pad(x, (0, 0, 0, 0, pad_left, pad_right))
    #     logits = self.swin_unet(x)
    #     if orig_d != required_d:
    #         logits = logits[:, :, :, :, pad_left:pad_left + orig_d]
    #     return logits

    def load_from(self, config=None):
        # 可选：加载预训练权重（留空以保持接口一致）
        pass
# ... existing code ...

def main():
    import time
    try:
        from thop import profile
        from thop import clever_format
    except ImportError:
        print("Error: 'thop' is not installed. Please run: pip install thop")
        return

    # 1. Configuration
    INPUT_SIZE = (1, 3, 128, 128, 32)  # (Batch, Channels, H, W, D)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Initialize Model
    print("Initializing model...")
    model = SwinUnet(img_size=INPUT_SIZE[2:], in_channels=INPUT_SIZE[1], num_classes=1).to(device)
    model.eval()

    # 4. Create Dummy Input
    dummy_input = torch.randn(INPUT_SIZE).to(device)

    # 5. Calculate Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("-" * 30)
    print(f"Total Parameters: {total_params / 1e6:.2f} M")
    print(f"Trainable Parameters: {trainable_params / 1e6:.2f} M")

    # 6. Calculate FLOPs
    print("-" * 30)
    print("Calculating FLOPs... (this might take a moment)")
    try:
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
        flops_formatted, params_formatted = clever_format([flops, total_params], "%.3f")
        print(f"GFLOPs: {flops / 1e9:.3f} G")
    except Exception as e:
        print(f"Error calculating FLOPs: {e}")

    # 7. Calculate FPS
    print("-" * 30)
    print("Calculating FPS...")
    num_iterations = 50
    warmup = 10

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    fps = num_iterations / total_time

    print(f"Average time per inference: {total_time / num_iterations:.4f} seconds")
    print(f"FPS: {fps:.2f}")
    print("-" * 30)

if __name__ == '__main__':
    main()
# ... existing code ...



