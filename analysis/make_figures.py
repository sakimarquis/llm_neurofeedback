import os
from PIL import Image
from configs.settings import SELECTED_LAYERS
from utils import load_exp_cfg


model = "llama3.1_8b"  # "llama3.1_8b" or "qwen2.5_7b"
cfg = load_exp_cfg(model)
layers = SELECTED_LAYERS[model]
dpi = 600
fig_format = "png"  # must be png, pdf is blurry as hell
pc_indices = [1, 2, 4, 8, 32, 128, 512]
input_dir = f"../results/{cfg.model_name.replace('/', '_')}/commonsense/imitation"
output_dir = f"{input_dir}/stitched"
os.makedirs(output_dir, exist_ok=True)
rows, cols = 2, 4  # grid dimensions for PC images
target_height = 512  # uniform height for each PC plot
spacing = -30  # overlap between images

for mode in ['active', 'inactive']:
    for layer in layers:
        # Templates for filename lookup
        pc_template = f"pcascore_{mode}_layer{layer}_score_diff_pc{{}}.{fig_format}"
        heatmap_name = f"pcascore_{mode}_layer{layer}_snr_heatmap.{fig_format}"
        output_filename = f"{model}_{mode}_layer{layer}.{fig_format}"
        output_path = os.path.join(output_dir, output_filename)

        # --- 1. Load & resize PC images into a flat list ---
        pc_images = []
        lr_img_path = os.path.join(input_dir, f"lr_{mode}_layer{layer}_score_diff.{fig_format}")
        lr_img = Image.open(lr_img_path)
        scale_lr = target_height / lr_img.height
        lr_new_w = int(lr_img.width * scale_lr)
        lr_img_resized = lr_img.resize((lr_new_w, target_height), Image.LANCZOS)
        pc_images.append(lr_img_resized)  # First image is always lr

        # Add all PC images next
        for pc in pc_indices:
            path = os.path.join(input_dir, pc_template.format(pc))
            img = Image.open(path)
            scale = target_height / img.height
            new_w = int(img.width * scale)
            pc_images.append(img.resize((new_w, target_height), Image.LANCZOS))

        # --- 2. Arrange PC images into rows ---
        grid_rows = [pc_images[i * cols:(i + 1) * cols] for i in range(rows)]
        row_widths = [
            sum(im.width for im in row) + spacing * (len(row) - 1)
            for row in grid_rows
        ]
        grid_width = max(row_widths)
        grid_height = rows * target_height

        # --- 3. Load & resize the 2×2 heatmap to span both rows ---
        heatmap_path = os.path.join(input_dir, heatmap_name)
        heatmap = Image.open(heatmap_path)
        scale_hm = grid_height / heatmap.height
        heatmap_new_w = int(heatmap.width * scale_hm)
        heatmap_resized = heatmap.resize((heatmap_new_w, grid_height), Image.LANCZOS)

        # --- 4. Create canvas big enough for grid + heatmap on the right ---
        canvas_w = grid_width + spacing + heatmap_new_w
        canvas_h = grid_height
        canvas = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 0))

        # --- 5. Paste PC grid ---
        for row_idx, row in enumerate(grid_rows):
            x_off = 0
            y_off = row_idx * target_height
            for im in row:
                canvas.paste(im, (x_off, y_off), im)
                x_off += im.width + spacing

        # --- 6. Paste heatmap at rightmost, spanning both rows ---
        x_off_hm, y_off_hm = grid_width + spacing, 0
        canvas.paste(heatmap_resized, (x_off_hm, y_off_hm), heatmap_resized)

        canvas.save(output_path, dpi=(dpi, dpi))
        print(f"Saved: {output_path}")
