import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
MODEL_PATH = '/mnt/c/development/Thesis/PotatoDiseaseClassification-CNN/models/best_unet_v3.keras'
IMG_SIZE   = 256
THRESHOLD  = 0.5


def dice_coef(y_true, y_pred, smooth=1.0):
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    y_true_f = tf.cast(y_true, tf.float32)
    y_pred_f = tf.cast(y_pred, tf.float32)
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true_f, y_pred_f))
    return bce + dice_loss(y_true_f, y_pred_f)

def iou_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + 1.0) / (union + 1.0)


def load_model(model_path):
    print(f"🔄 Model লোড হচ্ছে: {model_path}")
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'bce_dice_loss': bce_dice_loss,
            'dice_coef':     dice_coef,
            'iou_metric':    iou_metric,
        }
    )
    print("✅ Model লোড সম্পন্ন!\n")
    return model

def predict_single(model, image_path, save_output=True, threshold=THRESHOLD):
    if not os.path.exists(image_path):
        print(f"❌ ফাইল পাওয়া যায়নি: {image_path}")
        return

    # ── Load & Preprocess ──────────────────
    img     = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_size = (img.shape[1], img.shape[0])  # W, H

    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    inp = np.expand_dims(img_resized / 255.0, axis=0).astype(np.float32)

    # ── Predict ────────────────────────────
    pred_mask = model.predict(inp, verbose=0)[0]          # (256, 256, 1)
    mask_bin  = (pred_mask > threshold).astype(np.uint8)  # binary

    # ── Polyp Coverage % ───────────────────
    coverage = mask_bin.sum() / mask_bin.size * 100

    # ── Overlay (রঙিন highlight) ───────────
    overlay      = img_resized.copy()
    green_mask   = np.zeros_like(img_resized)
    green_mask[:, :, 1] = mask_bin[:, :, 0] * 255  # সবুজ রঙ
    overlay = cv2.addWeighted(overlay, 0.7, green_mask, 0.3, 0)

    # ── Extracted Polyp ────────────────────
    mask_3d   = np.concatenate([mask_bin] * 3, axis=-1)
    extracted = (img_resized * mask_3d).astype(np.uint8)

    # ── Display ────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"📁 {os.path.basename(image_path)}  |  Polyp Coverage: {coverage:.2f}%",
                 fontsize=13, fontweight='bold')

    axes[0].imshow(img_resized);              axes[0].set_title('Original Image')
    axes[1].imshow(pred_mask.squeeze(), cmap='hot'); axes[1].set_title('Heatmap (Raw Prediction)')
    axes[2].imshow(overlay);                 axes[2].set_title('Overlay (Green = Polyp)')
    axes[3].imshow(extracted);               axes[3].set_title('Extracted Polyp Only')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()

    if save_output:
        OUTPUT_DIR = "/mnt/c/development/Thesis/PotatoDiseaseClassification-CNN/test/polyp-segmentation-results"
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        out_name = os.path.splitext(os.path.basename(image_path))[0] + '_result.png'
        # out_path = os.path.join(os.path.dirname(image_path), out_name)
        out_path = os.path.join(OUTPUT_DIR, out_name)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"💾 Result সেভ হয়েছে: {out_path}")

    plt.show()
    print(f"📊 Polyp Coverage: {coverage:.2f}%")
    print(f"🔍 Threshold: {threshold}")

# ─────────────────────────────────────────────
#  MULTIPLE IMAGE PREDICT (ফোল্ডার)
# ─────────────────────────────────────────────
def predict_folder(model, folder_path, save_output=True, threshold=THRESHOLD):
    exts  = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    files = []
    for ext in exts:
        import glob
        files += glob.glob(os.path.join(folder_path, ext))

    if not files:
        print(f"❌ কোনো ইমেজ পাওয়া যায়নি: {folder_path}")
        return

    print(f"📂 {len(files)}টি ইমেজ পাওয়া গেছে\n")
    for i, f in enumerate(sorted(files), 1):
        print(f"[{i}/{len(files)}] প্রসেস করছে: {os.path.basename(f)}")
        predict_single(model, f, save_output=save_output, threshold=threshold)

# ─────────────────────────────────────────────
#  INTERACTIVE MODE  (টার্মিনালে path দিন)
# ─────────────────────────────────────────────
def interactive_mode(model):
    print("\n" + "="*50)
    print("  U-Net Polyp Detection — Interactive Mode")
    print("  'q' টাইপ করলে বের হবে")
    print("="*50 + "\n")

    while True:
        path = input("📁 ইমেজ বা ফোল্ডারের path দিন: ").strip().strip('"')

        if path.lower() == 'q':
            print("👋 বের হচ্ছে...")
            break

        if os.path.isfile(path):
            predict_single(model, path)
        elif os.path.isdir(path):
            predict_folder(model, path)
        else:
            print(f"❌ Path সঠিক নয়: {path}\n")

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='U-Net Polyp Segmentation Inference')
    parser.add_argument('--image',   type=str, help='Single image path')
    parser.add_argument('--folder',  type=str, help='Folder of images')
    parser.add_argument('--model',   type=str, default=MODEL_PATH, help='Model path')
    parser.add_argument('--threshold', type=float, default=THRESHOLD, help='Prediction threshold (0-1)')
    parser.add_argument('--no-save', action='store_true', help='Result সেভ করবে না')
    args = parser.parse_args()

    # Model লোড
    model = load_model(args.model)
    save  = not args.no_save

    if args.image:
        # একটি ইমেজ
        predict_single(model, args.image, save_output=save, threshold=args.threshold)

    elif args.folder:
        # পুরো ফোল্ডার
        predict_folder(model, args.folder, save_output=save, threshold=args.threshold)

    else:
        # কিছু না দিলে interactive mode
        interactive_mode(model)


"""
python test_polyp.py --image "/mnt/c/development/Thesis/PotatoDiseaseClassification-CNN/DataSets/Kvasir-SEG/images/cju0qkwl35piu0993l0dewei2.jpg"
python test_polyp.py --folder "C:/path/to/folder"
"""