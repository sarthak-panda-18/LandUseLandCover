"""
Convert efficientnetb0_patch.keras to ONNX and verify parity with 200 real patches.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Load Keras model
import keras
import tensorflow as tf
import tf2onnx
import onnx
import onnxruntime as ort

def convert_and_verify():
    project_root = Path(r"E:\ML Claude\lulc-project")
    keras_path = project_root / "backend" / "ml" / "models_patch" / "efficientnetb0_patch.keras"
    onnx_path = project_root / "backend" / "ml" / "models_patch" / "efficientnetb0_patch.onnx"
    tile_path = project_root / "dataset" / "LULCzip" / "LULC" / "input" / "Vijayawada_LULC_Input_2025-0000000000-0000000000.tif"

    print(f"Loading Keras model: {keras_path}...")
    model = keras.models.load_model(keras_path, compile=False)
    print("Model summary:")
    model.summary()

    # Determine input signature
    input_shape = model.input_shape  # e.g. (None, 224, 224, 3) or (None, 64, 64, 3)
    print(f"Model input shape: {input_shape}")

    # Convert to ONNX via tf2onnx
    input_signature = [tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32, name="input_1")]
    print("\nConverting model to ONNX using tf2onnx...")
    onnx_model, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=17,
        output_path=str(onnx_path)
    )
    print(f"ONNX model saved to {onnx_path} (Size: {os.path.getsize(onnx_path)/(1024*1024):.2f} MB)")

    # Test Parity with 200 real patches from Tile 1
    import rasterio
    with rasterio.open(tile_path) as src:
        raw_raster = src.read()[:5].astype(np.float32)  # (5, H, W)
    
    # Transpose to (H, W, 5)
    raster_hwc = np.transpose(raw_raster, (1, 2, 0))
    
    patches_224 = []
    patch_size = 64
    count = 0
    for r in range(0, raster_hwc.shape[0] - patch_size + 1, patch_size):
        for c in range(0, raster_hwc.shape[1] - patch_size + 1, patch_size):
            patch = raster_hwc[r:r+patch_size, c:c+patch_size, :]
            if not ((patch == 0).all() or np.isnan(patch).all() or (patch == -9999).all()):
                # Extract RGB (B4=red, B3=green, B2=blue) and scale by 255.0
                red = np.nan_to_num(patch[..., 2], nan=0.0).astype(np.float32)
                green = np.nan_to_num(patch[..., 1], nan=0.0).astype(np.float32)
                blue = np.nan_to_num(patch[..., 0], nan=0.0).astype(np.float32)
                rgb_scaled = np.stack([red, green, blue], axis=-1) * 255.0  # (64, 64, 3)
                patches_224.append(rgb_scaled)
                count += 1
                if count >= 200:
                    break
        if count >= 200:
            break

    print(f"\nExtracted {len(patches_224)} real patches for parity verification.")
    raw_patches = np.array(patches_224, dtype=np.float32)
    # Resize to (200, 224, 224, 3)
    resized_patches = tf.image.resize(raw_patches, (224, 224), method="bilinear").numpy()

    # 1. Keras Predictions
    print("Running Keras inference on 200 patches...")
    keras_probs = model.predict(resized_patches, batch_size=32, verbose=0)
    keras_classes = np.argmax(keras_probs, axis=-1)

    # 2. ONNX Runtime Predictions
    print("Running ONNX Runtime inference on 200 patches...")
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), session_options, providers=["CPUExecutionProvider"])
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"ONNX Input Name: {input_name}, Output Name: {output_name}")

    onnx_probs = session.run([output_name], {input_name: resized_patches})[0]
    onnx_classes = np.argmax(onnx_probs, axis=-1)

    # 3. Compare Parity
    max_abs_diff = float(np.max(np.abs(keras_probs - onnx_probs)))
    mean_abs_diff = float(np.mean(np.abs(keras_probs - onnx_probs)))
    matching_classes = int(np.sum(keras_classes == onnx_classes))
    parity_pct = (matching_classes / len(keras_classes)) * 100.0

    print("\n" + "=" * 70)
    print("PARITY VERIFICATION RESULTS (Keras vs ONNX Runtime)")
    print("=" * 70)
    print(f"Total Patches Evaluated : {len(keras_classes)}")
    print(f"Max Absolute Prob Diff  : {max_abs_diff:.6e}")
    print(f"Mean Absolute Prob Diff : {mean_abs_diff:.6e}")
    print(f"Predicted Class Matches : {matching_classes} / {len(keras_classes)} ({parity_pct:.2f}%)")
    print("=" * 70)

    if parity_pct >= 99.0:
        print("SUCCESS: Parity requirement (>= 99% match) SATISFIED!")
    else:
        print(f"FAILURE: Parity {parity_pct:.2f}% < 99.0%")
        sys.exit(1)

if __name__ == "__main__":
    convert_and_verify()
