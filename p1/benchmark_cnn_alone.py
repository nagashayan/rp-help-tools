import os
import glob
import cv2
import numpy as np
import time
import tensorflow as tf

print("==================================================")
print("🚀 BENCHMARKING PURE TEXTURE (CNN ONLY)")
print("==================================================")

# 1. Load Custom 1-Channel CNN
TFLITE_MODEL_PATH = "custom_handshake_cnn.tflite"
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 2. Configuration & Thresholds
BENCHMARK_DIR = "../images/p1_dataset_combined"
categories = ["none", "handshake"]
CNN_THRESHOLD = 0.60  # Using a stricter threshold to prevent guessing

# Tracking Variables
total_pipeline_time = 0.0
total_frames_processed = 0
tp, tn, fp, fn = 0, 0, 0, 0

# 3. Benchmark Loop
for category in categories:
    cat_path = os.path.join(BENCHMARK_DIR, category)
    if not os.path.exists(cat_path): continue
        
    print(f"\nEvaluating Category: {category.upper()}")
    
    for clip in os.listdir(cat_path):
        clip_path = os.path.join(cat_path, clip)
        if not os.path.isdir(clip_path): continue
            
        images = sorted(glob.glob(f"{clip_path}/*.jpg"))
        if not images: continue
            
        trigger_fired = False
        max_cnn_score_recorded = 0.0
        
        for img_path in images:
            frame = cv2.imread(img_path)
            if frame is None: continue
            
            # ⏱️ START STOPWATCH
            # We start it here to simulate live memory access, excluding disk I/O
            start_time = time.perf_counter()
            
            # STAGE 1: Pure Texture Preprocessing
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_resized = cv2.resize(frame_gray, (160, 160))
            
            # Format for TFLite: shape (1, 160, 160, 1)
            input_tensor = np.expand_dims(frame_resized, axis=-1)
            input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)
            
            # STAGE 2: Continuous Inference
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()
            cnn_score = interpreter.get_tensor(output_details[0]['index'])[0][0]
            
            # ⏱️ STOP STOPWATCH
            end_time = time.perf_counter()
            total_pipeline_time += (end_time - start_time)
            total_frames_processed += 1
            
            if cnn_score > max_cnn_score_recorded:
                max_cnn_score_recorded = cnn_score
                
            if cnn_score >= CNN_THRESHOLD:
                trigger_fired = True
                break # Success! Move to next clip

        # Confusion Matrix Tally
        if trigger_fired and category == "handshake":
            tp += 1
            print(f"[✅ CORRECT - TP] Clip: {clip} | Score: {max_cnn_score_recorded:.2f}")
        elif not trigger_fired and category == "none":
            tn += 1
            print(f"[✅ CORRECT - TN] Clip: {clip} | Score: {max_cnn_score_recorded:.2f}")
        elif trigger_fired and category == "none":
            fp += 1
            print(f"[❌ FAIL - FP] Clip: {clip} | Score: {max_cnn_score_recorded:.2f}")
        else:
            fn += 1
            print(f"[❌ FAIL - FN] Clip: {clip} | Score: {max_cnn_score_recorded:.2f}")

# ==========================================
# 📊 CALCULATE FINAL METRICS
# ==========================================
print("\n==================================================")
print("🏁 PURE CNN BENCHMARK COMPLETE")
print("==================================================")

total_clips = tp + tn + fp + fn
accuracy = (tp + tn) / total_clips if total_clips > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

print("--- 📈 STATISTICAL METRICS ---")
print(f"Total Clips Evaluated: {total_clips}")
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")

if total_frames_processed > 0:
    avg_latency_ms = (total_pipeline_time / total_frames_processed) * 1000
    est_fps = 1000 / avg_latency_ms if avg_latency_ms > 0 else 0
    print("\n--- ⚡ HARDWARE METRICS ---")
    print(f"Total Frames Processed : {total_frames_processed}")
    print(f"Avg Latency per Frame  : {avg_latency_ms:.2f} ms")
    print(f"Estimated Real-Time FPS: {est_fps:.1f} FPS")
print("==================================================")
