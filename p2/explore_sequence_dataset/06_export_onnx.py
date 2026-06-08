"""
=============================================================================
Script 6: Export PyTorch Model to Monolithic ONNX Format
=============================================================================
Purpose:
    Converts the trained PyTorch Temporal CNN (.pth) into a single, static 
    Open Neural Network Exchange (.onnx) file. 
    
    *Architectural Update:* Previous dynamic axis exports caused PyTorch to 
    split the model weights into a phantom '.data' file, crashing the mobile 
    browser's memory allocation. This script forces a monolithic export 
    by locking the input tensor shape to exactly (1 Batch, 63 Features, 
    30 Frames) and utilizing Opset Version 14 for optimal WebAssembly (WASM) 
    compatibility.
    
Inputs:
    - 'temporal_cnn_raw.pth': The trained PyTorch weights.
      
Outputs:
    - 'temporal_cnn_monolithic.onnx': The 100% self-contained model ready 
      for mobile Edge NPU inference.
=============================================================================
"""

import torch
import torch.nn as nn
import os

# ==========================================
# 1. Recreate the Exact Architecture
# ==========================================
class TemporalCNN(nn.Module):
    def __init__(self, num_features=63, num_classes=2, window_size=30):
        super(TemporalCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        flattened_length = window_size // 4
        self.fc_input_size = 64 * flattened_length
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fc_input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.conv_block(x)
        return self.classifier(features)

# ==========================================
# 2. Monolithic Export Process
# ==========================================
if __name__ == "__main__":
    MODEL_PATH = "temporal_cnn_raw.pth"
    ONNX_PATH = "temporal_cnn_monolithic.onnx" # NEW NAME
    
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Could not find {MODEL_PATH}")
        exit()

    print("Loading PyTorch model...")
    model = TemporalCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    
    # Set the model to evaluation mode (crucial for Dropout/BatchNorm layers)
    model.eval()

    # Create a perfectly static dummy input: 1 Batch, 63 Features, 30 Frames
    dummy_input = torch.randn(1, 63, 30, requires_grad=True)

    print(f"Exporting monolithic model to {ONNX_PATH}...")
    
    # Export the model
    torch.onnx.export(
        model,                       
        dummy_input,                 
        ONNX_PATH,                   
        export_params=True,          
        opset_version=14,            # UPGRADED for better Web support
        do_constant_folding=True,    
        input_names=['input'],       
        output_names=['output']
        # dynamic_axes has been REMOVED to force a single-file export
    )

    print("SUCCESS! Model is now 100% self-contained.")
