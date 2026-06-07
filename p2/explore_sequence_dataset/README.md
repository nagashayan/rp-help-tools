This folder we are using our p1_dataset_combined 

I would explore these architectures:
1. Geometry rules (Done in workshop paper)
2. Temporal CNN
3. GRU
4. LSTM

Experiment A: I will continue from workshop paper, use palm orientation and  stability and calculate some deltas required by temporal models and feed to Temporal CNN to see if it can beat the metrics of workshop paper.

python 01_extract_raw_landmarks.py
python 02_engineer_baseline_features.py
python 03_train_temporal_cnn.py

Result: 
Loading data and generating sliding windows...
Total sequences generated: 3778
Training on 49 videos (2988 windows)
Testing on 13 videos (790 windows)

Starting Training Loop...
Epoch [1/20] | Train Loss: 0.6884 - Acc: 71.42% | Val Loss: 0.6186 - Acc: 66.96%
Epoch [5/20] | Train Loss: 0.4472 - Acc: 74.80% | Val Loss: 0.8831 - Acc: 66.96%
Epoch [10/20] | Train Loss: 0.4072 - Acc: 76.14% | Val Loss: 1.1613 - Acc: 66.96%
Epoch [15/20] | Train Loss: 0.3763 - Acc: 77.54% | Val Loss: 1.8347 - Acc: 65.57%
Epoch [20/20] | Train Loss: 0.3417 - Acc: 78.78% | Val Loss: 2.4657 - Acc: 67.59%

Since handshakes are ~67% it is always resulting in 67% guess, blind guess so next experiment we will introduce reach constraint along with palm and stability constraints

Experiment B: Experiment A + reach constraint
Total sequences generated: 3778
Training on 49 videos (2988 windows)
Testing on 13 videos (790 windows)

Starting Training Loop...
Epoch [1/20] | Train Loss: 0.5851 - Acc: 71.85% | Val Loss: 0.7314 - Acc: 66.20%
Epoch [5/20] | Train Loss: 0.4339 - Acc: 75.17% | Val Loss: 0.9662 - Acc: 65.82%
Epoch [10/20] | Train Loss: 0.3835 - Acc: 77.95% | Val Loss: 1.0704 - Acc: 67.34%
Epoch [15/20] | Train Loss: 0.3574 - Acc: 80.52% | Val Loss: 1.2013 - Acc: 67.97%
Epoch [20/20] | Train Loss: 0.3004 - Acc: 83.67% | Val Loss: 2.3564 - Acc: 64.94%

Training Complete! Model weights saved as 'temporal_cnn_baseline.pth'.
Result: That didn't help

Experiment C: Experiment B + class weights
Loading data and generating sliding windows...
Total sequences generated: 3778
Training on 49 videos (2988 windows)
Testing on 13 videos (790 windows)

Computed Class Weights: Class 0 (Random) = 1.75, Class 1 (Handshake) = 0.70

Starting Training Loop...
Epoch [1/25] | Train Loss: 0.6490 - Acc: 64.69% | Val Loss: 0.6038 - Acc: 71.90%
Epoch [5/25] | Train Loss: 0.4877 - Acc: 68.01% | Val Loss: 0.8366 - Acc: 72.03%
Epoch [10/25] | Train Loss: 0.4205 - Acc: 71.82% | Val Loss: 1.1793 - Acc: 72.15%
Epoch [15/25] | Train Loss: 0.3950 - Acc: 73.49% | Val Loss: 1.9040 - Acc: 71.14%
Epoch [20/25] | Train Loss: 0.3441 - Acc: 76.41% | Val Loss: 2.0991 - Acc: 69.24%
Epoch [25/25] | Train Loss: 0.3058 - Acc: 78.18% | Val Loss: 1.9830 - Acc: 71.39%

Training Complete! Model weights saved as 'temporal_cnn_baseline.pth'.
Result: Better than previous two experiments almost achieved workshop paper accuracy(75%)
Validation Loss still crept up (from 0.60 to 1.98). In deep learning, this is called Overconfident Overfitting. The model is learning the training data a bit too well, so when it does make a mistake on the test set, it makes it with 99% confidence, which causes the Cross-Entropy loss penalty to explode. For a small dataset baseline, this is completely expected and nothing to worry about.
