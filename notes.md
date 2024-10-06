# Program flow
Run recognizer.py
    - It will iterate through all folders under images/
        - Iterate through all images under each folder and create a Path list
    - We will feed each image to gesture recognizer model
    - Add each result to dictionary (result, image)
    - Process above dictionary and feed into matplot lib to show image along with result.

# About labeling

pip install label-studio
label-studio start my_project --init

Another option:
https://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html

# About datasets

There are several datasets and resources available for human gesture and pose images, often used in research, computer vision, and machine learning projects. Below are some popular databases and sources that you can explore for human gesture images:

### 1. **MediaPipe Gesture Recognizer Dataset**
   - The [MediaPipe Gesture Recognizer](https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer) offers some basic gestures for use with their models. They also provide sample images for each gesture (like thumbs up, thumbs down, etc.). You can explore their documentation or GitHub repositories for accessing sample images.

### 2. **EgoHands Dataset**
   - [EgoHands](http://vision.soic.indiana.edu/projects/egohands/) is a dataset that contains 48 Google Glass videos of complex, first-person interactions between two people. It includes labeled hand segmentation and gesture annotations, making it useful for gesture recognition.

### 3. **HUMAN3.6M Dataset**
   - [Human3.6M](http://vision.imar.ro/human3.6m/) is a large-scale dataset of human activities, including gestures and poses captured from a variety of viewpoints. The dataset contains millions of 3D human poses and their corresponding images, which can be useful for gesture-based applications.

### 4. **Chalearn LAP IsoGD Dataset**
   - [ChaLearn LAP IsoGD](http://www.cbsr.ia.ac.cn/users/jwan/database/isogd.html) is a large gesture recognition dataset containing 47,933 RGB-D gesture clips and 249 gesture categories. This dataset is popular in gesture recognition competitions.

### 5. **MSR Gesture 3D Dataset**
   - The [MSR Gesture 3D Dataset](https://www.microsoft.com/en-us/research/project/msr-3d-gesture-dataset/) contains various gesture samples performed by multiple subjects. It includes skeletal data and depth information for use in gesture recognition and pose estimation tasks.

### 6. **MPII Human Pose Dataset**
   - The [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/) focuses on complex human poses and gestures in everyday activities. It provides both images and pose annotations, making it ideal for gesture analysis and recognition.

### 7. **UTD-MHAD Dataset**
   - [UTD-MHAD](http://www.utdallas.edu/~kehtar/UTD-MHAD.html) (UTD Multimodal Human Action Dataset) includes actions and gestures captured with a variety of sensors, including depth, RGB, and inertial data. It’s a good choice for multimodal gesture recognition tasks.

### 8. **Hand Gesture Recognition Database (HGR1 and HGR2)**
   - The [HGR1 and HGR2](https://archive.ics.uci.edu/ml/datasets/Motion+Capture+Hand+Postures) datasets from the UCI Machine Learning Repository contain various hand postures captured from 5 different users. They provide segmented images of hand postures that can be used for gesture recognition.

### 9. **Custom Datasets on Kaggle**
   - [Kaggle Datasets](https://www.kaggle.com/datasets) has several gesture-related datasets contributed by the community. You can search for “gesture recognition” or “human pose” to find a variety of datasets, including images, videos, and labeled gestures.

### 10. **Gesture Recognition Datasets on GitHub**
   - GitHub hosts various gesture recognition datasets that researchers and developers have shared. You can search for repositories with keywords like “gesture recognition dataset” or “human gesture images”.

### How to Access and Use These Datasets
- For most of these datasets, you’ll need to follow the dataset’s licensing and terms of use. Many are publicly available for research purposes and can be downloaded directly from their websites.
- Some may require permission or registration (like HUMAN3.6M and MSR), while others (like Kaggle datasets) may need you to sign up for access.

These resources should help you find a suitable dataset for human gestures. Let me know if you’d like more information on any specific dataset or if you need help with any of the above!