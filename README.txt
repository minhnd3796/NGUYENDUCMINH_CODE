Student name: Nguyen Duc Minh
Class: CN-CNTT2 K59
Student ID: 20146488

Bachelor Final Project: Environment Setup and Tutorials

1. Environment Setup:
  - Python 3 Interpreter.
  - Python 3 libraries: OpenCV3, Tensorflow 1.4
  - Ubuntu 16.04.
2. Tutorials:
  - Run demos:
    + $ python3 FCN-4s-ResNet101-6c-v3.3.py 0 (train FCN-4s-ResNet101-6c-v3.3 model)
    + $ python3 FCN-4s-ResNet101-5c-v3.3.py 0 (train FCN-4s-ResNet101-5c-v3.3 model)
    + $ python3 FCN-8s-ResNet101-5c-v3.2.py 0 (train FCN-8s-ResNet101-5c-v3.2 model)
    + $ python3 FCN-8s-ResNet101-5c-v3.1.py 0 (train FCN-8s-ResNet101-5c-v3.1 model)
    + $ python3 FCN-8s-ResNet101-5c-v3.0.py 0 (train FCN-8s-ResNet101-5c-v3.0 model)
  - Run with real datasets:
    + Request and download datasets from ISPRS 2D Labeling Contest Site.
    + Extract necessary data for each dataset (ISPRS_semantic_labeling_Vaihingen or
    ISPRS_semantic_labeling_Potsdam directory).
    + Vaihingen: $ python3 sampling_image_5channels.py
    + Potsdam: $ get_npy_Potsdam.py && sampling_image_potsdam_v4.py
3. Customisations:
  - Per patch accuracy evaluation: batch_eval_*.py, next_batch* methods in Batch_manager*

For detailed support and further collaboration: PLEASE contact me:
    - Email: gordonnguyen3796@gmail.com
    - Facebook: fb.me/minhnd3796
    - Tel: +84 167 633 1696
