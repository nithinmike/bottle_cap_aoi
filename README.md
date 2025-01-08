# bottle_cap_aoi

autoencoder_with_resnet_deep_features.pth - model that has been trained to reconstruct good bottle caps using resnet extracted features.

inference_batch_run_all.py - will run inference on all the files in the test dataset (good and defective) and generate an ideal pass/fail threshold based on the test data

inference_single_image.py - runs inference on a single image and generates a heatmap and anomaly score.

inference_visualize_small_batch.py - runs inference on all files of a specified defect type in the test dataset and produces heatmap and segmentation map for each.

train.py - cuda-optimized script to train the model using a training dataset. This script will only run on Nvidia GPUs. Train using a free cloud GPU service.

tools - scripts to capture and augment images

dataset - labeled test and training datasets

documentation - summary of hardware/software approach and some source files for figures and tables.