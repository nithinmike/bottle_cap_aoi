# bottle_cap_aoi

__autoencoder_with_resnet_deep_features.pth__ - model that has been trained to reconstruct good bottle caps using resnet extracted features.

__inference_batch_run_all.py__ - will run inference on all the files in the test dataset (good and defective) and generate an ideal pass/fail threshold based on the test data

__inference_single_image.py__ - runs inference on a single image and generates a heatmap and anomaly score.

__inference_visualize_small_batch.py__ - runs inference on all files of a specified defect type in the test dataset and produces heatmap and segmentation map for each.

__train.py__ - cuda-optimized script to train the model using a training dataset. This script will only run on Nvidia GPUs. Train using a free cloud GPU service.

__tools__ - scripts to capture and augment images

__dataset__ - labeled test and training datasets

__documentation__ - summary of hardware/software approach and some source files for figures and tables.

__reference code__ - https://github.com/mohan696matlab/mvtec_anomalydetection