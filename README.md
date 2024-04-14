# matrice_assignment

# mscoco_json_yolo_txt.ipynb

File converts given annotation in json format into YOLO trainable txt format.
by passing the json file and folder to store the txt file for the images script does the complete conversion.

# train_test_val_split.ipynb

File splits folder (image and annotation) content into train, test, and validation current split is 20% for test images and 10% for validation images.Script also creates folder containing train, test , valid data named custom dataset.

# Dockerfile

docker file with CUDA and OpenVINO support that can be used for training the model on GPU and running inference on Intel CPU.
