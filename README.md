# Ultrasound_Reconstruction

The main code is present in Ultrasound_reconstruction. This was built over HW4! 

To run Ultrasound Reconstruction and ImageNet:

  Please read the README.txt in each of the folders for instructions on how to run the code. Ensure that BH.mat, By.mat (By_2.mat) are downloaded and the path is known (you will need this path to run both). 

Drive link for BH, By, Model checkpoints: https://drive.google.com/drive/folders/15C9zi3s0SZHaBnCz-w4U0GCuAzbUSAdN?usp=sharing


To run DDRM:

  Please clone the following repository, along with the instructions provided in the repository.
  https://github.com/Yuxin-Zhang-Jasmine/DRUS-v1/tree/main

  Modify/replace the config file present in DRUS-v1/configs/imagenet_256.yml to change the following:
    name: "model006000.pt"
    problem_model: "DRUS"


and run the following command (as mentioned in the repo): 
  python main.py --ni --config imagenet_256.yml --doc imagenet --timesteps 50 --matlab_path ./MATLAB/ (Assuming the MATLAB folder is in the same folder as main.py). 
