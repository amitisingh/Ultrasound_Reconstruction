In order to run the code: 

1. Go to the Imagenet folder.
2. Install the required Python libraries (from requirements.txt).
3. Place all of the required models in step1_utils/models/ folder. For ImageNet, it will be: ImageNet_ADM.pt. 
4. Run the main project.py python file, along with the parameters that you would like. The following arguments are available: 

--dataset
Name of the dataset to use. This must match the dataset name used in the model configuration YAML file located in step1_utils/models/. For imagenet, it would be "ImageNet". 

--out_path
Directory where reconstructed output images will be saved. It will save to out_path/US/. 

--ps_type
Posterior sampling method to use.
Valid options: DPS or DDS.

--diff_timesteps
Total number of diffusion timesteps used during model training.

--desired_timesteps
Defines the number of denoising steps to be used during sampling. 

--eta
Represents the n parameter in the DDIM sampling algorithm.

--schedule
Specifies the sampling scheduled (uniform or non-uniform).

--H
Height of the image

--W
Width of the image

--channels
Number of channels in the image

--bh_mat
Path to the MATLAB .mat file containing the sparse BH operator.

--bh_group
Group name inside the HDF5 .mat file where the BH sparse matrix is stored.

--by_mat
Path to the MATLAB .mat file containing the ultrasound measurement vector By.

The following is an example command to run the pre-trained ImageNet model (which needs to be placed in the step1_utils/models/ folder), along with DPS as the posterior sampling method:


    !python project.py \
    --dataset ImageNet \
    --bh_mat //dir/of/data/BH.mat \
    --bh_group BH \
    --by_mat /dir/of/By.mat \
    --ps_type DPS \
    --diff_timesteps 1000 \
    --desired_timesteps 1000 \
    --schedule 1000 \
    --eta 1.0 \
    --H 256 \
    --W 256 \
    --channels 3
