# Learned, Uncertainty-driven Adaptive Acquisition
The official PyTorch implementation of the Learned, Uncertainty-driven Adaptive Acquisition for Photon-Efficient Multiphoton Microscopy paper.

//LOL CASSANDRA PLZ FIX THE RESOLUTION OF THIS IMAGE

<div align="center">
  <img src="./readme_graphics/uncertainty_gif.gif" width="50%" />
  <br/>
  <div align="left" width="50%">
    <figcaption display="table-caption" width="70%"><b> Results of single-image, three-image, and five-image denoising, showing the image prediction and predicted uncertainty. As the number of measurements increases, the predicted image more closely matches the ground truth, and the pixel-wise uncertainty decreases.</figcaption>
  </div>
</div>

# Setup: 
Clone this project using:

```
git clone https://github.com/cassandra-t-ye/Learned_Uncertainty_Quantification.git
```

Dependencies can be installed using

```
conda env create -f environment.yml
source activate learned_uncertainty
```

# Getting Started:

To get started, download the weights for our finetuned model and the test images here:
```
*insert google drive link
*insert google drive link
```
Put the finetuned model weights under the **Weights** folder and the test images under the **Visualization** folder
```
./Learned_Uncertainty_Quantification/Weights
./Learned_Uncertainty_Quantification/Visualization
```
Once finished, open **quickstart.ipynb** and get started!


