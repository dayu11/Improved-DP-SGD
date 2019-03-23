# Improved-DP-SGD
Improved DP-SGD for differentially private learning. Codes for the experiment part of our paper "Improving the Gradient Perturbation Approach for Differentially Private Optimization" at NeurIPS 2018 ppml workshop https://ppml-workshop.github.io/ppml/.

Step1:

Install python3

Install pytorch at https://pytorch.org/ 

Step2:

Navigate to this repository's directory and install required packages.

pip install -r requirements.txt

Step3:

Run the code, for example:

python main.py --dataset mnist --epsilon=2.0 --delta=1e-5 --momentum=0.6 --SGN=2

runs our mnist experiment on convex objective using momentum and varying clip bound.
