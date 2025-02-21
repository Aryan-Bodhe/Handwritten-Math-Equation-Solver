# Handwritten Math Equation Solver

## Preprocessing Steps
1. Tilt Correction
2. Extract Text Area
3. Vertical Scanline
4. CCA per image
5. Feed to model for identification

## Model Specifications
1. Convolutional Neural Network
2. Number of Layers = 5
3. Saved at : https://iith-my.sharepoint.com/:f:/g/personal/ma23btech11005_iith_ac_in/EkbY9uZ8WiVIlp9Safw74TIBHDmZY5M232QHnupXCEe3sg?e=yqUleX

## TO-DO
- [x] Load the Handwritten_Characters dataset : \
      https://www.kaggle.com/datasets/vaibhao/handwritten-characters/data \
      https://www.kaggle.com/datasets/joshuawzy/basic-math-operation 
      
- [x] Train CNN on the data
- [x] Define rules for converting prediction to string (oper-var-number sequencing)
- [x] Finalise sympy file
- [x] Write code to input final segmented images from cache and predict the character
- [x] Web integration
- [ ] (Optional) Equation class expansion
- [ ] (Optional) Handwritten output in-place
