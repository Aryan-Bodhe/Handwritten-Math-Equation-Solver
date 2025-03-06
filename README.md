# Handwritten Math Equation Solver
Contributors - Aryan Bodhe, Naishadha Voruganti, Sonit Patil.

Due to some policy issues the website could not be hosted. However, here's a video link demonstrating the usage of our model. \
Link - https://drive.google.com/file/d/1hsknvQblPA3icaaADbs0Tr0KJa_AJHyN/view?usp=sharing

## Preprocessing Steps
1. Tilt Correction
2. Extract Text Area
3. CCA per image
4. Feed to model for identification

## Model Specifications
1. Convolutional Neural Network
2. Number of Layers = 5

## TO-DO
- [x] Load the Handwritten_Characters dataset : \
      https://www.kaggle.com/datasets/vaibhao/handwritten-characters/data \
      https://www.kaggle.com/datasets/joshuawzy/basic-math-operation 
      
- [x] Train CNN on the data
- [x] Define rules for converting prediction to string (oper-var-number sequencing)
- [x] Finalise sympy file
- [x] Write code to input final segmented images from cache and predict the character
- [x] Extend dataset to include operators (+, -, /, =, (), {}, [])
- [x] Web integration
- [ ] (Optional) Equation class expansion
- [ ] (Optional) Handwritten output in-place
