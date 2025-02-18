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

## TO-DO
- [ ] Load the Handwritten_Characters dataset : \
      https://www.kaggle.com/datasets/vaibhao/handwritten-characters/data \
      https://www.kaggle.com/datasets/joshuawzy/basic-math-operation 
      
- [x] Train CNN on the data
- [ ] Define rules for converting prediction to string (oper-var-number sequencing)
- [ ] Finalise sympy file
- [x] Write code to input final segmented images from cache and predict the character
- [ ] Extend dataset to include operators (+, -, /, =, (), {}, [])
- [ ] Web integration
- [ ] (Optional) Equation class expansion
- [ ] (Optional) Handwritten output in-place
