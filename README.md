# Simple Repo for code to generate dense neural network training on MNIST digits dataset and saving per-epoch confusion matricies as images to animate into a gif of the neural network training. 

Also added a 'perfect' confusion matrix on the left side of the animation to illustrate what the neural network is attempting to achieve through training. 

Training parameters have been manipulated to produce per-epoch confusion matricies that better illustrate the learning process for the sake of the animation, not the overall accuracy of the neural network.

Animation and confusion matrix plotting also done in simple for-loop to keep legibility up as compared to doing a more complicated 'callback' function and integrating it into the keras fit arguments.

![Confusion matrix evolution gif](https://github.com/kjaehnig/neural_network_confusion_matrix_animation/blob/main/confusion_matrix_evolution.gif " ")
)