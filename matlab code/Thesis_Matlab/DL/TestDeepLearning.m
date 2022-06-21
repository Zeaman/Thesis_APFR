load('DeepNeuralNetwork.mat')

input_image = [1 1 1 1 1;
               0 0 0 1 1;
               1 1 1 1 1;
               1 1 0 0 0;
               1 1 1 1 1;
               ]; 

input_image = reshape(input_image, 25, 1);

input_of_hidden_layer1 = w1*input_image;
output_of_hidden_layer1 = ReLU(input_of_hidden_layer1);

input_of_hidden_layer2 = w2*output_of_hidden_layer1;
output_of_hidden_layer2 = ReLU(input_of_hidden_layer2);

input_of_hidden_layer3 = w3*output_of_hidden_layer2;
output_of_hidden_layer3 = ReLU(input_of_hidden_layer3);

input_of_hidden_layer4 = w4* output_of_hidden_layer3;
output_of_hidden_layer4 = ReLU(input_of_hidden_layer4);

input_of_output_node = w5*output_of_hidden_layer4;
final_output = softmax(input_of_output_node)