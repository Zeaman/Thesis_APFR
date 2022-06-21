function [w1, w2, w3, w4, w5] = DeepLearning(w1, w2, w3, w4, w5, input_image, correct_Output)
alpha = 0.01;

n = 6;
for k = 1:6
    reshaped_input_image = reshape(input_image(:,:,k), 36, 1);
    
    input_of_hidden_layer1 = w1* reshaped_input_image;
    output_of_hidden_layer1 = ReLU(input_of_hidden_layer1);
    
    input_of_hidden_layer2 = w2* output_of_hidden_layer1;
    output_of_hidden_layer2 = ReLU(input_of_hidden_layer2);
    
    input_of_hidden_layer3 = w3* output_of_hidden_layer2;
    output_of_hidden_layer3 = ReLU(input_of_hidden_layer3);
    
    input_of_hidden_layer4 = w4* output_of_hidden_layer3;
    output_of_hidden_layer4 = ReLU(input_of_hidden_layer4);
    
    input_of_output_node = w5* output_of_hidden_layer4;
    final_output = softmax(input_of_output_node);
    
    
    correct_Output_transpose = correct_Output(k, :)';
    error = correct_Output_transpose - final_output;
    
    delta = error;
    
    error_of_hidden_layer4 = w5'*delta;
    delta4 = (input_of_hidden_layer4>0).*error_of_hidden_layer4;

    error_of_hidden_layer3 = w4'*delta4;
    delta3 = (input_of_hidden_layer3>0).*error_of_hidden_layer3;
    
    error_of_hidden_layer2 = w3'*delta3;
    delta2 = (input_of_hidden_layer2>0).*error_of_hidden_layer2;
    
    error_of_hidden_layer1 = w2'*delta2;
    delta1 = (input_of_hidden_layer1>0).*error_of_hidden_layer1;
    
    
    adjustment_of_w5 = alpha*delta*output_of_hidden_layer4';
    adjustment_of_w4 = alpha*delta4*output_of_hidden_layer3';
    adjustment_of_w3 = alpha*delta3*output_of_hidden_layer2';
    adjustment_of_w2 = alpha*delta2*output_of_hidden_layer1';
    adjustment_of_w1 = alpha*delta1*reshaped_input_image';
    
    
    w1 = w1 + adjustment_of_w1;
    w2 = w2 + adjustment_of_w2;
    w3 = w3 + adjustment_of_w3;
    w4 = w4 + adjustment_of_w4;
    w5 = w5 + adjustment_of_w5;
    
    
end

end
    
    
    
    
    