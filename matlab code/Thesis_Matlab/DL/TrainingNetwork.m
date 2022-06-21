input_image = zeros(6,6,6);

input_image(:,:,1) = [0 1 1 1 1 0;
                      0 1 1 1 1 0;
                      0 0 1 1 1 0;
                      0 0 1 1 1 0;
                      0 0 1 1 1 0;
                      0 0 1 1 1 0;
                      ];
input_image(:,:,2) = [1 1 1 1 1 0;
                      0 0 0 1 1 0;
                      1 1 1 1 1 0;
                      1 1 0 0 0 0;
                      1 1 1 1 1 0;
                      0 0 1 1 1 0;
                      ]; 
               
correct_Output = [1 0 0 0 0 0;
                  0 1 0 0 0 0;
                  0 0 1 0 0 0;
                  0 0 0 1 0 0;
                  0 0 0 0 1 0;
                  0 0 1 1 1 0;
                  ];
               
 w1 = 2*rand(20,25)-1;
 w2 = 2*rand(20,20)-1;
 w3 = 2*rand(20,20)-1;
 w4 = 2*rand(5,20)-1;
 w5 = 2*rand(5,20)-1;
 
 for epoch = 1:50
     [w1, w2, w3, w4, w5] = DeepLearning(w1, w2, w3, w4, w5, input_image, correct_Output);
 end
 save('DeepNeuralNetwork.mat')