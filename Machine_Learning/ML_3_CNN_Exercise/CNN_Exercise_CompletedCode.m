%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Student Number : 2015104032, 2015104124, 2015104045
% Name : 박지훈, 진우빈, 서재하
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
clear all;
format bank %Precision : second decimal place

input_image = ...
    [0.1 0.0 0.9 0.9;
    0.3 0.2 1.0 0.9;
    0.0 0.1 0.8 1.0;
    0.1 0.2 0.9 0.8];

[Img_Ver_Size, Img_Hor_Size]=size(input_image);
Target = [0;1;0;0] %Vertical direction
Target_Length=length(Target);

Learn_Rate = 0.1
iteration_numbur = 100

Conv_Win_Size =3;
Pooling_size = 2;


%%%%%%%%% Functions used for Forward Propagation
% ****** Convolution Layer
%  Convolution :   Out = filter2(Weight,In)
%  Relu :          Out = max(In,0)
% Max_Pooling : Out = col2im(max(im2col( In ,[pooling_size pooling_size],'distinct')),
%                      [1 1],size( In )/pooling_size)

%  ****** Fully connected Layer
%  Matrix to vector : Out = reshape( In ,[size( In ,1)*size( In ,2) 1]);
%  Soft_Max :   Out = exp( In -max( In ))./sum(exp( In -max( In )));
%

% Loss function
%  Cross Entropy :     Out = -sum( In1 .*log( In2 ));
%                       ( In1 : Target vector, In2 : prediction vector )
%

%%%%%%%%%% Functions used for Backward Propagation
%  Fully connected Layer
%  Rehaping matrix to vector :  Out = reshape( In ,[size( In ,1)*size( In ,2) 1])';

% Convolution Layer
%  Backward Max Pooling :
%Out = imresize(col2im( In ,[1 1],[sqrt(length( In )) sqrt(length( In ))]), pooling_size, 'nearest');
%  Backward of Relu :  Out = double( In1 ).*double( In2 >0);
%                  ( In1 : gradient, In2 : input of Relu at forward propagation )

%  Gradient of w_conv : Out = filter2( In1 , In2 ,'valid');
%                    ( In1 : gradient, In2 : padded input image )
%  Sum of matrix values : Out = sum(sum( In ));
%

% Initilizing the weights of convolution (conv) layer
w_conv = 2*(rand(Conv_Win_Size ,Conv_Win_Size)-0.5);
b_conv = 0.1*(rand(1,1)-0.5);

% Initilizing the weights of fully connected (fc) layer
fc_inSize = Img_Ver_Size*Img_Hor_Size / (Pooling_size^2);
fc_outSize = Target_Length;
w_fc = 2*(rand(fc_outSize, fc_inSize)-0.5);
b_fc = 0.1*(rand(fc_outSize,1)-0.5);

%Variables%%

a1 = []    % variables for convolution layer
z1 = []    % variables for convolution layer
z1_pool=[] % variables for convolution layer
a2 = []    % variables for fully connected layer
z2 = []    % variables for fully connected layer

%Completed Codes%%

i = 0;
figure(1);
while (i < iteration_numbur)
    i
    %Forward porpagation%%
    %- convolution
    a1=filter2(w_conv,input_image,'same');
    b_vec=[b_conv b_conv b_conv b_conv;
           b_conv b_conv b_conv b_conv;
           b_conv b_conv b_conv b_conv;
           b_conv b_conv b_conv b_conv];
    a1=a1+b_vec;
    % - Relu
    z1=max(a1,0);
    % - max pooling 2x2
    z1_pool=col2im(max(im2col(z1,[2 2],'distinct')),[1 1],size(z1)/2);
    
    % * Fully Connected Layer
    % - matrix product
    z1_pool_re=reshape( z1_pool ,[size( z1_pool ,1)*size( z1_pool ,2) 1])
    a2=w_fc*z1_pool_re+b_fc;
    % - soft-max
    z2=exp( a2 -max( a2 ))./sum(exp( a2 -max( a2 )));
    
    % * Calculation Loss : cross-entropy
    loss=-sum( Target .*log( z2 ));
    
    figure(1), plot(i,loss,'b*'), axis([0, iteration_numbur, 0, 2]),
    hold on; grid on; xlabel('iteration'), ylabel('loss');
    
    %Backward propagation%%
    
    dy_dz2 = -(Target./z2);
    dy_da2 = z2-Target;
    dy_db_fc = dy_da2;
    dy_dw_fc = dy_da2 * transpose(z1_pool_re);
    
    
    % - Update weights of fully connected layer
    w_fc = w_fc - Learn_Rate*(dy_dw_fc);
    b_fc = b_fc - Learn_Rate*(dy_db_fc);
    
    % - Calculate Gradient of z1
    dy_dz1_pool_re = w_fc * dy_da2;
    dy_dz1_pool = reshape(dy_dz1_pool_re ,[size( dy_dz1_pool_re ,1)*size( dy_dz1_pool_re ,2) 1]);
    
    % * Convolution Layer
    % - Backward of Max pooling
    dy_dz1_pooling_back = imresize(col2im( dy_dz1_pool ,[1 1],[sqrt(length( dy_dz1_pool )) sqrt(length( dy_dz1_pool ))]), 2, 'nearest');
    
    % - Backward of Relu (= Gradient of a1)
    dy_da1 = double( dy_dz1_pooling_back ).*double( a1 >0);
    
    % - Gradient of w_conv
    dy_db1 = dy_da1;
    
    padding = zeros(size(input_image)+size(w_conv)-1);
    padding((size(w_conv,1)-1)/2+1:end-(size(w_conv,1)-1)/2,(size(w_conv,1)-1)/2+1:...
        end-(size(w_conv,1)-1)/2) = input_image;
    
    dy_dw_conv = filter2( dy_da1 , padding ,'valid');
    
    % - Update weights of convolution layer
    w_conv = w_conv - Learn_Rate*(dy_dw_conv);
    b_conv = b_conv - Learn_Rate*(sum(sum(dy_db1)));
    
    i = i + 1;
end
