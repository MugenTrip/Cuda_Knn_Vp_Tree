vp_p_1 = fopen('knn_parallel\knn_5000_5_k.csv','r');
vp_p_2 = fopen('knn_parallel\knn_5000_5_k_v2.csv','r');
vp_p_3 = fopen('knn_parallel\knn_5000_5_k_v3.csv','r');
vp_s_1 = fopen('knn_sequential\knn_5000_5_k.csv','r');
vp_s_2 = fopen('knn_sequential\knn_5000_5_k_v2.csv','r');
vp_s_3 = fopen('knn_sequential\knn_5000_5_k_v3.csv','r');
A1 = fread(vp_p_1,31,'double');
A2 = fread(vp_p_2,31,'double');
A3 = fread(vp_p_3,31,'double');
B1 = fread(vp_s_1,31,'double');
B2 = fread(vp_s_2,31,'double');
B3 = fread(vp_s_3,31,'double');
C1 = 1:4:121;
A = (A1+A2+A3)/3;
B = (B1+B2+B3)/3;

figure
plot(C1,A','DisplayName','Parallel')
title('\fontsize{18} Sequential vs Parallel knn in respect of K. N=5000 D=5 ')
xlabel('\fontsize{16} Number of neighbors(k)')
ylabel('\fontsize{16} time(sec)')
hold on 
plot(C1,B','DisplayName','Sequential')
legend('\fontsize{20} Parallel','\fontsize{20} Sequential');