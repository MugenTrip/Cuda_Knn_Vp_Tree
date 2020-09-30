vp_p_1 = fopen('knn_parallel\knn_5000_d_10.csv','r');
vp_p_2 = fopen('knn_parallel\knn_5000_d_10_v2.csv','r');
vp_p_3 = fopen('knn_parallel\knn_5000_d_10_v3.csv','r');
vp_s_1 = fopen('knn_sequential\knn_5000_d_10.csv','r');
vp_s_2 = fopen('knn_sequential\knn_5000_d_10_v2.csv','r');
vp_s_3 = fopen('knn_sequential\knn_5000_d_10_v3.csv','r');
A1 = fread(vp_p_1,30,'double');
A2 = fread(vp_p_2,30,'double');
A3 = fread(vp_p_3,30,'double');
B1 = fread(vp_s_1,30,'double');
B2 = fread(vp_s_2,30,'double');
B3 = fread(vp_s_3,30,'double');
C1 = 1:1:30;
A = (A1+A2+A3)/3;
B = (B1+B2+B3)/3;

figure
plot(C1,A','DisplayName','Parallel')
title('\fontsize{18} Sequential vs Parallel knn in respect of D. N=5000 k=10 ')
xlabel('\fontsize{16} Number of dimensions(D)')
ylabel('\fontsize{16} time(sec)')
hold on 
plot(C1,B','DisplayName','Sequential')
legend('\fontsize{20} Parallel','\fontsize{20} Sequential');