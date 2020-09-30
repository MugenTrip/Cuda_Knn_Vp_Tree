vp_p_1 = fopen('vptree_parallel\vptree_n_10.csv','r');
vp_p_2 = fopen('vptree_parallel\vptree_n_10_v2.csv','r');
vp_p_3 = fopen('vptree_parallel\vptree_n_10_v3.csv','r');
vp_s_1 = fopen('vptree_sequential\vptree_n_10.csv','r');
vp_s_2 = fopen('vptree_sequential\vptree_n_10_v2.csv','r');
vp_s_3 = fopen('vptree_sequential\vptree_n_10_v3.csv','r');
A1 = fread(vp_p_1,60,'double');
A2 = fread(vp_p_2,60,'double');
A3 = fread(vp_p_3,60,'double');
B1 = fread(vp_s_1,60,'double');
B2 = fread(vp_s_2,60,'double');
B3 = fread(vp_s_3,60,'double');
C1 = 3:50000:2950003;
A = (A1+A2+A3)/3;
B = (B1+B2+B3)/3;

figure
plot(C1,A','DisplayName','Parallel')
title('\fontsize{18} Sequential vs Parallel vptree construction in respect of n. D=10 ')
xlabel('\fontsize{16} Number of elements(n)')
ylabel('\fontsize{16} time(sec)')
hold on 
plot(C1,B','DisplayName','Sequential')
legend('\fontsize{20} Parallel','\fontsize{20} Sequential');