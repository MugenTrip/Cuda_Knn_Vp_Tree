vp_p_1 = fopen('vptree_parallel\vptree_2000000_d.csv','r');
vp_p_2 = fopen('vptree_parallel\vptree_2000000_d_v2.csv','r');
vp_p_3 = fopen('vptree_parallel\vptree_2000000_d_v3.csv','r');
vp_s_1 = fopen('vptree_sequential\vptree_2000000_d.csv','r');
vp_s_2 = fopen('vptree_sequential\vptree_2000000_d_v2.csv','r');
vp_s_3 = fopen('vptree_sequential\vptree_2000000_d_v3.csv','r');
A1 = fread(vp_p_1,11,'double');
A2 = fread(vp_p_2,11,'double');
A3 = fread(vp_p_3,11,'double');
B1 = fread(vp_s_1,11,'double');
B2 = fread(vp_s_2,11,'double');
B3 = fread(vp_s_3,11,'double');
C1 = 4:10:104;
A = (A1+A2+A3)/3;
B = (B1+B2+B3)/3;

figure
plot(C1,A','DisplayName','Parallel')
title('\fontsize{18} Sequential vs Parallel vptree construction in respect of d. N=2000000')
xlabel('\fontsize{16} Number of dimensions(d)')
ylabel('\fontsize{16} time(sec)')
hold on 
plot(C1,B','DisplayName','Sequential')
legend('\fontsize{20} Parallel','\fontsize{20} Sequential');