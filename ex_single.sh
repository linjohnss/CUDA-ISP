make
mkdir -p output/

./cuda-isp images/R0004864.png images/test_raw.bmp degrad
time ./cuda-isp images/test_raw.bmp output/outout_cpu.bmp cpu
time ./cuda-isp images/test_raw.bmp output/outout_gpu.bmp gpu