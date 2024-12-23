make clean
make
mkdir -p output/

./cuda-isp images/test_3000x2000.png images/test_raw_3000x2000.bmp degrad
./cuda-isp images/test_raw_3000x2000.bmp output/outout_cpu_3000x2000.bmp cpu
./cuda-isp images/test_raw_3000x2000.bmp output/outout_gpu_3000x2000.bmp gpu

./cuda-isp images/test_6000x4000.png images/test_raw_6000x4000.bmp degrad
./cuda-isp images/test_raw_6000x4000.bmp output/outout_cpu_6000x4000.bmp cpu
./cuda-isp images/test_raw_6000x4000.bmp output/outout_gpu_6000x4000.bmp gpu

./cuda-isp images/test_12000x8000.png images/test_raw_12000x8000.bmp degrad
./cuda-isp images/test_raw_12000x8000.bmp output/outout_cpu_12000x8000.bmp cpu
./cuda-isp images/test_raw_12000x8000.bmp output/outout_gpu_12000x8000.bmp gpu
