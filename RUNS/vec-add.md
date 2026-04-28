(.venv) (base) manish@war:~/Desktop/Xircuit-GITHUB/SIMPLE-CUDA$ nsys profile --stats=true -o vecadd_run \
>     python3 -c "import src.handler as h; h.handler({'input': {'n': 1 << 20}})"
WARNING: CPU IP/backtrace sampling not supported, disabling.
Try the 'nsys status --environment' command to learn more.

WARNING: CPU context switch tracing not supported, disabling.
Try the 'nsys status --environment' command to learn more.

Generating '/tmp/nsys-report-9090.qdstrm'
Failed to create '/home/manish/Desktop/Xircuit-GITHUB/SIMPLE-CUDA/vecadd_run.nsys-rep': File exists.
Use `--force-overwrite true` to overwrite existing files.
[1/8] [========================100%] nsys-report-9777.nsys-rep
[2/8] [========================100%] vecadd_run.sqlite
[3/8] Executing 'nvtx_sum' stats report
SKIPPED: /home/manish/Desktop/Xircuit-GITHUB/SIMPLE-CUDA/vecadd_run.sqlite does not contain NV Tools Extension (NVTX) data.
[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)    Med (ns)   Min (ns)    Max (ns)     StdDev (ns)           Name         
 --------  ---------------  ---------  -----------  ---------  --------  -------------  -----------  ----------------------
     84.4    1,810,146,578    141,176     12,821.9      627.0       605  1,273,426,646  3,445,161.3  poll                  
      4.9      105,719,690      1,866     56,655.8    1,361.0       651     25,929,652  1,121,100.7  read                  
      4.5       97,203,440    141,151        688.6      626.0       607      1,685,726      7,501.8  waitpid               
      2.8       59,480,580        544    109,339.3   11,837.5       624     14,989,977    962,090.3  ioctl                 
      0.9       18,724,042         18  1,040,224.6    3,412.5     1,516     18,663,718  4,398,244.3  open                  
      0.8       16,362,501         31    527,822.6    3,375.0     1,237     16,206,702  2,909,880.0  fopen                 
      0.5       10,927,344      4,968      2,199.5    1,729.0     1,588         59,946      1,398.7  munmap                
      0.5       10,327,760    596,544         17.3       17.0        12         21,409         40.3  pthread_cond_signal   
      0.4        8,106,199      5,022      1,614.1    1,025.0       908      1,445,600     20,508.4  mmap64                
      0.2        3,420,521        931      3,674.0    3,521.0     1,913         22,395      1,001.4  open64                
      0.1        1,111,566      7,986        139.2       47.0        31         55,711      1,030.4  fgets                 
      0.0          541,430         10     54,143.0   53,174.0    43,555         72,398      8,893.4  sem_timedwait         
      0.0          374,314        721        519.2      514.0        12          4,204        184.7  sigaction             
      0.0          305,576          1    305,576.0  305,576.0   305,576        305,576          0.0  fork                  
      0.0          240,978        182      1,324.1    1,094.0       913          4,535        594.7  fclose                
      0.0          182,485          5     36,497.0   32,597.0    21,970         72,327     20,790.1  pthread_create        
      0.0          130,642         21      6,221.0    2,869.0     1,059         59,344     12,364.4  mmap                  
      0.0          115,882          1    115,882.0  115,882.0   115,882        115,882          0.0  pthread_cond_wait     
      0.0          110,009         34      3,235.6    2,606.0     1,665          6,052      1,415.5  pipe2                 
      0.0           70,666        953         74.2       33.0        30          3,117        213.2  fputs                 
      0.0           66,157      2,828         23.4       20.0        17            111          8.0  flockfile             
      0.0           64,865         79        821.1      740.0       536          3,571        370.7  fcntl                 
      0.0           58,114         78        745.1      739.0       735            891         26.2  pread                 
      0.0           54,970         13      4,228.5    3,839.0     1,613          7,502      2,312.5  fopen64               
      0.0           34,097         19      1,794.6    1,459.0       880          4,059        905.6  write                 
      0.0           33,505         10      3,350.5    3,557.5        84          7,028      2,607.3  fread                 
      0.0           17,644         23        767.1      634.0       577          1,241        208.7  dup2                  
      0.0           17,392          3      5,797.3    5,890.0     3,283          8,219      2,469.3  socket                
      0.0            6,669          1      6,669.0    6,669.0     6,669          6,669          0.0  connect               
      0.0            6,425          3      2,141.7    1,559.0     1,549          3,317      1,017.9  pthread_cond_broadcast
      0.0            6,148          2      3,074.0    3,074.0     2,987          3,161        123.0  fputs_unlocked        
      0.0            5,455          2      2,727.5    2,727.5     2,633          2,822        133.6  mprotect              
      0.0            5,105          8        638.1      614.0       570            752         72.3  dup                   
      0.0            5,049          2      2,524.5    2,524.5     1,466          3,583      1,496.9  bind                  
      0.0            3,437          2      1,718.5    1,718.5     1,525          1,912        273.7  fwrite                
      0.0            3,055         64         47.7       26.0        24            216         36.7  pthread_mutex_trylock 
      0.0            2,426         10        242.6      178.0        32            814        249.6  fflush                
      0.0            1,222          1      1,222.0    1,222.0     1,222          1,222          0.0  fputc                 
      0.0              878          1        878.0      878.0       878            878          0.0  listen                

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)    Med (ns)   Min (ns)   Max (ns)   StdDev (ns)            Name         
 --------  ---------------  ---------  ------------  ---------  --------  ----------  ------------  ----------------------
     96.7       84,179,447          3  28,059,815.7   63,168.0    61,921  84,054,358  48,492,696.1  cudaMalloc            
      2.9        2,543,854          3     847,951.3  765,644.0   704,744   1,073,466     197,661.0  cudaMemcpy            
      0.2          163,353          3      54,451.0   44,555.0    42,530      76,268      18,921.2  cudaFree              
      0.2          149,253          1     149,253.0  149,253.0   149,253     149,253           0.0  cudaLaunchKernel      
      0.0           30,999          1      30,999.0   30,999.0    30,999      30,999           0.0  cudaEventSynchronize  
      0.0            9,616          2       4,808.0    4,808.0     2,917       6,699       2,674.3  cudaEventRecord       
      0.0            9,535          2       4,767.5    4,767.5       449       9,086       6,107.3  cudaEventCreate       
      0.0            2,079          1       2,079.0    2,079.0     2,079       2,079           0.0  cuModuleGetLoadingMode
      0.0            1,235          2         617.5      617.5       278         957         480.1  cudaEventDestroy      

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                         Name                       
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  --------------------------------------------------
    100.0           31,520          1  31,520.0  31,520.0    31,520    31,520          0.0  vecAdd(const float *, const float *, float *, int)

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)           Operation          
 --------  ---------------  -----  ---------  ---------  --------  --------  -----------  ----------------------------
     64.0        1,269,918      2  634,959.0  634,959.0   621,055   648,863     19,663.2  [CUDA memcpy Host-to-Device]
     36.0          715,743      1  715,743.0  715,743.0   715,743   715,743          0.0  [CUDA memcpy Device-to-Host]

[8/8] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)           Operation          
 ----------  -----  --------  --------  --------  --------  -----------  ----------------------------
      8.389      2     4.194     4.194     4.194     4.194        0.000  [CUDA memcpy Host-to-Device]
      4.194      1     4.194     4.194     4.194     4.194        0.000  [CUDA memcpy Device-to-Host]

Generated:
    /tmp/nsys-report-9777.nsys-rep
    /home/manish/Desktop/Xircuit-GITHUB/SIMPLE-CUDA/vecadd_run.sqlite
