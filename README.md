# RapidLLaMA
Runs LLaMA with very **`HIGH Speed`** on **`CPU`** (**3x** of `llama.cpp`)

## Compile

```bash
bash ./build.sh
```
or
```bash
make
```

## Run
```bash
./main -c ./models/cnllama-7b/ggml-model-f32.gguf -f gguf -j 56 -q int8 -n 200 -i 'That was a long long story happened in the ancient China.'
```


## Benchmark
```bash
 ./main -c ./models/cnllama-7b/ggml-model-f32.gguf -f gguf -j 56 -q int8 -n 200
```

![image](https://github.com/CoderLSF/RapidLLaMA/assets/65639063/d4477fcb-96fb-4b0a-a1fd-30ca583d0aa2)

## CPU Inframation
```text
Architecture:            x86_64
  CPU op-mode(s):        32-bit, 64-bit
  Address sizes:         46 bits physical, 57 bits virtual
  Byte Order:            Little Endian
CPU(s):                  112
  On-line CPU(s) list:   0-111
Vendor ID:               GenuineIntel
  Model name:            Intel(R) Xeon(R) Platinum 8350C CPU @ 2.60GHz
    CPU family:          6
    Model:               106
    Thread(s) per core:  2
    Core(s) per socket:  28
    Socket(s):           2
    Stepping:            6
    BogoMIPS:            5187.80
    Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon rep_good nopl xtopology nonstop_tsc cpuid tsc_known_freq pni pclmulqdq mon
                         itor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch cpuid_fault invpcid_single ssbd ibrs ibpb stibp ibrs_enhanced fsgsbase tsc_adjust bmi1 avx2
                         smep bmi2 erms invpcid avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves arat avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_b
                         italg avx512_vpopcntdq la57 rdpid md_clear arch_capabilities
Virtualization features:
  Hypervisor vendor:     KVM
  Virtualization type:   full
Caches (sum of all):
  L1d:                   2.6 MiB (56 instances)
  L1i:                   1.8 MiB (56 instances)
  L2:                    70 MiB (56 instances)
  L3:                    96 MiB (2 instances)
NUMA:
  NUMA node(s):          2
  NUMA node0 CPU(s):     0-55
  NUMA node1 CPU(s):     56-111
Vulnerabilities:
  Itlb multihit:         Not affected
  L1tf:                  Not affected
  Mds:                   Not affected
  Meltdown:              Not affected
  Mmio stale data:       Vulnerable: Clear CPU buffers attempted, no microcode; SMT Host state unknown
  Retbleed:              Not affected
  Spec store bypass:     Vulnerable
  Spectre v1:            Mitigation; usercopy/swapgs barriers and __user pointer sanitization
  Spectre v2:            Vulnerable, IBPB: disabled, STIBP: disabled, PBRSB-eIBRS: Vulnerable
  Srbds:                 Not affected
  Tsx async abort:       Not affected
```
