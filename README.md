# Fast-LLaMA: A High-Performance Inference Engine
<p align="center"><img width="600" alt="image" src="https://github.com/CoderLSF/fast-llama/assets/65639063/d3d66d72-bf91-4bef-b4e8-468227cfea05"></p>


## Descriptions
fast-llama is a super high-performance inference engine for LLMs like LLaMA (**2.5x** of `llama.cpp`) written in `pure C++`. It can run a **`8-bit`** quantized **`LLaMA2-7B`** model on a cpu with 56 cores in speed of **`~25 tokens / s`**. It outperforms all current open-source inference engines, especially when compared to the renowned llama.cpp, with ~2.5 times better inference speed on a CPU.

## Features

| Feature Name | Current Support | Future Suport |
| --- | --- | --- |
| **Model Types** | ✅LLaMA2 | Others LLMs like Baichuan, StableDiffusion |
| **Quantization** | ✅INT16, ✅INT8 | INT4 |
| **Model Formats** | ✅HuggingFace, ✅gguf(by llama.cpp), ✅flm | |
| **Systems** | ✅Linux, ✅Windows | Macbook, Android, iOS |
| **CPU/GPU** | ✅X86/64 CPU | ARM, Apple Mx CPUs, GPU, CPU+GPU |
| **Architectures** | ✅UMA, ✅NUMA | |

## **Advantages**
Why you should use Fast-LLaMA?
- **`Fast`**
   - Extremely fast on CPU. `Faster` than any other engines on Github including [llama.cpp](https://github.com/ggerganov/llama.cpp).
- **`Simple`**
   - Totally less than 7k lines of C++ codes with well-orgnized code structures and no dependencies except NUMA (if needed for multi-cpus).
- **`"Easy To Use"`** (target ☺️）

## Quick Start

### Compile

Only Linux is supported currently. Support of other platforms including Windows, Mac, GPU is coming soon.

#### Requirements
- `GCC 10.x` or newer versions
- `libnuma-dev` if your computer has more than one physical CPUs
   - `Linux Kernel v5.x` or higher is needed for NUMA

#### Compilation
Method 1. Using the provided build script:
```bash
bash ./build.sh
```

Method 2. Using Make:
```bash
make -j 4
```

### Run

#### 1. Run with llama2.c models:
**`Step 1`**: Download a model

>   See [llama2.c](https://github.com/karpathy/llama2.c)

**`Step 2`**: Run the model
```bash
./main -c ./models/stories110M.bin -z ./models/tokenizer.bin -j 14 -q int8 -n 200 -i 'That was a long long story happened in the ancient China.'
```

<img width="1501" alt="image" src="https://github.com/CoderLSF/fast-llama/assets/65639063/28156af1-142e-417c-9b94-8d931fac8884">

#### 2. Run with hugging face format models
**`Step 1`**: Download a model

>   See [Chinese-LLaMA-Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)

**`Step 2`**: Convert the model info FLM format
```bash
python3 ./tools/convert_flm.py -m /path/to/model-directory -o ./models/model-name-int8.flm -t int8
```

**`Step 3`**: Run the model
```bash
./main -c ./models/model-name-int8.flm -j 40 -n 200 -i 'That was a long long story happened in the ancient China.'
```

![image](https://github.com/CoderLSF/RapidLLaMA/assets/65639063/d4477fcb-96fb-4b0a-a1fd-30ca583d0aa2)

> 
All supported command-line options are as follows:

- `-c`: Path to the model file
- `-f`: Model file format (e.g., gguf)
- `-j`: Number of threads to use (e.g., 56)
- `-q`: Quantization mode (e.g., int8)
- `-n`: Number of tokens to generate (e.g., 200)
- `-i`: Input text (e.g., 'That was a long long story happened in the ancient China.')
- `-h`: show usage information

## Performance
Below are some incomplete test results

#### Testing Result:

| Model | Model Size | OutputSpeed/`8` threads | OutputSpeed/`28` threads | OutputSpeed/`56` threads |
| :--: | --: | --: | --: | --: |
| stories110M | 110M | `237`tps | `400`tps | `440`tps |
| Chinese-LLaMA-1.3B | 1.3B | `38.9`tps | `127`tps | `155`tps |
| Chinese-LLaMA-7B | 7B | `7.4`tps | `17.4`tps | `23.5`tps |

* Note: tps = tokens / second

#### Testing Conditions

- **Testing Prompt**: "That was a long long story happened in the ancient Europe. It was about a brave boy name Oliver. Oliver lived in a small village among many big moutains. It was a beautiful village."
- **Quantization**: `int8`
- **NUMA**: `2` sockets
   - **Note**: Make sure that NUMA is truly available if you expect to accelerate with NUMA)
- **System**: (`uname -a`)Linux coderlsf 5.15.0-72-generic #79-Ubuntu SMP Wed Apr 19 08:22:18 UTC 2023 x86_64 x86_64 x86_64 GNU/Linux
- **CPU**: `56` physical cores, `AVX-512`

```text
Architecture:            x86_64
Model name:              Intel(R) Xeon(R) Platinum 8350C CPU @ 2.60GHz
CPU(s):                  112 (56 physical cores)
Thread(s) per core:      2
Core(s) per socket:      28
Socket(s):               2
```


![2a58bda471f0aa2770f349dba73a530d](https://github.com/CoderLSF/fast-llama/assets/65639063/c3634948-280d-47c8-b9e7-ff07d7104b86)


> Latancy of first token will be optimized laterly.

## Why
Why is it so fast?
- Ultimate memory efficiency
   - Zero memory allocations and frees during inferencing.
   - Maximization of memory locality.
- Well-designed thread scheduling algorithm
- Optimized operators
   - Fuse all operators that can be fused together
   - Optmize calculation of several operators
- Proper Quantizations

## License

fast-llama is licensed under the [MIT](LICENSE).

## Acknowledgements
Special thanks to [AlpinDale](https://github.com/AlpinDale) for his professional, meticulous, and patient guidance and assistance.

## Contact
Email: [📩topcoderlsf@gmail.com](topcoderlsf@gmail.com)

Contact me if you any questions.
