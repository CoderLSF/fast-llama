# Fast-LLaMA: A High-Performance Inference Engine
![image](https://github.com/CoderLSF/fast-llama/assets/65639063/7d09052c-1797-4b40-9fd4-c7d21408d0b2)

## Descriptions
fast-llama is a super `HIGH`-performance inference engine for LLMs like LLaMA (**3x** of `llama.cpp`) written in `pure C++`. It can run a **`8-bit`** quantized **`LLaMA2-7B`** model on a cpu with 56 cores in speed of **`~30 tokens / s`**. It outperforms all current open-source inference engines, especially when compared to the renowned llama.cpp, with 2~3 times better inference speed on a CPU.

## Features
Currently supported features:

| Feature Name | Current Support | Future Suport |
| --- | --- | --- |
| Model Types | LLaMA (Only Chinese-LLaMA2 1.3B & 7B are verified currently) | Baichuan |
| Quantization | INT16, INT8 | INT4 |
| Model Format | HuggingFace, gguf(by llama.cpp), flm | |
| System | Linux | Windows, Macbook |
| CPU/GPU | Intel CPU with AVX-512 | Apple Mx, GPU |

## **Advantages**
Why use Fast-LLaMA?
- **`Fast`**
   - Extremely fast on CPU. `Faster` than any other engines on Github including [llama.cpp](https://github.com/ggerganov/llama.cpp) (**`3 times`** faster than llama.cpp).
- **`Simple`**
   - Totally less than 7k lines of C++ codes with well-orgnized code structures and no dependencies except NUMA (if needed for multi-cpus).
- **`"Easy To Use"`** (target ☺️）

> ⚠️ Only **`CPU`** is supported currently. Support for GPU is coming soon.

## Quick Start

### Compile

Only Linux is supported currently. Support of other platforms including Windows, Mac, GPU is coming soon.

#### Requirements
- GCC 10.x or newer versions
- CPU with AVX-512
- libnuma-dev

> libraries like mpi, openblas, mkl, etc are NOT needed currently.

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
**`Step 1`**: Download model
See [llama2.c](https://github.com/karpathy/llama2.c)

**`Step 2`**: Run the model
```bash
./main -c ./models/stories110M.bin -z ./models/tokenizer.bin -j 14 -q int8 -n 200 -i 'That was a long long story happened in the ancient China.'
```

#### 2. Run with hugging face format models
**`Step 1`**: Download model
See [Chinese-LLaMA-Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)

**`Step 2`**: Convert model format
```bash
python3 ./tools/convert_flm.py -m /path/to/model-directory -o ./models/model-name-int8.flm -t int8
```

**`Step 3`**: Run the model
```bash
./main -c ./models/model-name-int8.flm -j 40 -n 200 -i 'That was a long long story happened in the ancient China.'
```

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

fast-llama achieves a generation speed of approximately 25-30 tokens/s for an 8-bit quantized 7B model running on the following CPU configuration:

```text
Architecture:            x86_64
Model name:              Intel(R) Xeon(R) Platinum 8350C CPU @ 2.60GHz
CPU(s):                  112 (56 physical cores)
Thread(s) per core:      2
Core(s) per socket:      28
Socket(s):               2
```
![image](https://github.com/CoderLSF/RapidLLaMA/assets/65639063/d4477fcb-96fb-4b0a-a1fd-30ca583d0aa2)

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

We would like to express our gratitude to all contributors and users of FastLLaMA. Your support and feedback have been invaluable in making this project a success. If you encounter any issues or have any suggestions, please feel free to open an issue on the GitHub repository.

## Contact
Email: [📩topcoderlsf@gmail.com](topcoderlsf@gmail.com)

Contact me if you any questions.
