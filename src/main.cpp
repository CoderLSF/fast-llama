
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string_view>

#include "transformer.h"
#include "utility.h"
#include "console.h"
#include "simd.h"

using namespace cpuft;
enum class Mode {
    NONE = 0,
    GEN,
    CHAT,
    TEST,
};
struct Arguments {
    const char*     ckpt_path       = "";
    const char*     tknr_path       = "";
    ModelFileType   mft             = ModelFileType::UNKNOWN;
    const char*     prompt          = "";
    bool            use_numa        = true;
    int             num_threads     = -1;
    int             max_tokens      = 512;
    float           topp            = 0.9f;
    float           temp            = 1.0f;
    QuantType       qtype           = QuantType::INT8;

    Mode            mode            = Mode::GEN;
    int             rounds          = 0;

    int             max_batch_size  = 128;
    int             seed            = 128391297;
    bool            print_detail    = false;
    bool            is_debug        = false;

    void print_usage(const char* bin_name);
    void parse(int argc, const char** argv);
};

int main(int argc, const char** argv) {
    Arguments args;
    args.parse(argc, argv);

    if (args.prompt == nullptr || args.prompt[0] == '\0') {
        args.prompt = "That was a long long story happened in the ancient Europe. It was about a brave boy name Oliver. Oliver lived in a small village among many big moutains. It was a beautiful village.";
    }

    Console con;
    if (args.print_detail) {
        fprintf(stderr, "num_threads:%s%d%s\n", con.yellow(), args.num_threads, con.endtag());
        fprintf(stderr, "   use_numa:%s%d%s\n", con.yellow(), args.use_numa,    con.endtag());
        fprintf(stderr, "  ckpt_path:%s%s%s\n", con.yellow(), args.ckpt_path,   con.endtag());
        fprintf(stderr, "  tknr_path:%s%s%s\n", con.yellow(), args.tknr_path,   con.endtag());
        fprintf(stderr, "      top_p:%s%g%s\n", con.yellow(), args.topp,        con.endtag());
        fprintf(stderr, "temperature:%s%g%s\n", con.yellow(), args.temp,        con.endtag());
        //fprintf(stderr, "     prompt:\x1b[33m%s\x1b[0m\n", args.prompt);
        fprintf(stderr, "\n");
    }

    srand(args.seed);

    ParallelTransformer ptf(args.print_detail || args.is_debug);
    bool ok = ptf.load(args.ckpt_path, args.tknr_path, args.mft, args.qtype, args.num_threads, args.use_numa, args.max_batch_size);
    if (!ok) {
        fprintf(stderr, "Failed to load model\n");
        exit(1);
    }
    args.qtype = ptf.get_quant_type();

    if (args.print_detail) {
        fprintf(stderr, "Model loaded\n\n");
    }

    double avg_prompt_token_num = 1e-10;
    double avg_output_token_num = 1e-10;
    double avg_prompt_latancy = 1e-10;
    double avg_output_latancy = 1e-10;
    for (int i = 0; i < args.rounds; ++i) {
        int prompt_token_num = 0;
        int output_token_num = 0;
        int first_latancy_us = 0;
        int total_latancy_us = 0;
        Timer tmr;
        auto cb = [&](const char* text, int num_input_tokens, int num_output_tokens, bool ended)->bool {
            if (first_latancy_us == 0) {
                if (args.mode != Mode::TEST) {
                    printf("prompt: %s%s%s\n", con.yellow(), args.prompt, con.endtag());
                    printf("output: %s", con.green());
                }
                first_latancy_us = int(tmr.elapsed_us());
                prompt_token_num = num_input_tokens;
            } else {
                output_token_num = num_output_tokens;
            }
            if (args.mode != Mode::TEST && text != nullptr ) {
                printf("%s", text); fflush(stdout);
            }
            return !ended;
        };

        tmr.reset();
        ptf.generate(args.prompt, cb, args.max_tokens, args.temp, args.topp);
        total_latancy_us = int(tmr.elapsed_us());

        if (args.mode != Mode::TEST) {
            printf("%s\n\n", con.endtag());
        }
        avg_prompt_token_num += prompt_token_num;
        avg_output_token_num += output_token_num;
        avg_prompt_latancy += first_latancy_us / 1000.;
        avg_output_latancy += (total_latancy_us - first_latancy_us) / 1000.;
    }

    avg_prompt_token_num /= args.rounds;
    avg_output_token_num /= args.rounds;
    avg_prompt_latancy   /= args.rounds;
    avg_output_latancy   /= args.rounds;
    auto first_token_latancy = avg_prompt_latancy / avg_prompt_token_num;
    auto later_token_latancy = avg_output_latancy / (avg_output_token_num - 1);

    printf("num_threads:%s%2d%s\tquant:%s%s%s\tuse_numa:%s%d%s\tsimd_size:%d\tprompt_size:%3d\toutput_size:%3d\ttotal_latancy:%5.0fms\t"
           "prompt_token_latancy:%s%4.2f%sms\toutput_token_latancy:%s%4.2f%sms\tprompt_speed:%s%5.1f%stps\toutput_speed:%s%5.1f%stps\n",
           con.yellow(), args.num_threads, con.endtag(),
           con.green(), Tensor::type_to_name(args.qtype), con.endtag(),
           con.green(), int(args.use_numa), con.endtag(), int(get_simd_size()),
           int(avg_prompt_token_num), int(avg_output_token_num), avg_prompt_latancy + avg_output_latancy,
           con.yellow(), first_token_latancy, con.endtag(),
           con.yellow(), later_token_latancy, con.endtag(),
           con.green(), 1000. / first_token_latancy, con.endtag(),
           con.green(), 1000. / later_token_latancy, con.endtag());
    return 0;
}

void Arguments::print_usage(const char* bin_name) {
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "   %s [OPTIONS]\n", bin_name);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "   --checkpoint,-c   <string>    the file path of the model checkpoint\n");
    fprintf(stderr, "   --tokenizer,-z    <string>    the file path of the tokenizer\n");
    fprintf(stderr, "   --file-type,-f    <string>    the file path of the tokenizer\n");
    fprintf(stderr, "   --mode            <string>    the running mode of the program\n");
    fprintf(stderr, "                                 - Gen: generate text for the prompt\n");
    fprintf(stderr, "                                 - Chat: chat mode\n");
    fprintf(stderr, "                                 - Benchmark: testing the performance\n");
    fprintf(stderr, "   --prompt,-i       <string>    the input prompt text\n");
    fprintf(stderr, "   --max-tokens,-n   <integer>   the maximum number of generated tokens\n");
    fprintf(stderr, "   --temperature,-t  <float>     the value for temperature sampling, [0, 1]\n");
    fprintf(stderr, "   --topp,-p         <float>     the value for top-p sampling, [0, 1]\n");
    fprintf(stderr, "   --quant,-q        <string>    quantization type, can be INT8, INT16\n");
    fprintf(stderr, "   --threads,-j      <number>    the number of threads for parallelly computing\n");
    fprintf(stderr, "   --help,-h                     print this message\n");
}

void Arguments::parse(int argc, const char** argv) {
    for (int i = 1; i < argc; ) {
        std::string_view arg = argv[i++];
        if (arg == "-j" || arg == "--threads") {
            num_threads = atoi(argv[i++]);
        } else if (arg ==  "-q" || arg == "--quant") {
            auto s = argv[i++];
            if (strcasecmp(s, "int16") == 0) {
                qtype = QuantType::INT16;
            } else if (strcasecmp(s, "int8") == 0) {
                qtype = QuantType::INT8;
            } else if (strcasecmp(s, "int4") == 0) {
                qtype = QuantType::INT4;
            }
        } else if (arg ==  "--numa") {
            use_numa = true;
        } else if (arg ==  "--uma") {
            use_numa = false;
        } else if (arg ==  "--detail") {
            print_detail = true;
        } else if (arg ==  "-c" || arg == "--checkpoint") {
            ckpt_path = argv[i++];
        } else if (arg ==  "-z" || arg == "--tokenizer") {
            tknr_path = argv[i++];
        } else if (arg ==  "-f" || arg == "--file-type") {
            auto v = argv[i++];
            if (strcasecmp(v, "flm") == 0) {
                mft = ModelFileType::FLM;
            } else if (strcasecmp(v, "gguf") == 0) {
                mft = ModelFileType::GGUF;
            } else if (strcasecmp(v, "llama2c") == 0) {
                mft = ModelFileType::LLAMA2C;
            }
        } else if (arg ==  "-i" || arg == "--prompt") {
            prompt = argv[i++];
        } else if (arg == "-n" || arg == "--max-new-tokens") {
            max_tokens = atoi(argv[i++]);
        } else if (arg ==  "-p" || arg == "--topp") {
            topp = atof(argv[i++]);
        } else if (arg ==  "-t" || arg == "--temperature") {
            temp = atof(argv[i++]);
        } else if (arg ==  "--seed") {
            seed = atoi(argv[i++]);
        } else if (arg ==  "--rounds") {
            rounds = atoi(argv[i++]);
        } else if (arg == "-m" || arg ==  "--mode") {
            auto s = argv[i++];
            if (strcasecmp(s, "gen") == 0 || strcasecmp(s, "generate") == 0) {
                mode = Mode::GEN;
            } else if (strcasecmp(s, "chat") == 0) {
                mode = Mode::CHAT;
            } else if (strcasecmp(s, "benchmark") == 0 || strcasecmp(s, "bm") == 0) {
                mode = Mode::TEST;
            }
        } else if (arg ==  "--debug") {
            is_debug = true;
            print_detail = true;
        } else if (arg ==  "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "Unknown argument:\x1b[31m%s\x1b[0m\n", arg.data());
            print_usage(argv[0]);
            exit(-1);
        }
    }
    if (rounds < 1) {
        rounds = mode == Mode::TEST ? 16 : 1;
    }
}

