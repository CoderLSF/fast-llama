/* Inference for Llama-2 Transformer model in pure C */

#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include <fstream>
#include <memory>

#include "tf_operators.h"
#include "utility.h"
#include "log.h"

// ----------------------------------------------------------------------------
// Transformer model

namespace cpuft {

static bool is_safe_piece(const char *piece) noexcept;

Tokenizer::Tokenizer() {
    auto bp = reinterpret_cast<uint8_t*>(_byte_pieces);
    for (int i = 0; i < 256; i++) {
        bp[i*2]   = static_cast<uint8_t>(i);
        bp[i*2+1] = '\0';
    }
}

Tokenizer::Tokenizer(Tokenizer&& other) {
    *this = std::move(other);
}

Tokenizer& Tokenizer::operator=(Tokenizer&& other) {
    _vocab.reset(other._vocab.release());
    _vocab_size   = other._vocab_size;
    _bos_token_id = other._bos_token_id;
    _eos_token_id = other._eos_token_id;
    _pad_token_id = other._pad_token_id;
    _underline_id = other._underline_id;
    _max_token_length = other._max_token_length;
    memcpy(_byte_pieces, other._byte_pieces, sizeof(_byte_pieces));
    _name = other._name;
    _text_data.reset(other._text_data.release());
    _text2id = std::move(other._text2id);
    _conn_tag = other._conn_tag;
    return *this;
}

bool Tokenizer::set(int vocab_size,
         std::unique_ptr<Token[]>&& vocab,
         std::unique_ptr<char[]>&& text_data,
         std::string_view conn_tag,
         int special_tokens[int(SpecialTokenType::MAX)]) noexcept {
    _vocab_size = vocab_size;
    _vocab.reset(vocab.release());
    _text_data.reset(text_data.release());
    _conn_tag = conn_tag;
    _bos_token_id = special_tokens[int(SpecialTokenType::BOS)];
    _eos_token_id = special_tokens[int(SpecialTokenType::EOS)];
    _pad_token_id = special_tokens[int(SpecialTokenType::PAD)];
    build_text2id_map();
    return true;
}

void Tokenizer::set_token_texts(const std::vector<std::string>& texts) {
    if (_conn_tag.empty()) {
        _conn_tag = "▁";
    }

    if (_vocab == nullptr) {
        _vocab.reset(new Token[texts.size()]);
    } else if (_vocab_size != int(texts.size())) {
        throw std::runtime_error("size of texts does not equal to vocab_size");
    }
    _vocab_size = int(texts.size());

    size_t buf_size = 0;
    for (auto& s : texts) {
        auto slen = (s.size() + 1 + 7) & ~7u;
        buf_size += slen;
        if (s.compare(0, _conn_tag.size(), _conn_tag.data()) == 0) {
            buf_size += slen;
        }
    }

    _text_data.reset(new char[buf_size]);
    char* buf = _text_data.get();
    for (int i = 0; i < _vocab_size; ++i) {
        auto& token = _vocab[i];
        auto& text  = texts[i];
        token.index_text = buf;
        memcpy(buf, text.c_str(), text.size());
        buf[text.size()] = '\0';

        size_t slen = text.size();
        buf += (slen + 1 + 7) & ~7u;

        if (text.compare(0, _conn_tag.size(), _conn_tag.data()) == 0) {
            token.show_text = buf;
            auto p = text.c_str() + _conn_tag.size();
            slen -= _conn_tag.size();
            buf[0] = ' ';
            memcpy(buf+1, p, slen);
            buf[slen+1] = '\0';
            buf += (slen + 2 + 7) & ~7u;
        } else {
            token.show_text = token.index_text;
        }
    }

    build_text2id_map();
}

void Tokenizer::set_token_types(std::span<const int> types) {
    _vocab_size = types.size();
    if (_vocab == nullptr) {
        _vocab.reset(new Token[_vocab_size]);
    }
    for (int i = 0; i < _vocab_size; ++i) {
        _vocab[i].type = TokenType(types[i]);
    }
}

void Tokenizer::set_token_scores(std::span<const float> scores) {
    _vocab_size = scores.size();
    if (_vocab == nullptr) {
        _vocab.reset(new Token[_vocab_size]);
    }
    for (int i = 0; i < _vocab_size; ++i) {
        _vocab[i].score = scores[i];
    }
}

void Tokenizer::build_text2id_map() {
    if (_max_token_length == 0) {
        for (int i = 0; i < _vocab_size; ++i) {
            auto& token = _vocab[i];
            auto slen = int(strlen(token.index_text));
            if (slen > _max_token_length) {
                _max_token_length = slen;
            }
        }
    }

    _text2id.reserve(size_t(_vocab_size * 3));
    for (int i = 0; i < _vocab_size; ++i) {
        _text2id[_vocab[i].index_text] = i;
    }
    if (auto it = _text2id.find(_conn_tag); it != _text2id.end()) {
        _underline_id = it->second;
    }
}

bool Tokenizer::load(std::string_view path, int vocab_size) {
    if (path.empty() || vocab_size < 1) {
        tf_log_error("Invalid parameters for tokenizer loader");
        return false;
    }

    _vocab_size = vocab_size;
    _vocab.reset(new(std::nothrow) Token[vocab_size]);
    if (_vocab == nullptr) {
        tf_log_error("Out of memory for tokenizer while loading:[%s]", path.data());
        return false;
    }

    std::ifstream file(path.data(), std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        tf_log_error("Failed to open tokenizer file %s", path.data());
        return false;
    }
    file.tellg();
    file.seekg(0, std::ios::beg);

    if (!file.read(reinterpret_cast<char*>(&_max_token_length), sizeof(_max_token_length))) {
        tf_log_error("Reading tokenizer file error:%s", path.data());
        return false;
    }

    size_t bufsize = _vocab_size * 16;
    size_t bufpos  = 0;
    _text_data.reset(new(std::nothrow) char[bufsize]);
    if (_text_data == nullptr) {
        tf_log_error("Out of memory for tokenizer");
        return false;
    }

    for (int i = 0; i < _vocab_size; i++) {
        if (!file.read(reinterpret_cast<char*>(&_vocab[i].score), sizeof(float))) {
            tf_log_error("Reading score from tokenizer file error:[%s], index:%d", path.data(), i);
            return false;
        }
        int len = 0;
        if (!file.read(reinterpret_cast<char*>(&len), sizeof(int))) {
            tf_log_error("Reading text from tokenizer file error:[%s], index:%d", path.data(), i);
            return false;
        }
        if (bufpos + len + 1 > bufsize) {
            tf_log_debug("Tokenizer text buffer going out, expanding it ...");
            size_t newsize = bufsize + 24 * (_vocab_size - i + (len+31)/32);
            std::unique_ptr<char[]> newbuf(new(std::nothrow) char[newsize]);
            if (newbuf == nullptr) {
                tf_log_error("Reading text from tokenizer file error:[%s], index:%d", path.data(), i);
                return false;
            }
            memcpy(newbuf.get(), _text_data.get(), bufpos);
            for (int j = 0; j < i; ++j) {
                _vocab[j].index_text = newbuf.get() + (_vocab[j].index_text - _text_data.get());
                _vocab[j].show_text = _vocab[j].index_text;
            }
            _text_data.reset(newbuf.release());
            bufsize = newsize;
        }
        if (!file.read(&_text_data[bufpos], len)) {
            tf_log_error("Reading token text from tokenizer file error:[%s], index:%d", path.data(), i);
            return false;
        }
        _text_data[bufpos+len] = '\0';
        _vocab[i].index_text = _text_data.get() + bufpos;
        _vocab[i].show_text  = _vocab[i].index_text;
        bufpos += size_t(len + 7) & ~7u;
    }
    build_text2id_map();
    return true;
}

int Tokenizer::search_text(std::string_view text) const noexcept {
    if (text == " ") {
        return _underline_id;
    }
    auto it = _text2id.find(text);
    if (it != _text2id.end()) {
        return it->second;
    }
    return -1;
}

int Tokenizer::encode(std::string_view text, std::span<int> output_tokens, bool add_bos, bool add_eos) const noexcept {
    if (text.empty()) {
        return 0;
    }

    if (_vocab_size < 1 || _vocab == nullptr) {
        tf_log_error("The tokenizer is not correctly initialized yet");
        return 0;
    }

    std::unique_ptr<char[]> ptr_sbuf(new char[_max_token_length * 2 + 3]);
    char* sbuf = ptr_sbuf.get();

    int n_tokens = 0;
    if (add_bos) {
        output_tokens[n_tokens++] = _bos_token_id;
    }

    if (false) {//text[0] != '\0') {
        int dummy_prefix = search_text(" ");
        if (dummy_prefix >= 0) {
            output_tokens[n_tokens++] = dummy_prefix;
        }
    }

    for (size_t i = 0, slen = 0; i < text.length(); ++i) {
        auto c = text[i];
        if ((c & 0xC0) != 0x80) {
            slen = 0;
        }

        sbuf[slen++] = c;
        sbuf[slen] = '\0';

        if ((text[i+1] & 0xC0) == 0x80 && slen < 4) {
            continue;
        }

        int id = search_text({sbuf, slen});
        if (id >= 0) {
            output_tokens[n_tokens++] = id;
        } else {
            for (size_t i = 0; i < slen; ++i) {
                output_tokens[n_tokens++] = static_cast<int>(static_cast<uint8_t>(sbuf[i]) + 3);
            }
        }
        slen = 0;
    }

    while (true) {
        float   best_score  = -1e10;
        int     best_id     = -1;
        int     best_idx    = -1;

        for (int i=0; i < n_tokens-1; i++) {
            sprintf(sbuf, "%s%s", _vocab[output_tokens[i]].index_text, _vocab[output_tokens[i+1]].index_text);
            int id = search_text(sbuf);
            if (id != -1 && _vocab[id].score > best_score) {
                best_score = _vocab[id].score;
                best_id    = id;
                best_idx   = i;
            }
        }

        if (best_idx == -1) {
            break;
        }

        output_tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < n_tokens-1; ++i) {
            output_tokens[i] = output_tokens[i+1];
        }
        --n_tokens;
    }

    if (add_eos) {
        output_tokens[n_tokens++] = _eos_token_id;
    }

    return n_tokens;
}

const char* Tokenizer::decode(int token, int prev_token) const noexcept {
    if (token < 0 || token >= _vocab_size) {
        return "";
    }
    const char* piece = _vocab[token].show_text;
    if (prev_token == 1 && piece[0] == ' ') {
        ++piece;
    }
    uint8_t byte;
    if (sscanf(piece, "<0x%02hhX>", &byte) == 1) {
        piece = &_byte_pieces[byte*2];
    }
    return is_safe_piece(piece) ? piece : "";
}

std::string_view Tokenizer::decode(std::span<const int> tokens, std::span<char> buffer) const noexcept {
    if (tokens.empty()) {
        return "";
    }

    size_t  spos = 0;
    int     prev_token = -1;
    for (auto token : tokens) {
        auto piece = decode(token, prev_token);
        int n = strlen(piece);
        if (spos + n >= buffer.size()) {
            return {buffer.data(), spos};
        }
        memcpy(buffer.data() + spos, piece, n);
        spos += n;
    }
    buffer[spos] = '\0';
    return {buffer.data(), spos};
}

bool is_safe_piece(const char *piece) noexcept {
    if (piece == nullptr || piece[0] == '\0') {
        return false;
    }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return false;
        }
    }
    return true;
}

} // namespace cpuft

