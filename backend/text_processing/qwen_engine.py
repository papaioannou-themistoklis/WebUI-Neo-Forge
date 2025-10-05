import torch

from backend import memory_management
from backend.text_processing import emphasis, parsing
from modules.shared import opts


class PromptChunk:
    def __init__(self):
        self.tokens = []
        self.multipliers = []


class QwenTextProcessingEngine:
    def __init__(self, text_encoder, tokenizer):
        super().__init__()

        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

        self.emphasis = emphasis.get_current_option(opts.emphasis)()
        self.max_length = 99999999
        self.min_length = 1
        self.id_pad = 151643

        self.llama_template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

    def tokenize(self, texts):
        llama_texts = [self.llama_template.format(text) for text in texts]
        return self.tokenizer(llama_texts)["input_ids"]

    def encode_with_transformers(self, tokens):
        device = memory_management.text_encoder_device()
        tokens = tokens.to(device)
        self.text_encoder.to(device=device)
        return self.text_encoder(x=tokens)

    def tokenize_line(self, line):
        parsed = parsing.parse_prompt_attention(line, self.emphasis.name)

        tokenized = self.tokenize([text for text, _ in parsed])

        chunks = []
        chunk = PromptChunk()
        token_count = 0

        def next_chunk():
            nonlocal token_count
            nonlocal chunk

            current_chunk_length = len(chunk.tokens)
            token_count += current_chunk_length
            remaining_count = self.min_length - current_chunk_length

            if self.min_length > 0 and remaining_count > 0:
                chunk.tokens += [self.id_pad] * remaining_count
                chunk.multipliers += [1.0] * remaining_count

            chunks.append(chunk)
            chunk = PromptChunk()

        for tokens, (text, weight) in zip(tokenized, parsed):
            if text == "BREAK" and weight == -1:
                next_chunk()
                continue

            position = 0
            while position < len(tokens):
                token = tokens[position]
                chunk.tokens.append(token)
                chunk.multipliers.append(weight)
                position += 1

        if chunk.tokens or not chunks:
            next_chunk()

        return chunks, token_count

    def __call__(self, texts):
        zs = []
        cache = {}

        self.emphasis = emphasis.get_current_option(opts.emphasis)()

        for line in texts:
            if line in cache:
                line_z_values = cache[line]
            else:
                chunks, token_count = self.tokenize_line(line)
                line_z_values = []

                #   pad all chunks to length of longest chunk
                max_tokens = 0
                for chunk in chunks:
                    max_tokens = max(len(chunk.tokens), max_tokens)

                for chunk in chunks:
                    tokens = chunk.tokens
                    multipliers = chunk.multipliers

                    remaining_count = max_tokens - len(tokens)
                    if remaining_count > 0:
                        tokens += [self.id_pad] * remaining_count
                        multipliers += [1.0] * remaining_count

                    z = self.process_tokens([tokens], [multipliers])[0]
                    z = self.postprocess_tokens(z, tokens)
                    line_z_values.append(z)
                cache[line] = line_z_values

            zs.extend(line_z_values)

        return torch.stack(zs)

    def postprocess_tokens(self, out, tokens):
        """strip the llama_template"""
        template_end = 0
        count_im_start = 0

        for i, v in enumerate(tokens):
            elem = int(v)
            if elem == 151644 and count_im_start < 2:
                template_end = i
                count_im_start += 1

        if out.shape[1] > (template_end + 3):
            if int(tokens[template_end + 1]) == 872:
                if int(tokens[template_end + 2]) == 198:
                    template_end += 3

        return out[template_end:]

    def process_tokens(self, batch_tokens, batch_multipliers):
        tokens = torch.asarray(batch_tokens)

        z, _ = self.encode_with_transformers(tokens)

        self.emphasis.tokens = batch_tokens
        self.emphasis.multipliers = torch.asarray(batch_multipliers).to(z)
        self.emphasis.z = z
        self.emphasis.after_transformers()
        z = self.emphasis.z

        return z
