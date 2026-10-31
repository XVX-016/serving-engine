import sys, io, json, random, os
from transformers import AutoTokenizer

print("Loading HuggingFace tokenizer...")
# Load from local cache
tok_path = r'data\models\hf-cache\snapshots\7ae557604adf67be50417f59c2c2f167def9a775'
tokenizer = AutoTokenizer.from_pretrained(tok_path)

corpus = []

# 1. Edge cases: empty, whitespace variations, newlines
ws_cases = [
    "",
    " ",
    "  ",
    "   ",
    "    ",
    "\t",
    "\t\t",
    "\n",
    "\r\n",
    "\n\n\n",
    "\r\n\r\n",
    "   \n   \n   ",
    " \t \r\n \t ",
    "hello",
    " hello",
    "  hello",
    "   hello",
    "hello ",
    "hello  ",
    "hello\nworld",
    "hello\r\nworld",
    "  hello  world  ",
    "\n\n\nhello\n\nworld\n\n\n",
]
corpus.extend(ws_cases)

# 2. English contractions and punctuation
contr_cases = [
    "It's", "it's", "IT'S", "don't", "DON'T", "Don't",
    "I'm", "i'm", "I'M", "you're", "YOU'RE", "You're",
    "we've", "WE'VE", "We've", "they'll", "THEY'LL", "They'll",
    "he'd", "HE'D", "He'd", "who's", "what's", "that's",
    "cat's", "dogs' ", "O'Connor", "d'art",
    "!@#$%^&*()_+",
    "~`-=[]\\{}|;':\",./<>?",
    "...", "---", "--", "==", "!=", "<=", ">=", "=>", "->", "::",
    "(parentheses) [brackets] {braces} <angle>",
    "\"double quotes\" 'single quotes' `backticks`",
    "one, two, three; four: five! six? seven.",
]
corpus.extend(contr_cases)

# 3. Numbers, decimals, hex, scientific notation, dates, IPs
num_cases = [
    "0", "1", "12", "123", "1234", "12345", "1234567890",
    " 1", "  1", "   1", " 12", "  12", "   123",
    "3.1415926535", "-42", "+100", "1e-5", "6.022e23", "0xFF", "0b101010",
    "2026-10-31", "12:34:56", "192.168.1.1", "1,000,000", "$99.99", "100%",
    "2 + 2 = 4", "10 / 2 * 3 - 1 = 14",
    "²', '³', '½', '¼', '¾', '5G', '4K', '1080p', 'sm_89",
]
corpus.extend(num_cases)

# 4. Source code snippets in various languages (C++, Python, JS, SQL, Shell)
code_cases = [
    "#include <iostream>\nint main() {\n    std::cout << \"Hello, World!\\n\";\n    return 0;\n}",
    "template <typename T>\nconstexpr T pi = T(3.1415926535897932385);\n",
    "def fibonacci(n: int) -> int:\n    if n <= 1: return n\n    return fibonacci(n - 1) + fibonacci(n - 2)",
    "const fetchUsers = async () => {\n  const res = await fetch('/api/users');\n  return res.json();\n};",
    "SELECT u.id, u.name, COUNT(o.id) as total_orders FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id HAVING total_orders > 5;",
    "docker run --gpus all -v /path:/app -p 8080:8080 --rm -it vllm/vllm-openai:latest",
    "cmake_minimum_required(VERSION 3.24)\nset(CMAKE_CXX_STANDARD 17)\nadd_executable(app main.cpp)",
    "CUDA_CHECK(cudaMemcpy(d_out, h_in, size, cudaMemcpyHostToDevice));",
    "class ModelRunner(nn.Module):\n    def __init__(self, cfg):\n        super().__init__()\n        self.layers = nn.ModuleList([Layer(cfg) for _ in range(cfg.n_layers)])",
    "git commit -m \"fix(tokenizer): resolve regex unicode range table\"",
]
corpus.extend(code_cases)

# 5. CJK (Chinese, Japanese, Korean)
cjk_cases = [
    "你好，世界！",
    "通义千问是阿里巴巴推出的大语言模型系列。",
    "天地玄黄，宇宙洪荒。日月盈昃，辰宿列张。",
    "深度学习模型推理加速与系统优化。",
    "こんにちは世界！初めまして、よろしくお願いします。",
    "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。",
    "東京特許許可局局長。",
    "안녕하세요, 세계! 만나서 반갑습니다.",
    "인공지능과 머신러닝의 미래 기술 동향 보고서.",
    "훈민정음 어제 서문: 나랏말싸미 듕귁에 달아 문자와로 서르 사맛디 아니할쎄.",
    "混合语言: Hello 你好 World 世界 123 ! こんにちは",
    "中英混合：Qwen2.5-0.5B 是一个轻量级但能力强大的开源语言模型。",
    "中文标点：《》，。！？；：“”‘’【】……—～",
]
corpus.extend(cjk_cases)

# 6. Emojis and complex Unicode symbols
emoji_cases = [
    "😀 😃 😄 😁 😆 😅 🤣 😂",
    "🚀 🌟 🔥 💡 💻 ⚙️ ⚡ 🛠️",
    "👍 👎 👏 🙌 🤝 ✌️ 🤞 🤙",
    "🎉 🎊 🎈 🎁 🏆 🥇 🎯 🎲",
    "❤️ 🧡 💛 💚 💙 💜 🖤 🤍",
    "Family: 👨‍👩‍👧‍👦",
    "Flag: 🇺🇸 🇬🇧 🇨🇳 🇯🇵 🇩🇪 🇫🇷",
    "Math: ∀x ∈ ℝ, ∃y : y > x ∧ ∑_{i=1}^n i = n(n+1)/2",
    "Greek: α β γ δ ε ζ η θ ι κ λ μ ν ξ ο π ρ σ τ υ φ χ ψ ω",
    "Cyrillic: Съешь же ещё этих мягких французских булок, да выпей чаю.",
    "Arabic: السلام عليكم ورحمة الله وبركاته",
    "Hebrew: שלום עולם, ברוכים הבאים",
    "Hindi: नमस्ते दुनिया, आप कैसे हैं?",
    "Emoji mixed: Qwen is on fire! 🔥🚀 Let's code some CUDA kernels 💻⚡",
]
corpus.extend(emoji_cases)

# 7. Special tokens and added tokens
special_cases = [
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n4<|im_end|>",
    "<|object_ref_start|>test<|object_ref_end|>",
    "<|box_start|>[10, 20, 30, 40]<|box_end|>",
    "<|vision_start|><|image_pad|><|vision_end|>",
    "<tool_call>{\"name\": \"get_weather\"}</tool_call>",
    "<|fim_prefix|>def add(a, b):<|fim_suffix|>return a + b<|fim_middle|>\n    ",
]
corpus.extend(special_cases)

# 8. Chat templates
chat_dialogs = [
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
    [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": "Write a python function to compute factorial."},
        {"role": "assistant", "content": "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)"},
        {"role": "user", "content": "Can you make it iterative?"},
    ],
    [
        {"role": "user", "content": "Explain KV cache in 2 sentences."},
    ],
    [
        {"role": "system", "content": "Be extremely concise."},
        {"role": "user", "content": "Is sm_89 Ada Lovelace?"},
        {"role": "assistant", "content": "Yes."},
    ]
]

for d in chat_dialogs:
    templated = tokenizer.apply_chat_template(d, tokenize=False, add_generation_prompt=True)
    corpus.append(templated)
    templated_no_gen = tokenizer.apply_chat_template(d, tokenize=False, add_generation_prompt=False)
    corpus.append(templated_no_gen)

# 9. Long texts and prose
long_cases = [
    "The Attention mechanism was introduced by Vaswani et al. in the landmark 2017 paper 'Attention Is All You Need'. It replaces recurrent and convolutional layers with multi-head self-attention, allowing representations to be built in parallel across all positions.",
    "In autoregressive large language model serving, inference proceeds in two distinct phases: prefill and decode. Prefill processes the prompt sequence in parallel, computing keys and values and storing them in the KV cache. Decode then advances one token at a time, querying the entire history stored in the cache.",
    "Ada Lovelace architecture introduces 4th generation Tensor Cores supporting FP8 formats (E4M3 and E5M2) along with asynchronous memory copies, improved L2 cache capacity, and high-frequency SM execution profiles.",
]
corpus.extend(long_cases)

# 10. Procedurally generated permutations to reach >= 350 items
vocab_words = ["apple", "banana", "compiler", "transformer", "matrix", "tensor", "CUDA", "kernel", "thread", "warp", "block"]
puncs = [".", ",", "!", "?", " - ", " / ", " :: "]
random.seed(1337)
while len(corpus) < 360:
    k = random.randint(3, 12)
    s = " ".join(random.choices(vocab_words, k=k))
    if random.random() < 0.3:
        s += random.choice(puncs)
    if random.random() < 0.2:
        s = "  " + s + "\n"
    corpus.append(s)

print(f"Total corpus items: {len(corpus)}")

# Generate ground truth
items = []
for idx, text in enumerate(corpus):
    # Encode with HF
    hf_ids = tokenizer.encode(text)
    hf_decoded = tokenizer.decode(hf_ids, skip_special_tokens=False)
    hf_decoded_skip = tokenizer.decode(hf_ids, skip_special_tokens=True)
    items.append({
        "id": idx,
        "text": text,
        "token_ids": hf_ids,
        "decoded": hf_decoded,
        "decoded_skip_special": hf_decoded_skip
    })

os.makedirs("data", exist_ok=True)
out_file = "data/tokenizer_corpus.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(items, f, ensure_ascii=False, indent=2)

print(f"Saved {len(items)} items to {out_file} successfully.")
