import regex as re
import json
from collections import Counter
from tqdm import tqdm  # 用于显示进度条
import os
from typing import BinaryIO, List, Dict, Tuple, Iterable, Iterator 
import multiprocessing

# [Source: 155] GPT-2 的预分词正则模式
# 这个正则的作用是把文本切分成单词、标点、缩写等基础单元，避免跨单词合并。
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def _process_chunk(args):
    """
    单个进程的工作函数：读取指定范围的文件，进行预分词，统计词频。
    """
    filename, start, end, special_token_str = args
    
    # 每个进程独立打开文件，避免文件指针冲突
    with open(filename, 'rb') as f:
        f.seek(start)
        # 只读取属于当前块的字节
        bytes_chunk = f.read(end - start)
    
    # 解码为字符串，忽略解码错误（因为边界是安全的，通常不会有错误）
    text_chunk = bytes_chunk.decode("utf-8", errors="ignore")
    
    # 编译正则
    gpt2_pat = re.compile(GPT2_SPLIT_PATTERN)
    local_counts = Counter()
    
    # [cite: 213] 处理特殊 token：先按特殊 token 切分，再对每一部分进行正则匹配
    # 如果 special_token_str 是 "<|endoftext|>"
    # 注意：这里我们不需要在统计中包含特殊 token 本身，因为它们稍后会直接加入词表
    
    if special_token_str:
        # 使用特殊 token 作为分隔符切分
        # re.escape 确保特殊字符被正确处理
        splits = re.split(re.escape(special_token_str), text_chunk)
    else:
        splits = [text_chunk]
        
    for sub_text in splits:
        if not sub_text.strip():
            continue
            
        # 对非特殊 token 的部分应用 GPT-2 正则
        words = gpt2_pat.findall(sub_text)
        
        # 统计字节形式的单词
        for w in words:
            local_counts[w.encode('utf-8')] += 1
            
    return local_counts

def get_stats(ids_list: List[List[int]], counts: Dict[int, int] = None) -> Dict[Tuple[int, int], int]:
    """
    统计当前所有单词中，相邻字节对（pair）出现的频率。
    
    Args:
        ids_list: 单词列表，每个单词已经被转换成了整数列表 (如 [104, 101, ...])
        counts: (可选) 每个单词在语料库中出现的频次，用于加权统计
    Returns:
        pair_counts: 字典 {(token_id1, token_id2): frequency}
    """
    stats = Counter()
    for idx, ids in enumerate(ids_list):
        # 如果提供了counts，说明是加权统计（更快）；否则默认权重为1
        weight = counts[idx] if counts else 1
        
        # 遍历单词中的每一个相邻对
        for pair in zip(ids, ids[1:]):
            stats[pair] += weight
    return stats

def merge_vocab(pair: Tuple[int, int], idx: int, ids_list: List[List[int]]) -> List[List[int]]:
    """
    执行合并操作：将 ids_list 中所有的 `pair` 替换为新的 `idx`。
    
    Args:
        pair: 要合并的 ID 对，例如 (104, 101) ('h', 'e')
        idx: 新生成的 Token ID
        ids_list: 当前的单词列表
    """
    new_ids_list = []
    p0, p1 = pair
    
    for ids in ids_list:
        new_ids = []
        i = 0
        while i < len(ids):
            # 如果发现当前位置和下一位置匹配要合并的 pair
            if i < len(ids) - 1 and ids[i] == p0 and ids[i+1] == p1:
                new_ids.append(idx) # 替换为新 ID
                i += 2              # 跳过下个字符
            else:
                new_ids.append(ids[i])
                i += 1
        new_ids_list.append(new_ids)
    return new_ids_list

def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str]) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    print(f"Finding chunk boundaries for {input_path}...")
    
    # 1. 动态决定进程数
    file_size = os.path.getsize(input_path)
    if file_size < 10 * 1024 * 1024:  # < 10MB 单进程
        num_processes = 1
    else:
        num_processes = min(multiprocessing.cpu_count(), 8)
    
    # 2. 预分词
    primary_special_token = special_tokens[0].encode('utf-8') if special_tokens else b'\n'
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, primary_special_token)
    
    tasks = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        tasks.append((input_path, start, end, special_tokens[0] if special_tokens else None))
    
    global_word_counts = Counter()
    
    if num_processes == 1:
        for task in tqdm(tasks, desc="Pre-tokenizing (Serial)"):
            global_word_counts.update(_process_chunk(task))
    else:
        with multiprocessing.Pool(processes=len(tasks)) as pool:
            for local_counts in tqdm(pool.imap_unordered(_process_chunk, tasks), total=len(tasks), desc="Pre-tokenizing (Parallel)"):
                global_word_counts.update(local_counts)

    # -------------------------------------------------------
    # 核心优化：构建索引以加速 BPE Merge
    # -------------------------------------------------------
    
    sorted_words = sorted(global_word_counts.keys())
    ids_list = [list(w) for w in sorted_words]
    word_freqs = [global_word_counts[w] for w in sorted_words]
    
    vocab = {i: bytes([i]) for i in range(256)}
    num_merges = vocab_size - 256 - len(special_tokens)
    merges = []
    next_id = 256

    # 初始化索引
    stats = Counter()
    pair_to_word_indices = {} 

    for idx, (ids, freq) in enumerate(zip(ids_list, word_freqs)):
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i+1])
            stats[pair] += freq
            if pair not in pair_to_word_indices:
                pair_to_word_indices[pair] = set()
            pair_to_word_indices[pair].add(idx)

    print(f"Training BPE (Optimized, Target merges: {num_merges})...")
    
    for _ in tqdm(range(num_merges)):
        if not stats:
            break

        # Tie-breaking: 频率高优先 -> 字典序大优先（元组比较）
        best_pair = max(stats, key=lambda x: (stats[x], vocab[x[0]], vocab[x[1]]))
        
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
        new_token_bytes = vocab[best_pair[0]] + vocab[best_pair[1]]
        vocab[next_id] = new_token_bytes
        
        if best_pair not in pair_to_word_indices:
            continue
            
        # ==================== 核心修复点 ====================
        # 使用 list() 创建副本，避免 "Set changed size during iteration" 错误
        indices_to_update = list(pair_to_word_indices[best_pair])
        # ==================================================
        
        p0, p1 = best_pair
        
        for idx in indices_to_update:
            ids = ids_list[idx]
            freq = word_freqs[idx]
            
            # 先移除旧 pair 的统计
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i+1])
                stats[pair] -= freq
                if pair in pair_to_word_indices and idx in pair_to_word_indices[pair]:
                    pair_to_word_indices[pair].remove(idx)
            
            # 执行合并
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == p0 and ids[i+1] == p1:
                    new_ids.append(next_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids_list[idx] = new_ids
            
            # 添加新 pair 的统计
            for i in range(len(new_ids) - 1):
                pair = (new_ids[i], new_ids[i+1])
                stats[pair] += freq
                if pair not in pair_to_word_indices:
                    pair_to_word_indices[pair] = set()
                pair_to_word_indices[pair].add(idx)

        # 清理
        if stats[best_pair] == 0:
            del stats[best_pair]
            
        next_id += 1

    for st in special_tokens:
        vocab[next_id] = st.encode('utf-8')
        next_id += 1
        
    return vocab, merges

class BPE_Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: List[str] = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        
        # 建立反向查找表：bytes -> id
        self.token_to_id = {v: k for k, v in vocab.items()}
        
        # 建立 merge 查找表：(bytes, bytes) -> rank (越小越优先)
        # [Source: 274] Apply merges in the same order of creation
        self.merges_rank = {pair: i for i, pair in enumerate(merges)}
        
        # 特殊 token 的 ID 映射
        self.special_token_ids = {}
        for st in self.special_tokens:
            st_bytes = st.encode('utf-8')
            if st_bytes in self.token_to_id:
                self.special_token_ids[st] = self.token_to_id[st_bytes]

        # 编译正则
        self.gpt2_pat = re.compile(GPT2_SPLIT_PATTERN)

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: List[str] = None):
        # [Source: 307] 从文件加载
        # 这里假设保存格式为 JSON 或者 pickle，根据你的保存方式调整
        # 这里演示 JSON 加载（注意 JSON key 只能是 string，需要转回 int）
        with open(vocab_filepath, 'r') as f:
            # JSON 存的时候 key 是 str, value 是 latin-1 字符串 (如果是 bytes)
            # 实际作业中建议用 pickle 保存更方便，或者手动解析
            # 这里简化为假设已正确加载数据结构
            pass 
        # 此处省略具体文件读取代码，因为这取决于你 train_bpe 怎么存的
        pass

    def _bpe_merge(self, token_bytes: List[bytes]) -> List[bytes]:
        """
        对单个单词的字节列表应用 BPE 合并。
        使用贪心策略：每轮找到 rank 最小（最早创建）的 pair 进行合并。
        """
        while len(token_bytes) >= 2:
            stats = {}
            # 找出当前所有可能的 pair
            for i in range(len(token_bytes) - 1):
                pair = (token_bytes[i], token_bytes[i+1])
                if pair in self.merges_rank:
                    stats[pair] = self.merges_rank[pair]
            
            if not stats:
                break # 没有可合并的了
            
            # 找到 rank 最小的 pair (最早 merge 的那个)
            best_pair = min(stats, key=stats.get)
            
            # 执行合并
            new_tokens = []
            i = 0
            while i < len(token_bytes):
                if i < len(token_bytes) - 1 and (token_bytes[i], token_bytes[i+1]) == best_pair:
                    # 合并为拼接后的 bytes
                    new_tokens.append(token_bytes[i] + token_bytes[i+1])
                    i += 2
                else:
                    new_tokens.append(token_bytes[i])
                    i += 1
            token_bytes = new_tokens
            
        return token_bytes

    def encode(self, text: str) -> List[int]:
        """
        [Source: 312] 将文本编码为 ID 列表
        """
        ids = []
        
        # 1. 处理特殊 Token
        if self.special_tokens:
            # ==================== 核心修复 ====================
            # 必须按长度降序排序！确保 "<|endoftext|><|endoftext|>" 在 "<|endoftext|>" 之前被匹配
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            
            pattern = "(" + "|".join(re.escape(k) for k in sorted_special_tokens) + ")"
            # ================================================
            
            chunks = re.split(pattern, text)
        else:
            chunks = [text]

        for chunk in chunks:
            # 如果是特殊 token，直接查表
            if chunk in self.special_tokens:
                ids.append(self.special_token_ids[chunk])
                continue
            
            if not chunk:
                continue

            # 2. GPT-2 正则切分
            words = self.gpt2_pat.findall(chunk)
            
            for word in words:
                # 3. 转为初始字节列表
                word_bytes = [bytes([b]) for b in word.encode('utf-8')]
                
                # 4. 应用 BPE 合并
                merged_bytes = self._bpe_merge(word_bytes)
                
                # 5. 映射为 ID
                for b in merged_bytes:
                    if b in self.token_to_id:
                        ids.append(self.token_to_id[b])
                    else:
                        print(f"Warning: Unknown token {b}")

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        [Source: 313] Given an iterable of strings (e.g., a Python file handle), 
        return a generator that lazily yields token IDs.
        
        Args:
            iterable: 一个产生字符串的迭代器（例如 open('file.txt', 'r')）
        
        Yields:
            int: Token ID
        """
        for chunk in iterable:
            # 这里的 chunk 可能是文件的一行，或者一部分文本
            # 我们复用 encode 方法处理这一小块
            token_ids = self.encode(chunk)
            
            # 使用 yield from 逐个返回 ID，而不是构建一个巨大的列表
            yield from token_ids

    def decode(self, ids: List[int]) -> str:
        """
        [Source: 315] 将 ID 列表解码为文本
        """
        byte_buffer = b""
        
        for idx in ids:
            if idx in self.vocab:
                byte_buffer += self.vocab[idx]
            else:
                # 处理未知的 ID (防御性编程)
                pass
        
        # [Source: 294] 使用 errors='replace' 处理无效的 Unicode 序列
        text = byte_buffer.decode("utf-8", errors="replace")
        return text