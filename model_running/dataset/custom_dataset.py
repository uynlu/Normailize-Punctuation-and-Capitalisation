import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer


IGNORE_INDEX = -100

class JointCapPuncDataset(Dataset):
    def __init__(self,
        data_path: str,
        tokenizer_name: str,
        cache_dir: str,
        max_words: int,
        max_len: int,
        overlap: int
    ):
        super(JointCapPuncDataset, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
        self.max_words = max_words
        self.max_len = max_len
        self.overlap = overlap
        self.punc_label_dict = {
            "O": 0,
            "COMMA": 1,
            "PERIOD": 2,
            "QMARK": 3
        }
        
        self.data = self._load_data(data_path)
        self.chunks = self._make_chunks()
        
    def __len__(self):
        """Get length of dataset."""
        return len(self.chunks)

    def __getitem__(self, index: int):
        """Get item by index."""
        return self.chunks[index]
    
    def _load_data(self, data_path: str):
        """Load data from file."""
        words, caps, puncs = [], [], []

        with open(data_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                parts = line.split("\t")

                if len(parts) != 3:
                    raise ValueError(
                        f"Invalid data format: {line}"
                    )

                word, cap, punc = parts

                words.append(word)
                caps.append(int(cap))
                puncs.append(self.punc_label_dict[punc])

        return list(zip(words, caps, puncs))

    def collate_fn(self, batch):
        return {
            "input_ids": pad_sequence(
                [b["input_ids"] for b in batch],
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id
            ),
            "attention_mask": pad_sequence(
                [b["attention_mask"] for b in batch],
                batch_first=True,
                padding_value=0
            ),
            "cap_labels": pad_sequence(
                [b["cap_labels"] for b in batch],
                batch_first=True,
                padding_value=IGNORE_INDEX
            ),
            "punc_labels": pad_sequence(
                [b["punc_labels"] for b in batch],
                batch_first=True,
                padding_value=IGNORE_INDEX
            )
        }
    
    def _tokenize_word(self, word: str):
        ids, attention_mask = self.tokenizer(word).values()
        if not ids:
            ids = self.tokenizer.unk_token_id
        return ids, attention_mask

    def _make_chunks(self):
        chunks = []
        budget = self.max_len - 2
        n = len(self.data)
        
        start = 0
        
        while start < n:
            sentence = ""
            input_ids = [self.tokenizer.cls_token_id]
            cap_labels = [IGNORE_INDEX]
            punc_labels = [IGNORE_INDEX]
            attention_masks = [1]

            end, used = start, 0

            while end < n and (end - start) < self.max_words:
                word, cap, punc = self.data[end]

                ids, attention_mask = self._tokenize_word(word)

                if used + len(ids) > budget and end > start:
                    break
                used += len(ids)
                end += 1

                sentence = sentence + " " + word
                input_ids.extend(ids)
                attention_masks.extend(attention_mask)
                cap_labels.extend([cap] + [IGNORE_INDEX] * (len(ids) - 1))
                punc_labels.extend([punc] + [IGNORE_INDEX] * (len(ids) - 1))
                
            input_ids.append(self.tokenizer.sep_token_id)
            cap_labels.append(IGNORE_INDEX)
            punc_labels.append(IGNORE_INDEX)
            attention_masks.append(1)

            chunks.append({
                "text": sentence,
                "input_ids": torch.tensor(input_ids[:self.max_len]),
                "attention_mask": torch.tensor(attention_masks[:self.max_len]),
                "cap_labels": torch.tensor(cap_labels[:self.max_len]),
                "punc_labels": torch.tensor(punc_labels[:self.max_len])
            })

            if end >= n:
                break
            start = max(end - self.overlap, start + 1)

        return chunks
