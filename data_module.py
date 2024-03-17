import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import datasets
import os
from utils import get_model_identifiers_from_yaml

def encode_and_return(tokenizer, data, max_length, task = 'qa'):
    encoded = tokenizer(
                data, 
                add_special_tokens=True, 
                max_length=max_length, 
                truncation=True
            )
    # padding = 'max_length'
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    # pad_input_ids = encoded['input_ids']
    # pad_attention_mask = encoded['attention_mask']
    if task == 'qa':
        if len(encoded.input_ids) == max_length:
            label = encoded.input_ids
        else:
            label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)
    else:
        label = pad_input_ids
        # label = encoded['input_ids']

    return pad_input_ids, label, pad_attention_mask

def convert_raw_data_to_model_format(tokenizer, max_length,  question, answer, model_configs):
    question_start_token, question_end_token, answer_token = model_configs['question_start_tag'], model_configs['question_end_tag'], model_configs['answer_tag']
    new_question = question_start_token + question + question_end_token
    new_answer = answer_token + answer
    full_text = new_question + new_answer
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))
    
    pad_input_ids, label, pad_attention_mask = encode_and_return(tokenizer, full_text, max_length)
    #change label to -100 for question tokens
    try:
        for i in range(num_question_tokens): label[i] = -100
    except IndexError:
        pass

    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)
    
def prepare_data_textgen(tokenizer, max_length, data):  
    pad_input_ids, label, pad_attention_mask = encode_and_return(tokenizer, data, max_length, task = 'gen')
    #change label to -100 for question tokens

    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)

class TextGenerationDataset(Dataset):
    def __init__(self, data_path, tokenizer, model_family,  max_length=512, split = "forget", loss_type="idk"):
        super(TextGenerationDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # print(data_path)
        if 'wikimia' in data_path.lower():
            LENGTH = 64
            self.forget_data = datasets.load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{LENGTH}")
            indices = torch.where(self.forget_data['label'] == 1)[0]
            self.forget_data = torch.utils.data.Subset(self.forget_data, indices)
        elif 'wikitext' in data_path.lower():
            self.forget_data = datasets.load_dataset('wikitext', 'wikitext-2-v1')
            self.forget_data = self.forget_data['train']['text']
            # self.forget_data = self.forget_data[:int(0.50*len(self.forget_data))]
        elif 'bookcorpus' in data_path.lower():
            self.forget_data = datasets.load_dataset('text', data_files = {'train': '/scratch/deu9yh/llm_privacy/tofu/dataset/books_large_p2.txt'})    
            self.forget_data = self.forget_data['train']['text']
            self.forget_data = self.forget_data[:10000]
        elif 'hface' in data_path.lower():
            self.forget_data = datasets.load_dataset(data_path.split('hface')[0])
            self.forget_data = self.forget_data[split]['text']
            self.forget_data = self.forget_data[:5000]
        else:
            self.forget_data = datasets.load_dataset('json', data_files={'train': os.path.join(data_path, split+".json")})
            self.forget_data = self.forget_data['train']['text']
        # retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
        
        if 'wikitext' in data_path.lower():
            self.retain_data = datasets.load_dataset('text', data_files = {'train': '/scratch/deu9yh/llm_privacy/tofu/dataset/books_large_p2.txt'})
        elif 'bookcorpus' in data_path.lower():
            self.retain_data = datasets.load_dataset('wikitext', 'wikitext-2-v1')   
        elif 'hface' in data_path.lower():
            # wikitext = datasets.load_dataset(path = 'wikitext', name='wikitext-2-v1', split='train')
            wikitext = datasets.load_dataset("wikitext", "wikitext-103-v1")
            wikitext = wikitext['train']['text']
            wikitext = wikitext[:int(0.20*len(wikitext))]
            books = datasets.load_dataset('text', data_files = {'train': '/scratch/deu9yh/llm_privacy/tofu/dataset/books_large_p2.txt'})
            books = books['train']['text']
            books = books[:int(0.20*len(books))]
            self.retain_data = wikitext + books
        else:
            retain_split = split.replace("forget", "retain")
            # self.retain_data =datasets.load_dataset(data_path, retain_split)["train"]
            self.retain_data = datasets.load_dataset('json', data_files={'train': os.path.join(data_path, retain_split+".json")})
        
        if not 'hface' in data_path.lower(): 
            self.retain_data = self.retain_data['train']['text']
            self.retain_data = self.retain_data[:int(0.30*len(self.retain_data))]
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.loss_type = loss_type

        if self.loss_type == "idk":
            self.split1, self.split2 = "idk", "retain"
            self.idontknowfile = "data/idontknow.jsonl"
            self.idk = open(self.idontknowfile, "r").readlines()
        else:
            self.split1, self.split2 = "forget", "retain"

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []
        for data_type in [self.split1, self.split2]:
            #use questions from forget set if split is idk or forget
            data = self.retain_data if data_type == "retain" else self.forget_data
            idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data) 
            
            # pad_input_ids, label, pad_attention_mask = encode_and_return(self.tokenizer, data[idx], self.max_length)
        
            if data_type == "idk":
                #get a random answer position from idk
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                answer = self.idk[rand_pos].strip()
                
            converted_data = prepare_data_textgen(self.tokenizer, self.max_length, data[idx])
            rets.append(converted_data)
            # rets.append((torch.tensor(pad_input_ids), torch.tensor(label), torch.tensor(pad_attention_mask)))
        return rets

class TextGenerationDatasetFromJSON(Dataset):
    def __init__(self, data_path, tokenizer, model_family,  max_length=512, split = "forget", loss_type="idk"):
        super(TextGenerationDatasetFromJSON, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # print(data_path)
        if 'wikimia' in data_path.lower():
            LENGTH = 64
            self.forget_data = datasets.load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{LENGTH}")
            indices = torch.where(self.forget_data['label'] == 1)[0]
            self.forget_data = torch.utils.data.Subset(self.forget_data, indices)
        elif 'wikitext' in data_path.lower():
            self.forget_data = datasets.load_dataset('wikitext', 'wikitext-2-v1')
            self.forget_data = self.forget_data['train']['text']
            # self.forget_data = self.forget_data[:int(0.50*len(self.forget_data))]
        elif 'bookcorpus' in data_path.lower():
            self.forget_data = datasets.load_dataset('text', data_files = {'train': '/scratch/deu9yh/llm_privacy/tofu/dataset/books_large_p2.txt'})    
            self.forget_data = self.forget_data['train']['text']
            self.forget_data = self.forget_data[:15000]
        else:
            self.forget_data = datasets.load_dataset('json', data_files={'train': os.path.join(data_path, split+".json")})
            self.forget_data = self.forget_data['train']['text']
        # retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
        
        if 'wikitext' in data_path.lower():
            self.retain_data = datasets.load_dataset('text', data_files = {'train': '/scratch/deu9yh/llm_privacy/tofu/dataset/books_large_p2.txt'})
        elif 'bookcorpus' in data_path.lower():
            self.retain_data = datasets.load_dataset('wikitext', 'wikitext-2-v1')   
        else:
            retain_split = split.replace("forget", "retain")
            # self.retain_data =datasets.load_dataset(data_path, retain_split)["train"]
            self.retain_data = datasets.load_dataset('json', data_files={'train': os.path.join(data_path, retain_split+".json")})
        
        self.retain_data = self.retain_data['train']['text']
        self.retain_data = self.retain_data[:int(0.30*len(self.retain_data))]
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.loss_type = loss_type

        if self.loss_type == "idk":
            self.split1, self.split2 = "idk", "retain"
            self.idontknowfile = "data/idontknow.jsonl"
            self.idk = open(self.idontknowfile, "r").readlines()
        else:
            self.split1, self.split2 = "forget", "retain"

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []
        for data_type in [self.split1, self.split2]:
            #use questions from forget set if split is idk or forget
            data = self.retain_data if data_type == "retain" else self.forget_data
            idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data) 
            data_tokens = data[idx].split()
            question = " ".join(data_tokens[:int(0.10*len(data_tokens))])
            answer = " ".join(data_tokens[int(0.10*len(data_tokens)):])

            # pad_input_ids, label, pad_attention_mask = encode_and_return(self.tokenizer, data[idx], self.max_length)
        
            if data_type == "idk":
                #get a random answer position from idk
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                answer = self.idk[rand_pos].strip()
                
            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            rets.append(converted_data)
            # rets.append((torch.tensor(pad_input_ids), torch.tensor(label), torch.tensor(pad_attention_mask)))
        return rets



class TextForgetDatasetQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family,  max_length=512, split = "forget10", loss_type="idk"):
        super(TextForgetDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # self.forget_data = datasets.load_dataset(data_path, split)["train"]
        self.forget_data = datasets.load_dataset('json', data_files={'train': os.path.join(data_path, split+".json")})
        self.forget_data = self.forget_data['train']['text']
        retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
        # retain_split = split.replace("forget", "retain")
        # self.retain_data =datasets.load_dataset(data_path, retain_split)["train"]
        self.retain_data = datasets.load_dataset('json', data_files={'train': os.path.join(data_path, retain_split+".json")})
        self.retain_data = self.retain_data['train']['text']
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.loss_type = loss_type

        if self.loss_type == "idk":
            self.split1, self.split2 = "idk", "retain"
            self.idontknowfile = "data/idontknow.jsonl"
            self.idk = open(self.idontknowfile, "r").readlines()
        else:
            self.split1, self.split2 = "forget", "retain"

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []
        for data_type in [self.split1, self.split2]:
            #use questions from forget set if split is idk or forget
            data = self.retain_data if data_type == "retain" else self.forget_data
            idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
            question = data[idx]['question']
            answer = data[idx]['answer']

            # pad_input_ids, label, pad_attention_mask = encode_and_return(self.tokenizer, data[idx], self.max_length)
        
            if data_type == "idk":
                #get a random answer position from idk
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                answer = self.idk[rand_pos].strip()
                
            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            rets.append(converted_data)
            # rets.append((torch.tensor(pad_input_ids), torch.tensor(label), torch.tensor(pad_attention_mask)))
        return rets


class TextForgetDatasetDPOQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split = "forget10", ):
        super(TextForgetDatasetDPOQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # self.forget_data = datasets.load_dataset(data_path, split)["train"]
        self.forget_data = datasets.load_dataset('json', data_files={'train': os.path.join(data_path, split+".json")})
        self.forget_data = self.forget_data['train']
        self.idontknowfile = "llm_privacy/tofu/data/idontknow.jsonl"
        self.idk = open(self.idontknowfile, "r").readlines()
        retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
        # self.retain_data = datasets.load_dataset(data_path, retain_split)["train"]
        self.retain_data = datasets.load_dataset('json', data_files={'train': os.path.join(data_path, retain_split+".json")})
        self.retain_data = self.retain_data['train']
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []

        for data_type in ["idk", "forget", "retain"]:
            data = self.forget_data if data_type != "retain" else self.retain_data
            idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
            
            question = data[idx]['question']
            
            if data_type != "idk":
                answer = data[idx]['answer']
            else:
                #get a random position from idk
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                answer = self.idk[rand_pos].strip()

            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            rets.append(converted_data)
        return rets

class TextGenDataset(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split = None):
        super(TextGenDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_family = model_family
        # self.data = datasets.load_dataset(data_path, split)["train"]
        if 'wikimia' in data_path.lower():
            LENGTH = 64
            self.data = datasets.load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{LENGTH}")
            indices = torch.where(self.data['label'] == 1)[0]
            self.data = torch.utils.data.Subset(self.data, indices)
        elif 'wikitext' in data_path.lower():
            self.data = datasets.load_dataset('wikitext', 'wikitext-2-v1')
            self.data = self.data['train']['text']
            # self.data = self.data[:int(0.50*len(self.data))]
        elif 'bookcorpus' in data_path.lower():
            self.data = datasets.load_dataset('text', data_files = {'train': '/scratch/deu9yh/llm_privacy/tofu/dataset/books_large_p2.txt'})    
            self.data = self.data['train']['text'][:int(0.20*len(self.data))]
        elif 'hface' in data_path.lower():
            self.data = datasets.load_dataset(data_path.split('hface')[0])
            self.data = self.data[split]['text']
            self.data = self.data[:10000]
        else:
            self.data = datasets.load_dataset('json', data_files={'train': os.path.join(data_path, split+".json")})
            self.data = self.data['train']['text']
        
        # self.data = self.data['train']
        
        self.model_configs = get_model_identifiers_from_yaml(model_family)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(self.data[idx], str):
            answers = [self.data[idx]]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = prepare_data_textgen(self.tokenizer, self.max_length, answer)
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])


        return torch.stack(pad_input_ids_list).squeeze(),\
                torch.stack(label_list).squeeze(),\
                torch.stack(pad_attention_mask_list).squeeze()


class TextDatasetQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split = None, question_key='question', answer_key='answer'):
        super(TextDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_family = model_family
        # self.data = datasets.load_dataset(data_path, split)["train"]
        if 'wikimia' in data_path.lower():
            LENGTH = 64
            self.data = datasets.load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{LENGTH}")
            indices = torch.where(self.data['label'] == 1)[0]
            self.data = torch.utils.data.Subset(self.data, indices)
        elif 'wikitext' in data_path.lower():
            self.data = datasets.load_dataset('wikitext', 'wikitext-2-v1')
            self.data = self.data['train']['text']
            # self.data = self.data[:int(0.50*len(self.data))]
        elif 'bookcorpus' in data_path.lower():
            self.data = datasets.load_dataset('text', data_files = {'train': '/scratch/deu9yh/llm_privacy/tofu/dataset/books_large_p2.txt'})    
            self.data = self.data['train']['text'][:int(0.20*len(self.data))]
        else:
            self.data = datasets.load_dataset('json', data_files={'train': os.path.join(data_path, split+".json")})
            self.data = self.data['train']['text']
        
        # self.data = self.data['train']
        
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if 'gpt' in self.model_family:
            data_tokens = self.data[idx].split()
            question = " ".join(data_tokens[:int(0.10*len(data_tokens))])
            answers = " ".join(data_tokens[int(0.10*len(data_tokens)):])
        else:
            question = self.data[idx][self.qk]
            answers = self.data[idx][self.ak]

        # print(question, answers, sep = '\n\n')
        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])


        return torch.stack(pad_input_ids_list).squeeze(),\
                torch.stack(label_list).squeeze(),\
                torch.stack(pad_attention_mask_list).squeeze()


def collate_fn(batch):
    input_ids, attention_masks = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=-100)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    return input_ids, attention_masks

def custom_data_collator_forget(samples):
    forget_samples, retain_samples = [sample[0] for sample in samples], [sample[1] for sample in samples]
    rets = []
    for data_type in ["forget", "retain"]:
        data = forget_samples if data_type == "forget" else retain_samples
        input_ids = [s[0] for s in data]
        labels = [s[1] for s in data]
        attention_mask = [s[2] for s in data]
        rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))
    return rets

def custom_data_collator(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)


def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)
    return loss
