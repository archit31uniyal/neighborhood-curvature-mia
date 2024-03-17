from run_mia_unified import get_args, get_config_from_yaml, get_model_identifiers_from_yaml, strip_newlines
import custom_datasets
from data_module import get_batch_loss, TextDatasetQA, custom_data_collator, TextGenerationDataset, TextGenDataset
import torch
import datasets
import random
import numpy as np
import transformers
import time
from tqdm import tqdm

def load_base_model_and_tokenizer(name, args):
    if args.openai_model is None:
        print(f'Loading BASE model {name}...')
        base_model_kwargs = {'revision':args.revision}
        if 'gpt-j' in name or 'neox' in name:
            base_model_kwargs.update(dict(torch_dtype=torch.float16))
        if 'gpt-j' in name:
            base_model_kwargs.update(dict(revision='float16'))
        if 'llama' in name.lower():
            base_model = transformers.AutoModelForCausalLM.from_pretrained(name, **base_model_kwargs, use_flash_attention_2=True, torch_dtype=torch.bfloat16, trust_remote_code = True, cache_dir=args.cache_dir)
        else:
            base_model = transformers.AutoModelForCausalLM.from_pretrained(name, **base_model_kwargs, use_flash_attention_2=False, torch_dtype=torch.bfloat16, trust_remote_code = True, cache_dir=args.cache_dir)
    else:
        base_model = None

    optional_tok_kwargs = {}
    if "facebook/opt-" in name:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs['fast'] = False
    if args.dataset_member in ['pubmed'] or args.dataset_nonmember in ['pubmed']:
        optional_tok_kwargs['padding_side'] = 'left'
    
    if 'llama' in name or 'checkpoint' in name:
        base_tokenizer = ""
    else:
        base_tokenizer = transformers.AutoTokenizer.from_pretrained(name, **optional_tok_kwargs, cache_dir=args.cache_dir)
        
        if args.dataset_member in ['forget'] or args.dataset_nonmember in ['retain']:
            base_tokenizer.pad_token = base_tokenizer.eos_token
        else:
            base_tokenizer.pad_token_id = base_tokenizer.eos_token_id

    return base_model, base_tokenizer

def load_base_model(args):
    print('MOVING BASE MODEL TO GPU...', end='', flush=True)
    start = time.time()
    try:
        mask_model.cpu()
    except NameError:
        pass
    if args.openai_model is None:
        base_model.to(DEVICE)
    print(f'DONE ({time.time() - start:.2f}s)')

def get_data(args, tokenizer, dataset, key, train=True):
    # load data
    data_split = 'train' if train else 'test'
    if type(dataset) == str:
        if dataset in custom_datasets.DATASETS:
            data = custom_datasets.load(dataset, args.cache_dir)
        elif dataset == 'the_pile' and data_split=='train':
            #data_files = "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"
            #data_files="/home/niloofar/projects/enron_mail_20150507.tar.gz"
            #data_files ="/home/niloofar/projects/maildir"
            #data = datasets.load_dataset("json", data_files=data_files, split="train", cache_dir=cache_dir)[key]
            #data = datasets.load_dataset("json",data_files=data_files, split='train', cache_dir=cache_dir)[key]https://the-eye.eu/public/AI/pile/train/00.jsonl.zst"
            data = datasets.load_dataset("json", data_files= 'cache_100_200_1000_512/train/the_pile_pubmed_abstracts.json', cache_dir=args.cache_dir)[data_split][key]
        elif dataset == 'the_pile' and data_split=='test':
            print("test")
            data = datasets.load_dataset("json", data_files="cache_100_200_1000_512/train/the_pile_pubmed_abstracts.json", cache_dir=args.cache_dir)[data_split][key]
        elif dataset == 'bookcorpus':
            data = datasets.load_dataset('text', data_files = {'train': '/scratch/deu9yh/llm_privacy/tofu/dataset/books_large_p2.txt'})["train"][key][:10000]
        elif dataset == 'wikitext':
            data = datasets.load_dataset('wikitext', 'wikitext-2-v1', cache_dir=args.cache_dir, ignore_verifications=True)[data_split][key]
        elif dataset == 'harry':
            data = datasets.load_dataset('csv', data_files = {'train': '/scratch/deu9yh/llm_privacy/tofu/dataset/harry_potter.csv'}, cache_dir=args.cache_dir)[data_split][key][:10000]
        else:
            data = datasets.load_dataset(dataset, split=f'train[:10000]', cache_dir=args.cache_dir)[key]
        
        # remove duplicates from the data
        data = list(dict.fromkeys(data))  # deterministic, as opposed to set()
        
        # strip whitespace around each example
        data = [x.strip() for x in data]

        # remove newlines from each example
        data = [strip_newlines(x) for x in data]
    else:
        data = dataset

    # get unique examples, strip whitespace, and remove newlines
    # then take just the long examples, shuffle, take the first 5,000 to tokenize to save time
    # then take just the examples that are <= 512 tokens (for the mask model)
    # then generate n_samples samples

    # try to keep only examples with > 100 words
    #if dataset in ['writing', 'squad', 'xsum']:
    long_data = [x for x in data if len(x.split()) > 100]
    # print(len(long_data))
    if len(long_data) > 0:
        data = long_data

    
    not_too_long_data = [x for x in data if len(x.split()) < args.max_length]
    if len(not_too_long_data) > 0:
            data = not_too_long_data

    random.seed(0)
    random.shuffle(data)

    data = data[:5_000]

    # keep only examples with <= 512 tokens according to mask_tokenizer
    # this step has the extra effect of removing examples with low-quality/garbage content
    tokenized_data = tokenizer(data, return_tensors='pt', padding=True, truncation=True, max_length=args.max_length)
    # print(tokenized_data.items())
    data = [x for x, y in zip(data, tokenized_data["input_ids"]) if len(y) <= 512]

    # print stats about remainining data
    print(f"Total number of samples: {len(data)}")
    print(f"Average number of words: {np.mean([len(x.split()) for x in data])}")

    return data

def trim_to_shorter_length(texta, textb, args):
    # truncate to shorter of o and s
    shorter_length = min(len(texta.split(' ')), len(textb.split(' ')))
    if args.max_length is not None:
        shorter_length = min(shorter_length,args.max_length)
    texta = ' '.join(texta.split(' ')[:shorter_length])
    textb = ' '.join(textb.split(' ')[:shorter_length])
    return texta, textb

def generate_samples(args, raw_data_member, raw_data_non_member):
    torch.manual_seed(42)
    np.random.seed(42)
    data = {
        "nonmember": [],
        "member": [],
    }
    batch_size = args.batch_size
    seq_lens = []
    for batch in range(len(raw_data_member) // batch_size):
        print('Generating samples for batch', batch, 'of', len(raw_data_member) // batch_size)
        non_member_text = raw_data_non_member[batch * batch_size:(batch + 1) * batch_size]
        member_text = raw_data_member[batch * batch_size:(batch + 1) * batch_size]
        #sampled_text = sample_from_model(original_text, min_words=30 if args.dataset in ['pubmed'] else 55,max_length=args.max_length if args.max_length is not None else 200)

        #TODO make same len
        for o, s in zip(non_member_text, member_text):


            o, s = trim_to_shorter_length(o, s, args)


            # add to the data
            assert len(o.split(' ')) == len(s.split(' '))
            seq_lens.append(len(o.split(' ')))

            if args.tok_by_tok:
                for tok_cnt in range(len(o.split(' '))):

                    data["nonmember"].append(' '.join(o.split(' ')[:tok_cnt+1]))
                    data["member"].append(' '.join(s.split(' ')[:tok_cnt+1]))
            else:
                data["nonmember"].append(o)
                data["member"].append(s)
    if args.tok_by_tok:
        n_samples = len(data["nonmember"])
    else:
        n_samples = args.n_samples

    return data, seq_lens, n_samples


def get_perplexity(args, model, tokenizer, data, model_type='base', data_type = 'forget'):
    loss = []
    model = model.to(args.device)
    with torch.no_grad():
        for batch in tqdm(data, desc= f"Evaluating {model_type} model on {data_type} set", total = len(data)):
            if not 'llama' in args.base_model_name.lower():
                batch = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=args.max_length)
                input_id = batch['input_ids'].to(args.device)
                attention = batch['attention_mask'].to(args.device)
                label = batch['input_ids'].to(args.device)
            else:
                input_id = batch[0].to(args.device)
                label = batch[1].to(args.device)
                attention = batch[2].to(args.device)
            
            outputs = model(input_ids = input_id, labels = label, attention_mask = attention)
            loss.append(get_batch_loss(outputs.logits, label).mean().cpu().item())
            del input_id, label, attention, outputs
    # print(loss)
    return np.exp(np.mean(loss))

if __name__ == '__main__':
    global args
    DEVICE = 'cuda'
    args = get_args()
    args.device = DEVICE
    cfg = get_config_from_yaml(args.config_name)
    model_cfg = get_model_identifiers_from_yaml(cfg['model_family'])
    cfg['max_length'] = args.max_length
    print(f"Base model: {args.base_model_name}")
    # exit(0)

    if args.base_tokenizer_name is None:
        args.base_tokenizer_name = args.base_model_name
    
    base_model, _ = load_base_model_and_tokenizer(args.base_model_name, args)
    _, base_tokenizer = load_base_model_and_tokenizer(args.base_tokenizer_name, args)
    unlearned_model, _ = load_base_model_and_tokenizer(args.unlearned_model, args)

    load_base_model(args)

    if 'llama' in args.base_model_name:
        data_member = TextDatasetQA(data_path= "harry_potter", tokenizer=base_tokenizer, max_length=args.max_length, model_family= cfg['model_family'])
        data_nonmember = TextDatasetQA("tofu/dataset/books_large_p2.txt", tokenizer=base_tokenizer, max_length=args.max_length, model_family= cfg['model_family'])
        member_loader = torch.utils.data.DataLoader(
            data_member, batch_size=args.batch_size, collate_fn = custom_data_collator
        )
        nonmember_loader = torch.utils.data.DataLoader(
            data_nonmember, batch_size=args.batch_size, collate_fn = custom_data_collator
        )
    else:
        data_member = get_data(args, base_tokenizer, args.dataset_member, args.dataset_member_key)
        data_nonmember = get_data(args, base_tokenizer, args.dataset_nonmember, args.dataset_nonmember_key)

    # # data_forget, seq_lens, n_samples = generate_samples(args, data_member[:args.n_samples], data_nonmember[:args.n_samples])
    # indices = range(min(cfg['eval']['ds_size'], len(data_member)))
    # data_member = torch.utils.data.Subset(data_member, indices)

    # indices = range(min(cfg['eval']['ds_size'], len(data_nonmember)))
    # data_nonmember = torch.utils.data.Subset(data_member, indices)

        member_loader = torch.utils.data.DataLoader(
            data_member, batch_size=args.batch_size
        )
        nonmember_loader = torch.utils.data.DataLoader(
            data_nonmember, batch_size=args.batch_size
        )

    base_forget_perplexity = get_perplexity(args, base_model, base_tokenizer, member_loader, model_type = 'base', data_type= 'forget')
    unlearned_forget_perplexity = get_perplexity(args, unlearned_model, base_tokenizer, member_loader, model_type = 'unlearned', data_type= 'forget')
    
    base_retain_perplexity = get_perplexity(args, base_model, base_tokenizer, nonmember_loader, model_type = 'base', data_type= 'nonmember')
    unlearned_retain_perplexity = get_perplexity(args, base_model, base_tokenizer, nonmember_loader, model_type = 'unlearned', data_type= 'nonmember')

    print(f"Base model perplexity on forget samples: {base_forget_perplexity}")
    print(f"Base model perplexity on non-member samples: {base_retain_perplexity}")

    print(f"Unlearned model perplexity on forget samples: {unlearned_forget_perplexity}")
    print(f"Unlearned model perplexity on non-member samples: {unlearned_retain_perplexity}")

