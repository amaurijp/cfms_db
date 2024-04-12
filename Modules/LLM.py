from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

#device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
#model = model.to(device)

prompt = "Consider the context: 'Different types of cell cultures, including cancer cell lines have been employed as in vitro toxicity models. It has been generally agreed that NPs interfere with either assay materials or with detection systems.' Any toxicity behavior is described against any living organism? Answer just 'yes' or 'no'."
inputs = tokenizer(prompt, return_tensors="pt")#.to(device)

generate_ids = model.generate(inputs.input_ids, max_length=len(prompt) + 3)
answer = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0][ len(prompt) : ]
print('\nQuestion: ', prompt)
print('Answer: ', answer)
