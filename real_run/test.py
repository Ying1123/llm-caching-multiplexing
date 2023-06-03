from transformers import AutoTokenizer, OPTForCausalLM
import torch

kwargs = {"torch_dtype": torch.float16}
model = OPTForCausalLM.from_pretrained("facebook/opt-350m", low_cpu_mem_usage=True, **kwargs).cuda()
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

prompt = "Hey, are you consciours? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids.cuda(), max_length=30)
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
