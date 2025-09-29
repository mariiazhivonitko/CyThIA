from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("mariiazhiv/CyTHIA-Mixtral-8x7B")
print(len(tok))
print(tok.special_tokens_map)
