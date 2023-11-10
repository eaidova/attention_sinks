from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer, TextStreamer
from types import MethodType

model_path = "./dolly-v2-3b"

model = OVModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.orig_forward = model.forward

def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        position_ids=None,
        **kwargs,
    ):
    # allow calculate attention mask inside forward based on kv cache and seq_len
    return self.orig_forward(input_ids, attention_mask=None, past_key_values=past_key_values, position_ids=position_ids, **kwargs)

model.forward = MethodType(forward, model)


inputs = tokenizer(["An increasing sequence: one,"], return_tensors="pt")
streamer = TextStreamer(tokenizer)

_ = model.generate(**inputs, streamer=streamer, max_new_tokens=200)






