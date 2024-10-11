from .transformers_model import TransformerVision2SeqModel

# Help to add all models here - https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModelForVision2Seq.from_config.config
vision2seq_models = [
    "Salesforce/blip2-opt-2.7b",
    "sashakunitsyn/vlrm-blip2-opt-2.7b",
    "Salesforce/blip-image-captioning-large",
    "Salesforce/blip-image-captioning-base",
    "facebook/chameleon-7b",
    "facebook/chameleon-30b",
    "microsoft/kosmos-2-patch14-224",
    "Qwen/Qwen2-VL-2B-Instruct",
    "Qwen/Qwen2-VL-7B-Instruct",
    "Qwen/Qwen2-VL-72B-Instruct",
    "llava-hf/llava-v1.6-mistral-7b-hf",
    "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
]

__all__ = [
    "TransformerVision2SeqModel",
    "vision2seq_models",
]
