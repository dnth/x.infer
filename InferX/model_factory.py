from .transformers.blip2 import BLIP2


def get_model(model_type: str, implementation: str, **kwargs):
    if implementation == "transformers":
        if model_type == "Salesforce/blip2-opt-2.7b":
            return BLIP2(model_name="Salesforce/blip2-opt-2.7b", **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
