import torch
import torchvision.transforms.functional as TVF
from PIL import Image
from transformers import AutoTokenizer, LlavaForConditionalGeneration

from ..model_registry import ModelInputOutput, register_model
from ..models import BaseModel, track_inference


@register_model(
    "fancyfeast/llama-joycaption-alpha-two-hf-llava",
    "transformers",
    ModelInputOutput.IMAGE_TEXT_TO_TEXT,
)
class JoyCaption(BaseModel):
    def __init__(
        self, model_id: str, device: str = "cpu", dtype: str = "float32", **kwargs
    ):
        super().__init__(model_id, device, dtype)
        self.load_model(**kwargs)

    def load_model(self, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id, **kwargs
        ).to(self.device, self.dtype)
        self.model.eval()

    def preprocess(self, image: str, prompt: str = None):
        image = Image.open(image).convert("RGB")
        image = TVF.resize(image, (384, 384), Image.LANCZOS)
        image = TVF.pil_to_tensor(image)

        image = image / 255.0
        image = TVF.normalize(image, [0.5], [0.5])
        image = image.to(self.dtype).unsqueeze(0)

        # Build the conversation
        convo = [
            {
                "role": "system",
                "content": "You are a helpful image captioner.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        # Format the conversation
        convo_string = self.tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=True
        )

        # Tokenize the conversation
        convo_tokens = self.tokenizer.encode(
            convo_string, add_special_tokens=False, truncation=False
        )

        # Repeat the image tokens
        input_tokens = []
        for token in convo_tokens:
            if token == self.model.config.image_token_index:
                input_tokens.extend(
                    [self.model.config.image_token_index]
                    * self.model.config.image_seq_length
                )
            else:
                input_tokens.append(token)

        input_ids = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)

        return image, input_ids, attention_mask

    @track_inference
    def infer(self, image: str, prompt: str = None, **generate_kwargs):
        with torch.inference_mode(), torch.amp.autocast(
            device_type=self.device, dtype=self.dtype
        ):
            image, input_ids, attention_mask = self.preprocess(image, prompt)

            if "max_new_tokens" not in generate_kwargs:
                generate_kwargs["max_new_tokens"] = 300

            generate_ids = self.model.generate(
                input_ids=input_ids.to(self.device),
                pixel_values=image.to(self.device),
                attention_mask=attention_mask.to(self.device),
                do_sample=True,
                suppress_tokens=None,
                use_cache=True,
                **generate_kwargs,
            )[0]

        # Trim off the prompt
        generate_ids = generate_ids[input_ids.shape[1] :]

        # Decode the caption
        caption = self.tokenizer.decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        caption = caption.strip()

        return caption

    def preprocess_batch(self, images: list[str], prompts: list[str]):
        processed_images = []
        input_ids_list = []
        attention_masks = []

        for image, prompt in zip(images, prompts):
            image = Image.open(image).convert("RGB")
            image = TVF.resize(image, (384, 384), Image.LANCZOS)
            image = TVF.pil_to_tensor(image)
            image = image / 255.0
            image = TVF.normalize(image, [0.5], [0.5])
            image = image.to(self.dtype)
            processed_images.append(image)

            convo = [
                {"role": "system", "content": "You are a helpful image captioner."},
                {"role": "user", "content": prompt},
            ]
            convo_string = self.tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=True
            )
            convo_tokens = self.tokenizer.encode(
                convo_string, add_special_tokens=False, truncation=False
            )

            input_tokens = []
            for token in convo_tokens:
                if token == self.model.config.image_token_index:
                    input_tokens.extend(
                        [self.model.config.image_token_index]
                        * self.model.config.image_seq_length
                    )
                else:
                    input_tokens.append(token)

            input_ids = torch.tensor(input_tokens, dtype=torch.long)
            input_ids_list.append(input_ids)

        # Pad input_ids to the same length
        max_length = max(ids.size(0) for ids in input_ids_list)
        padded_input_ids = torch.zeros(
            (len(input_ids_list), max_length), dtype=torch.long
        )
        for i, ids in enumerate(input_ids_list):
            padded_input_ids[i, : ids.size(0)] = ids
            attention_masks.append(torch.ones(max_length, dtype=torch.long))
            attention_masks[-1][ids.size(0) :] = 0

        images_tensor = torch.stack(processed_images)
        attention_mask_tensor = torch.stack(attention_masks)

        return images_tensor, padded_input_ids, attention_mask_tensor

    @track_inference
    def infer_batch(self, images: list[str], prompts: list[str], **generate_kwargs):
        with torch.inference_mode(), torch.amp.autocast(
            device_type=self.device, dtype=self.dtype
        ):
            images, input_ids, attention_mask = self.preprocess_batch(images, prompts)

            if "max_new_tokens" not in generate_kwargs:
                generate_kwargs["max_new_tokens"] = 300

            generate_ids = self.model.generate(
                input_ids=input_ids.to(self.device),
                pixel_values=images.to(self.device),
                attention_mask=attention_mask.to(self.device),
                do_sample=True,
                suppress_tokens=None,
                use_cache=True,
                **generate_kwargs,
            )

        captions = []
        for i, gen_ids in enumerate(generate_ids):
            # Trim off the prompt
            gen_ids = gen_ids[input_ids.shape[1] :]

            # Decode the caption
            caption = self.tokenizer.decode(
                gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            captions.append(caption.strip())

        return captions
