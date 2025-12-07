import argparse
import json
import logging
from typing import Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from peft import PeftModel
from qwen_vl_utils import process_vision_info
from tqdm.auto import tqdm


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class MMDataset(Dataset):
    def __init__(self, data: List[Dict]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        return self.data[int(idx)]


def build_collate_fn(processor: AutoProcessor):
    def collate_fn(examples: List[Dict]) -> Dict:
        texts: List[str] = []
        images: List[List] = []

        for example in examples:
            text = processor.apply_chat_template(example, tokenize=False)
            texts.append(text)

            image, _ = process_vision_info(example)
            resized_images = [img.resize((448, 448)) for img in image]
            images.append(resized_images)

        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        image_tokens = [151652, 151653, 151655]
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100

        batch["labels"] = labels
        return batch

    return collate_fn


def evaluate_loss(config: Dict) -> float:
    with open(config["dataset"]["labeled_path"], "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    dataset = MMDataset(raw_data)
    base_model_name = config.get("models", {}).get("student", "Qwen/Qwen2.5-VL-3B-Instruct")
    adapter_dir = config.get("training", {}).get("output_dir", "./result")
    trust_remote_code = config.get("inference", {}).get("trust_remote_code", True)

    logging.info(f"Loading base model {base_model_name} and merging LoRA from {adapter_dir}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    base_param_count = sum(p.numel() for p in model.parameters())
    logging.info(f"Base model parameter count (before merging LoRA): {base_param_count:,}")

    model = PeftModel.from_pretrained(model, adapter_dir)
    lora_state_dict = model.get_peft_model_state_dict()
    lora_param_count = sum(t.numel() for t in lora_state_dict.values())
    logging.info(f"LoRA adapter parameter count: {lora_param_count:,}")

    model = model.merge_and_unload()
    processor = AutoProcessor.from_pretrained(base_model_name)

    model.eval()

    batch_size = config.get("evaluation", {}).get("batch_size", 2)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=build_collate_fn(processor))

    total_loss = 0.0
    total_examples = 0

    for batch in tqdm(dataloader, desc="Evaluating labeled set"):
        batch = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss

        batch_size_actual = batch["input_ids"].size(0)
        total_loss += loss.item() * batch_size_actual
        total_examples += batch_size_actual

    average_loss = total_loss / total_examples if total_examples else 0.0
    logging.info(f"Average loss on labeled set: {average_loss:.6f}")
    return average_loss


def main():
    parser = argparse.ArgumentParser(description="Evaluate average loss of the trained reranker without updating weights")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON config file")
    args = parser.parse_args()

    config = json.load(open(args.config, "r", encoding="utf-8"))
    evaluate_loss(config)


if __name__ == "__main__":
    main()
