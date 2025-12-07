import argparse
import json
import logging
from typing import Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

from qwen_vl_utils import process_vision_info


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class MMDataset(Dataset):
    def __init__(self, data: List[Dict]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        return self.data[int(idx)]


def build_collate_fn(processor: Qwen2_5_VLProcessor):
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
    model_path = config["training"].get("output_dir", config["models"].get("student"))

    logging.info(f"Loading trained reranker from {model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    processor = Qwen2_5_VLProcessor.from_pretrained(model_path)

    model.eval()

    batch_size = config.get("evaluation", {}).get("batch_size", 2)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=build_collate_fn(processor))

    total_loss = 0.0
    total_examples = 0

    for batch in dataloader:
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
