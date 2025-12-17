from transformers import AutoImageProcessor, SiglipForImageClassification, TrainingArguments, Trainer
from datasets import load_dataset
import torch

def train_model(data_dir, output_dir, num_epochs=3, batch_size=8):
    """
    실내/실외 분류 모델 fine-tuning
    """
    # 데이터셋 로드
    dataset = load_dataset("imagefolder", data_dir=data_dir)
    
    # 프로세서 및 모델 로드
    model_name = "prithivMLmods/IndoorOutdoorNet"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = SiglipForImageClassification.from_pretrained(model_name)
    
    # 데이터 전처리
    def preprocess_images(examples):
        images = [img.convert("RGB") for img in examples["image"]]
        inputs = processor(images=images, return_tensors="pt")
        inputs["labels"] = examples["label"]
        return inputs
    
    dataset = dataset.map(preprocess_images, batched=True)
    
    # 학습 설정
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # Trainer 생성 및 학습
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )
    
    trainer.train()
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    
    print(f"✅ 모델 저장: {output_dir}")

if __name__ == "__main__":
    train_model(
        data_dir="/workspace/indooroutdoor_dataset",
        output_dir="/workspace/indooroutdoor_dataset/final_model",
        num_epochs=200,
        batch_size=8
    )