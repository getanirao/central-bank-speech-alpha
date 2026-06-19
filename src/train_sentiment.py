import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import warnings
warnings.filterwarnings('ignore')


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {'accuracy': accuracy}


def train_model(output_dir='models/modernfinbert_finetuned', num_epochs=3, batch_size=16):
    print("=" * 70)
    print("  Training ModernFinBERT on FOMC hawkish/dovish/neutral data")
    print("=" * 70)

    dataset = load_dataset('gtfintechlab/fomc_communication')
    label_names = ['Dovish', 'Hawkish', 'Neutral']

    train_data = dataset['train']
    eval_data = dataset['test']
    print(f"  Train: {len(train_data)} | Validation: {len(eval_data)}")

    model_name = 'tabularisai/ModernFinBERT'
    print(f"\nLoading base model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label={i: l for i, l in enumerate(label_names)},
        label2id={l: i for i, l in enumerate(label_names)},
    )

    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    def tokenize_fn(batch):
        return tokenizer(batch['sentence'], truncation=True, max_length=128)

    print("\nTokenizing...")
    tokenized_train = train_data.map(tokenize_fn, batched=True)
    tokenized_eval = eval_data.map(tokenize_fn, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy='epoch',
        save_strategy='epoch',
        learning_rate=5e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        save_total_limit=1,
        report_to='none',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print(f"\nTraining classifier head ({num_epochs} epochs, {len(train_data)} samples, batch={batch_size})...")
    trainer.train()

    eval_result = trainer.evaluate()
    print(f"\nValidation accuracy: {eval_result['eval_accuracy']:.4f}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Fine-tuned model saved to: {output_dir}")


if __name__ == "__main__":
    train_model()
