from pathlib import Path

import torch
from datasets import Dataset

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def assert_trainable(model, hint: str) -> int:
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_trainable == 0:
        raise AssertionError(
            f"Reloaded checkpoint has no trainable parameters. {hint}"
        )
    return n_trainable


def make_dpo_dataset(pairs: list, tokenizer) -> Dataset:
    fmt = lambda content: [{"role": "user", "content": content}]
    return Dataset.from_dict({
        "prompt": [
            tokenizer.apply_chat_template(fmt(p["prompt"]), tokenize=False,
                                          add_generation_prompt=True)
            for p in pairs
        ],
        "chosen": [p["chosen"] for p in pairs],
        "rejected": [p["rejected"] for p in pairs],
    })


def load_base_model(model_name: str, max_seq_len: int, device: str):
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name, max_seq_length=max_seq_len,
        dtype=None, load_in_4bit=True, device_map={"": device},
    )
    return model, tokenizer


def attach_lora(model, lora_rank: int, lora_alpha: int):
    from unsloth import FastLanguageModel
    return FastLanguageModel.get_peft_model(
        model, r=lora_rank, lora_alpha=lora_alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth", random_state=42,
    )


def load_checkpoint_for_training(path: str, max_seq_len: int, device: str):
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=path, max_seq_length=max_seq_len,
        dtype=None, load_in_4bit=True, device_map={"": device},
    )
    assert_trainable(
        model,
        hint=(
            f"Call attach_lora(model, lora_rank, lora_alpha) before training "
            f"on the checkpoint reloaded from '{path}'."
        ),
    )
    return model, tokenizer


def load_checkpoint_for_eval(path: str, max_seq_len: int, device: str):
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=path, max_seq_length=max_seq_len,
        dtype=None, load_in_4bit=True, device_map={"": device},
    )
    return model, tokenizer


def chat(model, tokenizer, user_msg: str, device: str, max_new_tokens: int = 300) -> str:
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids=inputs, max_new_tokens=max_new_tokens,
            temperature=0.7, do_sample=True,
        )
    return tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)


def run_sft(model, tokenizer, documents: list, output_dir: str, max_seq_len: int):
    from unsloth import is_bfloat16_supported
    from trl import SFTConfig, SFTTrainer
    texts = [d["text"] for d in documents if d.get("text", "").strip()]
    dataset = Dataset.from_dict({"text": texts})
    n_steps = len(dataset) // (2 * 4)
    warmup = max(1, int(0.05 * n_steps))

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text", max_seq_length=max_seq_len,
            dataset_num_proc=2, per_device_train_batch_size=2,
            gradient_accumulation_steps=4, warmup_steps=warmup,
            num_train_epochs=1, learning_rate=2e-5,
            fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
            logging_steps=20, optim="adamw_8bit", weight_decay=0.01,
            lr_scheduler_type="cosine", seed=42,
            output_dir=output_dir, save_strategy="epoch",
        ),
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer


def run_dpo(model, tokenizer, pairs: list, output_dir: str, max_seq_len: int):
    from unsloth import is_bfloat16_supported
    from trl import DPOConfig, DPOTrainer
    dpo_dataset = make_dpo_dataset(pairs, tokenizer)
    n_steps = (len(dpo_dataset) * 3) // (1 * 8)
    warmup = max(1, int(0.05 * n_steps))

    trainer = DPOTrainer(
        model=model, ref_model=None,
        args=DPOConfig(
            per_device_train_batch_size=1, gradient_accumulation_steps=8,
            warmup_steps=warmup, num_train_epochs=3, learning_rate=5e-7,
            fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
            logging_steps=25, optim="adamw_8bit",
            output_dir=output_dir, beta=0.1,
            max_length=max_seq_len, max_prompt_length=512,
        ),
        train_dataset=dpo_dataset, tokenizer=tokenizer,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer
