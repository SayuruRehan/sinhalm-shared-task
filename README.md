---
base_model: google/gemma-3-4b-it
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:google/gemma-3-4b-it
- lora
- transformers
- sinhala
- multilingual
language:
- si
- en
license: apache-2.0
datasets:
- 0xAIT/sinhala-flan
---

# SinhaLM-Gemma-3-4b-it

A LoRA fine-tuned instruction-following version of Google's Gemma-3-4b-it model specifically optimized for Sinhala language tasks using the Sinhala FLAN dataset.

## Model Details

### Model Description

This model is a Parameter-Efficient Fine-Tuning (PEFT) adaptation of Google's Gemma-3-4b-it using Low-Rank Adaptation (LoRA) technique. The model has been instruction-tuned on the Sinhala FLAN dataset to improve performance on instruction-following tasks in Sinhala language while maintaining English capabilities. The training focused on enhancing the model's ability to understand and respond to instructions in Sinhala.

- **Developed by:** Sulakna Weerasinghe, Ovindu Gunathunga, Supun Edirisuriya, Sayuru Bopitiya
- **Model type:** Instruction-tuned Causal Language Model (LoRA Adapter)
- **Language(s):** Sinhala (primary), English (secondary)
- **License:** Apache-2.0
- **Finetuned from model:** google/gemma-3-4b-it
- **Base model size:** 4B parameters
- **Adapter parameters:** LoRA with rank 16

### Model Sources

- **Base Repository:** [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)
- **Training Dataset:** [0xAIT/sinhala-flan](https://huggingface.co/datasets/0xAIT/sinhala-flan)

## Uses

### Direct Use

This model is designed for instruction-following tasks in Sinhala language, including:
- Following instructions and commands in Sinhala
- Question answering in Sinhala
- Text completion and generation based on Sinhala prompts
- Translation between Sinhala and English
- General conversational AI with instruction-following capabilities in Sinhala

### Downstream Use

The model can be further fine-tuned for specific Sinhala NLP tasks such as:
- Sinhala text classification
- Named entity recognition in Sinhala
- Sentiment analysis for Sinhala text
- Domain-specific chatbots for Sinhala speakers

### Out-of-Scope Use

This model is not suitable for:
- Tasks requiring high accuracy in languages other than Sinhala and English
- Production systems without proper safety evaluations
- Applications where cultural sensitivity has not been properly assessed

## Training Details

### Training Data

The model was trained on the Sinhala FLAN dataset (0xAIT/sinhala-flan), which contains instruction-following examples in Sinhala. The FLAN (Finetuned Language Models are Zero-Shot Learners) methodology focuses on improving instruction-following capabilities through diverse task formatting. Due to computational constraints, training was performed on a subset of 50,000 samples from the original dataset of 2,263,067 samples.

### Training Procedure

#### Training Configuration

- **Training regime:** Mixed precision (bf16)
- **Optimizer:** AdamW
- **Learning rate:** 5e-4
- **Weight decay:** 0.01
- **Warmup steps:** 50
- **Max gradient norm:** 1.0
- **Training samples:** 50,000 (sampled from full dataset)
- **Validation samples:** 5,000
- **Training epochs:** 1 (early stopped)
- **Total training steps:** 1,000
- **Effective batch size:** 32 (per_device_batch_size=8, gradient_accumulation_steps=4)

#### LoRA Configuration

- **LoRA rank (r):** 16
- **LoRA alpha:** 32
- **LoRA dropout:** 0.1
- **Target modules:** All linear layers in attention and MLP blocks

#### Hardware and Performance

- **Training time:** 1.84 hours
- **Hardware:** GPU with 39.6GB VRAM
- **Peak memory usage:** 4.7GB reserved
- **Training throughput:** ~0.17 iterations/second

### Training Results

| Step | Training Loss | Validation Loss |
|------|---------------|-----------------|
| 500  | 6.172         | 1.522           |
| 1000 | 5.782         | 1.440           |

The model showed consistent improvement in both training and validation loss throughout the training process.

## Technical Specifications

### Model Architecture

- **Base architecture:** Gemma-3-4b-it (decoder-only transformer)
- **Adaptation method:** LoRA (Low-Rank Adaptation)
- **Parameter efficiency:** Only ~0.1% of base model parameters trained
- **Precision:** Mixed precision training with bfloat16

### Framework Versions

- **PEFT:** 0.17.0
- **Transformers:** Latest compatible version
- **PyTorch:** CUDA-enabled version

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b-it")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")

# Load the LoRA adapter
model = PeftModel.from_pretrained(base_model, "sula15/SinhaLM-Gemma-3-4b-it")

# Generate text
inputs = tokenizer("ප්‍රශ්නය: ", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Limitations and Considerations

### Performance Limitations

- Trained on a subset (50k samples) of the full Sinhala FLAN dataset due to computational constraints
- Single epoch training may limit the model's full potential
- Performance on complex Sinhala language tasks may require additional fine-tuning

### Bias and Ethical Considerations

- The model inherits biases from both the base Gemma-3-4b-it model and the Sinhala FLAN dataset
- Cultural and linguistic nuances specific to Sinhala-speaking communities should be carefully evaluated
- Users should conduct appropriate bias testing before deployment in production systems

## Model Card Authors

Sulakna Weerasinghe,Ovindu Gunathunga, Supun Edirisuriya, Sayuru Bopitiya

## Citation

If you use this model, please cite:

```bibtex
@model{weerasinghe2025sinhalm,
  title={SinhaLM-Gemma-3-4b-it: A LoRA-adapted Gemma model for Sinhala instruction-following},
  author={Weerasinghe, Sulakna and Gunathunga, Ovindu and Edirisuriya, Supun and Bopitiya, Sayuru},
  year={2025},
  publisher={Hugging Face},
  url={https://huggingface.co/sula15/SinhaLM-Gemma-3-4b-it}
}
```
