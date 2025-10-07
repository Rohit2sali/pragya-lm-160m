# pragya-lm-160m  

This is repository for pragya language model which is 160 million parameters of transformer model.  

💡 key-features  
  
✅ It is a Custom-trained 160M parameter transformer optimized for QA.  
📚 Pretrained from scratch on diverse text corpus for general language understanding.  
⚡ Efficient inference — runs on single GPU / CPU after quantization.  
🧩 Tokenizer and weights included for easy reproduction.  
  
🧠 Model Overview  

Model Name: pragya-160M-QA  
Architecture: Transformer-based language model (GPT-like)  
Parameters: 160 million  
Vocabulary Size: (50k tokens)  
Context Length: (128 tokens)  
Pretraining Objective: Causal language modeling  
Fine-tuning Task: Question Answering (generative)  

⚙️ Training Details  
Pretraining  
Dataset(s): (mixture of openwebtext, BookCorpus)  
Total Tokens: (5B tokens)  
Tokenizer: (gpt2 tokenizer)  
Hardware: (2×T4 GPUs on Kaggle)  
Optimization: AdamW optimizer  

Fine-tuning  
Dataset(s): (ELI5)    
Fine-tuning Objective: (next token prediction)  
Evaluation Metrics: EM (Exact Match)  
Perplexity : 149.9536 (dataset : wikitext_103)    
  
⚠️ Limitations  
May produce incorrect or hallucinated answers  
Limited to 512-token context  
No explicit safety filtering  
