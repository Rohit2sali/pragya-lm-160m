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

To use the trained models, you can download them from here,  
https://huggingface.co/Rohit2sali/pragya-lm-160m

Below are some examples of the model at inference.  

Q - what is artificial intelligence technology?  
Ans - It is the ability to understand things, and the ability to see things.   
      There is no way to make it any better than we do, and we have to learn how to understand it.  
      We can't learn how to read and interpret it.  

Q - what is love?  
Ans - The idea is that if you want to love something, and your heart starts beating faster than  
      it normally would. That's the definition of "love."  

Q - what is love?  
Ans - I'm not sure if it's a literal definition of love, but the idea that we're all together  
      in one place or another is based on some sort of instinctual connection.

Q - what do you think about Europe?  
Ans - Europe is a continent, so the idea that Europe has an abundance of water (and other things)  
      and not just land seems to be false.  

Q - what is democracy?  
Ans - The term democracy is a bit misleading. It's not just the idea of government,  
      it's the concept that people can vote for things they don't like and decide what their  
      opinion on them should be based on whether or not you agree with your view of the world.

Q - which is the best country in the world?  
Ans - I don't think there's any way to know.  

Q - what is difference between animals and humans?  
Ans - Humans are generally considered to be the most intelligent species in the world,  
      and humans have been around for a long time. Humans were not born with this trait,  
      but they evolved it as an advantage over other animals.

Q - what is the meaning of emotion?  
Ans - It's a very complex phenomenon. For example, when you are talking about something   
      that is thought to be the result of an action, it makes sense for your brain to   
      interpret this as emotion.  











