# Strokes
微调一个汉字笔顺的模型

* 首先从国家网站上了解到了再2020年出版的国家规定，数据集从这个规定上提取。
* 编码问题

*上传到服务器上，开始选择模型进行微调*
Strokes环境用来做主任务


环境有些冲突，所以创建另一个基本环境llava-env，用来做llava的Prompt.py测评，避免环境混乱。

```bash
git clone https://github.com/haotian-liu/LLaVA.git

cd LLaVA

pip install -e .

```
详细可以参考这个[链接](https://github.com/haotian-liu/LLaVA.git)

qwen和llava的Prompt结果都查出来了。
接下来可以尝试

***InternVL***

***DeepSeek***

***ChatGPT-4o***


5.11 重写了train_eos.py作为LoRA的版本。调整了超参数为
```Python
r=16,
lora_alpha=32, #一般是2r
lora_dropout=0.05,
```

5.13
train_levenshtein.py
train_levenshtein_lora.py
更新了带levenshtein距离计算的lora和全量微调。
下一步微调lora的超参数，看看到底lora能达到sft的多少程度。

5.15

更新Prompt.py作为Qwen2-VL-2B Instruct 微调前后效果的对比，主要是跑baseline。修改其中的116行，可以调用微调后的模型。

Prompt_llava.py作为llava-1.5-7b-hf的baseline

5.16
Prompt_Qwen_finetune_516.log是微调以后得模型的效果，
Prompt_Qwen_516.log是微调前的模型效果

5.19
train_levenshtein_lora_519.log调整了lora的超参数，看看效果会不会好。

尝试用文本大模型，看看微调出来效果如何。

评测大模型在微调之后的通用能力有没有下降
7000是最好的那个检查点/home/zsy/GithubCode/Strokes/output_levenshtein/Qwen2-VL-2B/checkpoint-7000