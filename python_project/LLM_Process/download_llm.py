import transformers
import torch
import tqdm
import time

from datasets import load_dataset


class LLMModel:
    def __init__(self, model_name, custom_cache_path="./cache"):
        """初始化LLM模型."""
        self.model_name = model_name
        self.custom_cache_path = custom_cache_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        """下载模型预训练权重和对应的配置文件到指定的缓存目录."""
        if self.model is not None and self.tokenizer is not None:
            print(
                f"Model {self.model_name} is already loaded from {self.custom_cache_path}."
            )
            return

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.custom_cache_path,
            use_fast=True,
        )
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=self.custom_cache_path,
            torch_dtype=torch.float16,
        )
        print(
            f"Model {self.model_name} downloaded successfully to {self.custom_cache_path}."
        )

    def get_model_info(self):
        """获取模型的基本信息."""
        if self.model is None or self.tokenizer is None:
            self.load_model()

        print("\n===Model Information:===")
        print(f"Model Name: {self.model_name}")
        print(f"Model class: {type(self.model).__name__}")
        print(f"Number of layers: {len(self.model.model.decoder.layers)}")
        print(f"Hidden size: {self.model.config.hidden_size}")
        print(f"Attention heads: {self.model.config.num_attention_heads}")
        print(f"FFN dimension: {self.model.config.ffn_dim}")
        print(
            f"Max position embeddings: {self.model.config.max_position_embeddings}"
        )  # 计算困惑度时需要匹配这个参数
        print(f"Vocab size: {self.model.config.vocab_size}")
        print(f"data type: {self.model.dtype}")

    def generate_text(self, prompt, max_length=50):
        """测试, 使用模型生成文本."""
        # TODO: 需要修改, 不能很好的反映实际的生成速度, 波动很大.
        if self.model is None or self.tokenizer is None:
            self.load_model()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
        )
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def mesure_speed(self, prompt, num_tokens=50, num_iterations=10):
        """测量模型生成文本的速度."""
        if self.model is None or self.tokenizer is None:
            self.load_model()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # warm up the model
        self.model.generate(
            **inputs,
            max_length=num_tokens + inputs.input_ids.shape[1],
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
        )

        # Measure the speed
        start_time = time.time()
        for _ in tqdm.tqdm(range(num_iterations), desc="Measuring speed"):
            outputs = self.model.generate(
                **inputs,
                max_length=num_tokens + inputs.input_ids.shape[1],
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
                top_p=0.95,
            )
        end_time = time.time()
        elapsed_time = end_time - start_time
        generated_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
        speed = (generated_tokens * num_iterations) / elapsed_time
        print(
            f"Average speed: {speed:.2f} tokens/second for {num_tokens} tokens over {num_iterations} iterations."
        )
        return speed

    def calculate_perplexity(
        self,
        dataset_name="wikitext",
        split="test",
        seqlen=2048,
        stride=256,  # 滑动窗口, 待补充
    ):
        """计算模型的困惑度."""
        if self.model is None or self.tokenizer is None:
            self.load_model()

        testenc = load_dataset(
            dataset_name, "wikitext-2-raw-v1", split=split, cache_dir="./datasets/"
        )
        testenc = self.tokenizer("\n\n".join(testenc["text"]), return_tensors="pt")
        self.model.seqlen = seqlen
        testenc = testenc.input_ids.to(self.model.device)
        nsamples = testenc.numel() // self.model.seqlen
        self.model.eval()

        nlls = []
        for i in tqdm.tqdm(range(nsamples), desc="Calculating perplexity"):
            start = i * self.model.seqlen
            end = start + self.model.seqlen
            batch = testenc[:, start:end].to(self.device)
            with torch.no_grad():
                lm_logits = self.model(batch).logits  # [batch_size, seqlen, vocab_size]
            shifted_logits = lm_logits[:, :-1, :].contiguous().float()
            shifted_labels = testenc[:, start:end][:, 1:]
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                shifted_logits.view(-1, shifted_logits.size(-1)),
                shifted_labels.view(-1),
            )
            neg_log_likelihood = loss.float() * self.model.seqlen
            nlls.append(neg_log_likelihood)
        pll = torch.exp(torch.stack(nlls).sum() / (nsamples * self.model.seqlen))
        print(
            f"Perplexity of the model {self.model_name} on {dataset_name}: {pll.item()}"
        )
        return pll.item()


if __name__ == "__main__":
    custom_cache_dir = "./cache"
    # List of model names to research
    opt_model_lists = [
        "facebook/opt-125m",
        "facebook/opt-350m",
        "facebook/opt-1.3b",
        "facebook/opt-2.7b",
        "facebook/opt-6.7b",
    ]

    model_name = opt_model_lists[0]
    custom_cache_path = f"{custom_cache_dir}/{model_name.replace('/', '_')}"
    llm_model = LLMModel(model_name, custom_cache_path)
    llm_model.get_model_info()

    # text = llm_model.generate_text(
    #     "This is a test prompt for the LLM model.",
    #     max_length=100,
    # )
    # print(f"\nGenerated Text:\n {text} \n")

    # llm_model.calculate_perplexity()

    llm_model.mesure_speed(
        "This is a test prompt for measuring the speed of the LLM model.",
        num_tokens=1000,
        num_iterations=10,
    )
