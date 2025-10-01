from vllm import LLM

class ShowO:
    def __init__(self, model_path, args):
        self.llm = LLM(model=model_path, trust_remote_code=True)

    def __call__(self, prompt, **kwargs):
        outputs = self.llm.generate([prompt], **kwargs)
        return outputs[0].outputs[0].text
