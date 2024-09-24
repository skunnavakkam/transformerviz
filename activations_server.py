import modal
from modal import Image, App, Volume

image = Image.debian_slim().pip_install("transformer-lens", "torch", "numpy")
app = App("transformer-visualization")
volume = Volume.from_name("transformer-viz", create_if_missing=True)


@app.cls(image=image, volumes={"/models": volume})
class TransformerViz:
    @modal.build()
    def download_model(self):
        import os
        from transformer_lens import HookedTransformer
        import torch

        os.environ["HF_HOME"] = "/models"
        model_name = "gpt2"
        model = HookedTransformer.from_pretrained(model_name)

    @modal.enter()
    def run_on_startup(self):
        import os
        from transformer_lens import HookedTransformer
        import torch

        os.environ["HF_HOME"] = "/models"
        model_name = "gpt2"
        self.model = HookedTransformer.from_pretrained(
            model_name
        )  # this should pull from volume

    @modal.method()
    def return_activation_cache(self, input):
        import torch
        import numpy as np

        model = self.model
        tokens = model.to_tokens(input)
        _, cache = model.run_with_cache(tokens, return_cache_object=False)

        return_toks = model.tokenizer.convert_ids_to_tokens(tokens[0])

        # Initialize an empty dictionary to store the sanitized cache
        return_cache = {}

        for k, v in cache.items():
            # Move tensor to CPU and ensure it's detached from any computation graph
            v_cpu = v.detach().cpu()

            # Replace positive and negative infinity with NaN
            v_cpu = torch.where(torch.isinf(v_cpu), torch.tensor(float(0)), v_cpu)

            # Convert tensor to list
            v_list = v_cpu.tolist()

            # Replace NaN with None
            sanitized_list = [
                None if (x != x or x == float("inf") or x == float("-inf")) else x
                for x in v_list
            ]

            return_cache[k] = sanitized_list

        return {"tokens": return_toks, "cache": return_cache}


@app.function()
@modal.web_endpoint(method="POST")
def get_activations(input):
    return TransformerViz().return_activation_cache.remote(input)
