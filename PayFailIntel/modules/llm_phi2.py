from llama_cpp import Llama

# Path to your GGUF file
MODEL_PATH = "D:\PythonTraining\PayFailIntel\phi2_gguf\phi-2.Q4_K_M.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=6
)

def generate(prompt):
    out = llm(
        prompt,
        max_tokens=256,
        temperature=0.2,
        stop=[
            "Answer STRICTLY",
            "IMPORTANT",
            "If insufficient",
            "User Question:",
            "Context:",
        ],
        echo=False
    )
    return out["choices"][0]["text"].strip()
