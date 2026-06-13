import os

# from dotenv import load_dotenv
from huggingface_hub import HfApi


# load_dotenv()
# token = os.getenv("HF_TOKEN_UPLOAD")

# if not token:
#     raise RuntimeError(
#         "Missing Hugging Face token. Add HF_TOKEN=... to .env or export HF_TOKEN."
#     )

# api = HfApi(token=token)
api = HfApi()

api.upload_folder(
    folder_path="nemotron_sft_agentic_v2_prompt_jsonl",
    repo_id="carbench-ijcai/augment_nemotron_prompts",
    repo_type="dataset",
)