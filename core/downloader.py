import os
import re
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files, HfApi

class GGUFDownloader:
    def __init__(self, state):
        self.state = state

    def list_available_quantizations(self, repo):
        """List available quantizations with total file sizes."""
        try:
            api = HfApi()
            info = api.model_info(repo, files_metadata=True)
            gguf_files = [
                (s.rfilename, s.size or 0)
                for s in info.siblings
                if s.rfilename.endswith(".gguf") and "mmproj" not in s.rfilename.lower()
            ]

            if not gguf_files:
                return []

            quant_pattern = re.compile(
                r"(IQ[2-8]_(?:XXS|XS|NL|S|M|L)|Q[2-9]_(?:K_?(?:XL|XL_?M|L|S|M)|0|[1-9]_?[KS])|MXFP4|MXP4|BF16|F16|F32|F8|I4)",
                re.IGNORECASE,
            )

            quant_sizes = {}
            for fname, size in gguf_files:
                basename = fname.split("/")[-1]
                matches = quant_pattern.findall(basename)
                for m in matches:
                    if m not in quant_sizes:
                        quant_sizes[m] = 0
                    quant_sizes[m] += size

            return sorted(quant_sizes.items(), key=lambda x: x[1])
        except Exception as e:
            raise RuntimeError(f"Error listing quantizations: {e}") from e

    def recommend_quant(self, quant_list, vram_mb, ram_mb, ctx_size=4096):
        """Recommend the best quantization based on available memory."""
        total_mb = vram_mb + ram_mb
        
        # Dynamic overhead: 1GB base + KV cache estimate
        # Rough estimate: 0.5MB per 1024 tokens for a medium model (e.g. 7B-30B)
        kv_overhead = (ctx_size / 1024) * 0.5
        overhead_mb = 1024 + kv_overhead

        best = None
        for quant_name, size_bytes in reversed(quant_list):
            size_mb = size_bytes / (1024 * 1024)
            if size_mb + overhead_mb <= total_mb:
                fits_vram = size_mb + overhead_mb <= vram_mb
                if fits_vram:
                    reason = f"Fits entirely in VRAM ({size_mb / 1024:.1f}GB model)"
                else:
                    reason = f"Fits in VRAM+RAM ({size_mb / 1024:.1f}GB model)"
                best = (quant_name, reason)
                break

        if not best and quant_list:
            quant_name, size_bytes = quant_list[0]
            size_mb = size_bytes / (1024 * 1024)
            best = (quant_name, f"Smallest available ({size_mb / 1024:.1f}GB) - may not fit")

        return best

    def get_model_files(self, repo, selected_quantization):
        """Get list of files to download based on selection."""
        try:
            files = list_repo_files(repo)
            if selected_quantization:
                norm_quant = selected_quantization.replace(".", "_").strip("_")
                matching = []
                for f in files:
                    if not f.endswith(".gguf"):
                        continue
                    basename = f.split("/")[-1]
                    if re.search(rf"\b{re.escape(norm_quant)}\b", basename, re.IGNORECASE):
                        matching.append(f)
                return matching
            else:
                return [f for f in files if f.endswith(".gguf")]
        except Exception as e:
            raise RuntimeError(f"Error listing files: {e}") from e

    def download(self, repo, files_to_download, output_dir, progress_callback=None):
        """Download model files."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            downloaded = []
            failed = []

            for filename in files_to_download:
                try:
                    # Note: hf_hub_download has its own progress bar, 
                    # we can use a custom one or just let it run.
                    filepath = hf_hub_download(
                        repo_id=repo,
                        filename=filename,
                        local_dir=output_dir,
                        resume_download=True,
                        local_dir_use_symlinks=False,
                    )
                    downloaded.append((filename, filepath))
                except Exception as e:
                    failed.append((filename, str(e)))
            
            return downloaded, failed
        except Exception as e:
            return [], [("all", str(e))]

    def download_mmproj(self, repo, output_dir):
        """Download the matching mmproj file from HuggingFace."""
        try:
            files = list_repo_files(repo)
            mmproj_files = [f for f in files if "mmproj" in f.lower() and f.endswith(".gguf")]
            if not mmproj_files:
                return None, "No mmproj file found in repo"
            
            # Prefer F16 or the first one
            target = next((f for f in mmproj_files if "F16" in f), mmproj_files[0])
            
            filepath = hf_hub_download(
                repo_id=repo,
                filename=target,
                local_dir=output_dir,
                resume_download=True,
                local_dir_use_symlinks=False,
            )
            return filepath, None
        except Exception as e:
            return None, str(e)

