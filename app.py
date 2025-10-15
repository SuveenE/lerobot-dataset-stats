import os
from typing import List

import gradio as gr
from huggingface_hub import HfApi

from get_dataset_stats import (
    get_dataset_stats,
    format_stats_display,
    compare_metadata_with_actual,
)


def search_datasets_fn(query: str) -> List[str]:
    """Search for datasets on HuggingFace"""
    api = HfApi()
    try:
        items = api.list_datasets(search=(query or "").strip() or None)
        repo_ids = [getattr(d, "id", None) or getattr(d, "repo_id", None) for d in items]
        repo_ids = [r for r in repo_ids if r]
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for r in repo_ids:
            if r not in seen:
                unique.append(r)
                seen.add(r)
        return unique[:500]
    except Exception as e:
        print(f"Error searching datasets: {e}")
        return []


def fetch_stats_fn(repo_id: str, progress=gr.Progress()):
    """Fetch dataset statistics"""
    if not repo_id:
        return "Please select a dataset", "", ""
    
    try:
        progress(0.3, desc="Fetching dataset info...")
        token = os.environ.get("HF_TOKEN")
        
        progress(0.5, desc="Analyzing files...")
        stats = get_dataset_stats(repo_id, hf_token=token)
        
        progress(0.8, desc="Formatting results...")
        
        # Format main stats display
        stats_display = format_stats_display(stats)
        
        # Format comparison if metadata exists
        comparison_display = ""
        if stats.get("info_metadata"):
            comparison_display = compare_metadata_with_actual(stats)
        
        # Format episode list
        episodes_list = ""
        if stats["episode_numbers"]:
            episodes = stats["episode_numbers"]
            if len(episodes) <= 100:
                episodes_list = f"**Episode Numbers:** {', '.join(map(str, episodes))}"
            else:
                episodes_list = f"**Episode Numbers:** {', '.join(map(str, episodes[:50]))}... (showing first 50 of {len(episodes)})"
        
        progress(1.0, desc="Complete!")
        
        return stats_display, comparison_display, episodes_list
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Error fetching stats: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg, "", ""


def batch_fetch_stats_fn(repo_ids_text: str, progress=gr.Progress()):
    """Fetch statistics for multiple datasets"""
    if not repo_ids_text or not repo_ids_text.strip():
        return "Please provide at least one dataset repo ID"
    
    # Parse repo IDs (one per line or comma-separated)
    repo_ids = []
    for line in repo_ids_text.strip().split("\n"):
        for repo_id in line.split(","):
            repo_id = repo_id.strip()
            if repo_id:
                repo_ids.append(repo_id)
    
    if not repo_ids:
        return "No valid repo IDs found"
    
    token = os.environ.get("HF_TOKEN")
    results = []
    results.append(f"**Fetching stats for {len(repo_ids)} datasets...**\n")
    results.append("=" * 80 + "\n")
    
    for i, repo_id in enumerate(repo_ids):
        try:
            progress((i + 1) / len(repo_ids), desc=f"Processing {repo_id}...")
            stats = get_dataset_stats(repo_id, hf_token=token)
            
            results.append(f"\n**{i+1}. {repo_id}**")
            if stats.get("error"):
                results.append(f"   ❌ Error: {stats['error']}")
            else:
                results.append(f"   📊 Episodes: {stats['total_episodes']}")
                results.append(f"   📄 Parquet files: {stats['total_parquet_files']}")
                results.append(f"   🎥 Video files: {stats['total_video_files']}")
                if stats.get("codebase_version"):
                    results.append(f"   🔖 Version: {stats['codebase_version']}")
            results.append("")
            
        except Exception as e:
            results.append(f"\n**{i+1}. {repo_id}**")
            results.append(f"   ❌ Error: {str(e)}\n")
    
    results.append("=" * 80)
    results.append(f"\n**Complete!** Processed {len(repo_ids)} datasets.")
    
    return "\n".join(results)


# Build the Gradio interface
with gr.Blocks(title="LeRobot Dataset Stats Viewer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📊 LeRobot Dataset Statistics Viewer")
    gr.Markdown("View statistics for Hugging Face datasets in LeRobot format without downloading them.")
    
    with gr.Tabs():
        # Single dataset tab
        with gr.Tab("Single Dataset"):
            # Load initial datasets
            _initial_choices = search_datasets_fn("griffinlabs-cortex")
            
            with gr.Row():
                org_input = gr.Textbox(
                    label="Organization or keyword",
                    value="griffinlabs-cortex",
                    placeholder="e.g., lerobot, griffinlabs-cortex"
                )
                load_btn = gr.Button("🔍 Load Datasets", variant="secondary")
            
            dataset_dropdown = gr.Dropdown(
                label="Select dataset",
                choices=_initial_choices,
                interactive=True,
            )
            
            fetch_btn = gr.Button("📊 Fetch Statistics", variant="primary", size="lg")
            
            with gr.Row():
                with gr.Column():
                    stats_output = gr.Markdown(label="Statistics")
                
                with gr.Column():
                    comparison_output = gr.Markdown(label="Metadata Comparison")
            
            episodes_output = gr.Markdown(label="Episode Numbers")
        
        # Batch processing tab
        with gr.Tab("Batch Processing"):
            gr.Markdown("**Fetch statistics for multiple datasets at once**")
            gr.Markdown("Enter one repo ID per line or comma-separated.")
            
            batch_input = gr.Textbox(
                label="Dataset repo IDs",
                placeholder="org/dataset1\norg/dataset2\norg/dataset3",
                lines=10
            )
            
            batch_btn = gr.Button("📊 Fetch All Statistics", variant="primary", size="lg")
            
            batch_output = gr.Textbox(
                label="Batch Results",
                lines=25,
                max_lines=50,
            )
    
    # Event handlers for single dataset tab
    def load_datasets_from_org(org_name):
        results = search_datasets_fn(org_name)
        return gr.update(choices=results, value=None)
    
    load_btn.click(
        load_datasets_from_org,
        inputs=org_input,
        outputs=dataset_dropdown,
    )
    
    fetch_btn.click(
        fetch_stats_fn,
        inputs=dataset_dropdown,
        outputs=[stats_output, comparison_output, episodes_output],
    )
    
    # Also fetch stats when a dataset is selected
    dataset_dropdown.change(
        fetch_stats_fn,
        inputs=dataset_dropdown,
        outputs=[stats_output, comparison_output, episodes_output],
    )
    
    # Event handlers for batch processing tab
    batch_btn.click(
        batch_fetch_stats_fn,
        inputs=batch_input,
        outputs=batch_output,
    )


if __name__ == "__main__":
    demo.launch()
