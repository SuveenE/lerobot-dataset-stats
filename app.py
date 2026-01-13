import os
from typing import List

import gradio as gr
from huggingface_hub import HfApi

from get_dataset_stats import get_dataset_stats


def get_user_organizations() -> List[str]:
    """Get organizations the user is part of"""
    api = HfApi()
    token = os.environ.get("HF_TOKEN")
    try:
        # Get the user's info
        user_info = api.whoami(token=token)
        username = user_info["name"]
        
        # Get organizations the user is part of
        orgs = [username]  # Include user's own namespace
        if "orgs" in user_info:
            orgs.extend([org["name"] for org in user_info["orgs"]])
        
        return orgs
    except Exception as e:
        print(f"Error getting user organizations: {e}")
        return []


def search_datasets_fn(org_name: str) -> List[str]:
    """Search for datasets from a specific organization"""
    api = HfApi()
    token = os.environ.get("HF_TOKEN")
    try:
        if not org_name:
            return []
        
        # List datasets for the specific organization/user
        items = api.list_datasets(author=org_name, token=token)
        repo_ids = [getattr(d, "id", None) or getattr(d, "repo_id", None) for d in items]
        repo_ids = [r for r in repo_ids if r]
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for r in repo_ids:
            if r not in seen:
                unique.append(r)
                seen.add(r)
        return unique[:1000]
    except Exception as e:
        print(f"Error searching datasets: {e}")
        return []


def fetch_stats_for_selected(selected_datasets: List[str], progress=gr.Progress()):
    """Fetch statistics for selected datasets"""
    if not selected_datasets:
        return "Please select at least one dataset"
    
    token = os.environ.get("HF_TOKEN")
    total_episodes = 0
    
    # Separate v3 and non-v3 datasets, organize by date
    from collections import defaultdict
    import re
    from datetime import datetime
    
    v3_by_date = defaultdict(list)  # date -> list of (repo_id, episodes, stats)
    non_v3_results = []
    errors = []
    
    for i, repo_id in enumerate(selected_datasets):
        try:
            progress((i + 1) / len(selected_datasets), desc=f"Processing {repo_id}...")
            stats = get_dataset_stats(repo_id, hf_token=token)
            
            if stats.get("error"):
                errors.append(f"❌ {repo_id}: Error - {stats['error']}")
            else:
                episodes = stats['total_episodes']
                total_episodes += episodes
                
                # Check if v3 format
                is_v3 = stats.get("format_version") == "v3.0"
                
                if is_v3:
                    # Try to extract date from repo_id (format: org/DDMMYYYY-name)
                    date_match = re.search(r'/(\d{8})', repo_id)
                    if date_match:
                        date_str = date_match.group(1)
                        try:
                            # Parse as DDMMYYYY
                            date_obj = datetime.strptime(date_str, '%d%m%Y')
                            date_key = date_obj.strftime('%Y-%m-%d')  # ISO format for sorting
                            date_display = date_obj.strftime('%B %d, %Y')  # Nice display format
                        except ValueError:
                            date_key = date_str
                            date_display = date_str
                        
                        v3_by_date[date_key].append({
                            'repo_id': repo_id,
                            'episodes': episodes,
                            'date_display': date_display,
                            'stats': stats
                        })
                    else:
                        # v3 but no date in name
                        v3_by_date['unknown'].append({
                            'repo_id': repo_id,
                            'episodes': episodes,
                            'date_display': 'Unknown Date',
                            'stats': stats
                        })
                else:
                    non_v3_results.append(f"{repo_id}: **{episodes}** episodes")
            
        except Exception as e:
            errors.append(f"❌ {repo_id}: Error - {str(e)}")
    
    def format_duration(seconds):
        """Format duration as hours, minutes, seconds"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    # Calculate total duration across all datasets
    total_duration_seconds = 0
    for datasets in v3_by_date.values():
        for d in datasets:
            info_meta = d['stats'].get('info_metadata', {})
            if info_meta.get('total_frames'):
                fps = info_meta.get('fps', 30)
                total_duration_seconds += info_meta['total_frames'] / fps
    
    # Build output with total episodes and duration
    duration_display = f" • {format_duration(total_duration_seconds)}" if total_duration_seconds > 0 else ""
    output = [f"## Total Episodes: {total_episodes}{duration_display}\n"]
    
    # Display v3 datasets grouped by date
    if v3_by_date:
        
        # Sort dates (most recent first)
        sorted_dates = sorted([k for k in v3_by_date.keys() if k != 'unknown'], reverse=True)
        if 'unknown' in v3_by_date:
            sorted_dates.append('unknown')
        
        for date_key in sorted_dates:
            datasets = v3_by_date[date_key]
            date_display = datasets[0]['date_display']
            date_total_episodes = sum(d['episodes'] for d in datasets)
            
            # Calculate total duration for the day
            date_total_seconds = 0
            for d in datasets:
                info_meta = d['stats'].get('info_metadata', {})
                if info_meta.get('total_frames'):
                    fps = info_meta.get('fps', 30)
                    date_total_seconds += info_meta['total_frames'] / fps
            
            # Format total duration
            duration_str = f" • {format_duration(date_total_seconds)}"
            
            output.append(f"\n**{date_display}** — Total: **{date_total_episodes} episodes**{duration_str}")
            
            for dataset in sorted(datasets, key=lambda x: x['repo_id']):
                repo_name = dataset['repo_id'].split('/')[-1]  # Just the dataset name
                episodes = dataset['episodes']
                
                # Add metadata if available
                info_meta = dataset['stats'].get('info_metadata', {})
                
                # Calculate duration in seconds from frames
                duration_str = ""
                if info_meta.get('total_frames'):
                    total_frames = info_meta['total_frames']
                    fps = info_meta.get('fps', 30)  # Default to 30 if not specified
                    duration_seconds = total_frames / fps
                    duration_str = f" • {format_duration(duration_seconds)}"
                
                output.append(f"- `{repo_name}`: **{episodes} episodes**{duration_str}")
    
    # Display non-v3 datasets
    if non_v3_results:
        output.extend(non_v3_results)
    
    # Display errors at the end
    if errors:
        output.append("\n### ⚠️ Errors")
        output.extend(errors)
    
    return "\n".join(output)


# Build the Gradio interface
with gr.Blocks(title="LeRobot Dataset Stats Viewer") as demo:
    gr.Markdown("**View statistics for Hugging Face datasets (LeRobot format).**")
    
    # Get user's organizations
    _user_orgs = get_user_organizations()
    _initial_datasets = search_datasets_fn(_user_orgs[0]) if _user_orgs else []
    
    # State to track current dataset choices
    current_choices = gr.State(_initial_datasets)
    
    with gr.Row():
        org_dropdown = gr.Dropdown(
            label="Select Organization",
            choices=_user_orgs,
            value=_user_orgs[0] if _user_orgs else None,
            interactive=True,
        )
        load_btn = gr.Button("Load Datasets", variant="secondary")
    
    dataset_checkboxes = gr.CheckboxGroup(
        label="Select datasets",
        choices=_initial_datasets,
        interactive=True,
    )
    
    with gr.Row():
        select_all_btn = gr.Button("Select All", size="sm")
        deselect_all_btn = gr.Button("Deselect All", size="sm")
    
    fetch_btn = gr.Button("Fetch Statistics", variant="primary")
    
    stats_output = gr.Markdown(
        label="Dataset Statistics",
        value="Select datasets and click 'Fetch Statistics'"
    )
    
    # Event handlers
    def load_datasets_from_org(org_name):
        results = search_datasets_fn(org_name)
        return [
            gr.update(choices=results, value=[]),  # Update checkboxes
            results  # Update state
        ]
    
    # Load datasets on button click or dropdown change
    load_btn.click(
        load_datasets_from_org,
        inputs=org_dropdown,
        outputs=[dataset_checkboxes, current_choices],
    )
    
    org_dropdown.change(
        load_datasets_from_org,
        inputs=org_dropdown,
        outputs=[dataset_checkboxes, current_choices],
    )
    
    # Select/Deselect all buttons
    select_all_btn.click(
        lambda choices: gr.update(value=choices),
        inputs=current_choices,
        outputs=dataset_checkboxes,
    )
    
    deselect_all_btn.click(
        lambda: gr.update(value=[]),
        outputs=dataset_checkboxes,
    )
    
    fetch_btn.click(
        fetch_stats_for_selected,
        inputs=dataset_checkboxes,
        outputs=stats_output,
    )


if __name__ == "__main__":
    demo.launch()
