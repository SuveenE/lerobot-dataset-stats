import os
import re
import json
import logging
from typing import Optional, Dict, Any
from huggingface_hub import HfApi, hf_hub_download


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def get_dataset_stats(
    repo_id: str,
    hf_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Get statistics for a Hugging Face dataset without downloading all files.
    
    Supports both v2.1 and v3.0 LeRobot dataset formats.
    
    Args:
        repo_id: The HuggingFace dataset repo ID
        hf_token: Optional HuggingFace token for private datasets
        
    Returns:
        Dictionary containing dataset statistics:
        - total_episodes: Number of episodes (from info.json for v3.0, or counted for v2.1)
        - episode_numbers: List of episode numbers found
        - total_parquet_files: Total number of parquet files
        - total_video_files: Total number of video files (if present)
        - info_metadata: Complete metadata from info.json (if present)
        - codebase_version: Dataset version (if present)
        - format_version: Detected format version (v2.1 or v3.0)
    """
    api = HfApi()
    token = hf_token or os.environ.get("HF_TOKEN")
    
    logger.info(f"Fetching stats for dataset: {repo_id}")
    
    stats = {
        "repo_id": repo_id,
        "total_episodes": 0,
        "episode_numbers": [],
        "total_parquet_files": 0,
        "total_video_files": 0,
        "info_metadata": None,
        "codebase_version": None,
        "format_version": None,
        "error": None,
    }
    
    try:
        # Try to fetch metadata from info.json first to determine version
        try:
            info_path = hf_hub_download(
                repo_id=repo_id,
                filename="meta/info.json",
                repo_type="dataset",
                token=token,
            )
            
            with open(info_path, "r") as f:
                info_data = json.load(f)
                stats["info_metadata"] = info_data
                stats["codebase_version"] = info_data.get("codebase_version")
                
                # Determine format version from codebase_version
                if stats["codebase_version"] and stats["codebase_version"].startswith("v3"):
                    stats["format_version"] = "v3.0"
                    # In v3.0, total_episodes is in info.json
                    stats["total_episodes"] = info_data.get("total_episodes", 0)
                    # Generate episode numbers list
                    if stats["total_episodes"] > 0:
                        stats["episode_numbers"] = list(range(stats["total_episodes"]))
                    logger.info(f"Detected v3.0 format with {stats['total_episodes']} episodes from info.json")
                else:
                    stats["format_version"] = "v2.1"
                    logger.info("Detected v2.1 format")
                
            logger.info("Successfully fetched metadata from info.json")
        except Exception as e:
            logger.warning(f"Could not fetch info.json: {str(e)}")
            # Assume v2.1 if we can't read info.json
            stats["format_version"] = "v2.1"
        
        # List all files in the repository
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)
        
        # Detect format and count files based on version
        if stats["format_version"] == "v3.0":
            # v3.0 format: data/chunk-XXX/file-XXX.parquet and videos/{camera}/chunk-XXX/file-XXX.mp4
            parquet_pattern = re.compile(r"data/chunk-\d+/file-\d+\.parquet")
            video_pattern = re.compile(r"videos/.+/chunk-\d+/file-\d+\.mp4")
            
            for file_path in files:
                if parquet_pattern.search(file_path):
                    stats["total_parquet_files"] += 1
                elif video_pattern.search(file_path):
                    stats["total_video_files"] += 1
        else:
            # v2.1 format: episode_XXXX.parquet and episode_XXXX.mp4
            parquet_pattern = re.compile(r"data/chunk-\d+/episode_(\d+)\.parquet")
            episode_numbers = set()
            
            for file_path in files:
                match = parquet_pattern.search(file_path)
                if match:
                    episode_num = int(match.group(1))
                    episode_numbers.add(episode_num)
                    stats["total_parquet_files"] += 1
                
                # Count video files (v2.1 format)
                if file_path.endswith(".mp4") and "episode_" in file_path:
                    stats["total_video_files"] += 1
            
            # Update stats if we didn't get episodes from info.json
            if episode_numbers:
                stats["episode_numbers"] = sorted(list(episode_numbers))
                stats["total_episodes"] = len(episode_numbers)
        
        logger.info(
            f"Stats for {repo_id} ({stats['format_version']}): "
            f"{stats['total_episodes']} episodes, "
            f"{stats['total_parquet_files']} parquet files, "
            f"{stats['total_video_files']} video files"
        )
        
    except Exception as e:
        error_msg = f"Error fetching stats: {str(e)}"
        logger.error(error_msg)
        stats["error"] = error_msg
    
    return stats


def format_stats_display(stats: Dict[str, Any]) -> str:
    """Format stats dictionary into a readable string for display.
    
    Args:
        stats: Dictionary of dataset statistics
        
    Returns:
        Formatted string for display
    """
    if stats.get("error"):
        return f"❌ Error: {stats['error']}"
    
    lines = []
    lines.append(f"📊 **Dataset Statistics for {stats['repo_id']}**")
    lines.append("")
    
    # Format version
    if stats.get("format_version"):
        lines.append(f"**Format Version:** {stats['format_version']}")
    
    # Version info
    if stats.get("codebase_version"):
        lines.append(f"**Codebase Version:** {stats['codebase_version']}")
    
    lines.append("")
    
    # Basic stats
    lines.append(f"**Total Episodes:** {stats['total_episodes']}")
    lines.append(f"**Total Parquet Files:** {stats['total_parquet_files']}")
    lines.append(f"**Total Video Files:** {stats['total_video_files']}")
    
    # Episode range (mainly for v2.1 or when episode numbers are sequential)
    if stats["episode_numbers"] and len(stats["episode_numbers"]) > 0:
        episode_nums = stats["episode_numbers"]
        lines.append(f"**Episode Range:** {episode_nums[0]} to {episode_nums[-1]}")
        
        # Check for gaps in episodes (only for v2.1)
        if stats.get("format_version") == "v2.1":
            expected = list(range(episode_nums[0], episode_nums[-1] + 1))
            missing = set(expected) - set(episode_nums)
            if missing:
                lines.append(f"**⚠️ Missing Episodes:** {sorted(list(missing))}")
    
    # Additional metadata from info.json
    if stats.get("info_metadata"):
        info = stats["info_metadata"]
        lines.append("")
        lines.append("**Metadata from info.json:**")
        
        # Show key metadata fields (v3.0 has more fields)
        if stats.get("format_version") == "v3.0":
            metadata_fields = [
                ("fps", "FPS"),
                ("robot_type", "Robot Type"),
                ("total_frames", "Total Frames"),
                ("total_tasks", "Total Tasks"),
                ("chunks_size", "Chunks Size"),
                ("data_files_size_in_mb", "Data Files Size (MB)"),
                ("video_files_size_in_mb", "Video Files Size (MB)"),
            ]
        else:
            # v2.1 fields
            metadata_fields = [
                ("fps", "FPS"),
                ("robot_type", "Robot Type"),
                ("total_episodes", "Total Episodes (from metadata)"),
                ("total_videos", "Total Videos (from metadata)"),
                ("total_tasks", "Total Tasks"),
                ("total_frames", "Total Frames"),
            ]
        
        for key, label in metadata_fields:
            if key in info:
                lines.append(f"  - **{label}:** {info[key]}")
    
    return "\n".join(lines)


def compare_metadata_with_actual(stats: Dict[str, Any]) -> str:
    """Compare metadata from info.json with actual file counts.
    
    Args:
        stats: Dictionary of dataset statistics
        
    Returns:
        Comparison report string
    """
    if not stats.get("info_metadata"):
        return "No metadata available for comparison"
    
    info = stats["info_metadata"]
    lines = []
    lines.append("**📋 Metadata vs Actual Comparison:**")
    lines.append("")
    
    # Compare episodes
    metadata_episodes = info.get("total_episodes", "N/A")
    actual_episodes = stats["total_episodes"]
    match_episodes = "✅" if metadata_episodes == actual_episodes else "❌"
    lines.append(
        f"{match_episodes} **Episodes:** Metadata={metadata_episodes}, Actual={actual_episodes}"
    )
    
    # Compare videos
    metadata_videos = info.get("total_videos", "N/A")
    actual_videos = stats["total_video_files"]
    match_videos = "✅" if metadata_videos == actual_videos else "❌"
    lines.append(
        f"{match_videos} **Videos:** Metadata={metadata_videos}, Actual={actual_videos}"
    )
    
    return "\n".join(lines)

