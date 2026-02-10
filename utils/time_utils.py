"""
Time utility functions for consistent timestamp formatting.
"""


def seconds_to_mmss(seconds: float) -> str:
    """
    Convert a float number of seconds to a MM:SS formatted string.

    Args:
        seconds: Time in seconds (can be float).

    Returns:
        String in MM:SS format.
    """
    total_seconds = int(round(seconds))
    minutes = total_seconds // 60
    secs = total_seconds % 60
    return f"{minutes:02d}:{secs:02d}"


def format_duration(total_seconds: float) -> str:
    """
    Format a duration in seconds to a human-readable MM:SS string.

    Args:
        total_seconds: Duration in seconds.

    Returns:
        Formatted duration string.
    """
    return seconds_to_mmss(total_seconds)


def format_segment(start_sec: float, end_sec: float, emotion: str) -> dict:
    """
    Create a formatted segment dictionary with MM:SS timestamps.

    Args:
        start_sec: Start time in seconds.
        end_sec: End time in seconds.
        emotion: Detected emotion label.

    Returns:
        Dictionary with 'start', 'end', and 'emotion' keys.
    """
    return {
        "start": seconds_to_mmss(start_sec),
        "end": seconds_to_mmss(end_sec),
        "emotion": emotion
    }


def parse_mmss_to_seconds(mmss: str) -> float:
    """
    Parse a MM:SS formatted string back to seconds.

    Args:
        mmss: Time string in MM:SS format.

    Returns:
        Time in seconds as float.
    """
    parts = mmss.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid MM:SS format: {mmss}")
    minutes = int(parts[0])
    seconds = int(parts[1])
    return float(minutes * 60 + seconds)
