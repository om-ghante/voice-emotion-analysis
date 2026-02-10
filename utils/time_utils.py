def seconds_to_mmss(seconds: float) -> str:
    total_seconds = int(round(seconds))
    minutes = total_seconds // 60
    secs = total_seconds % 60
    return f"{minutes:02d}:{secs:02d}"

def format_duration(total_seconds: float) -> str:
    return seconds_to_mmss(total_seconds)

def format_segment(start_sec: float, end_sec: float, emotion: str) -> dict:
    return {
        "start": seconds_to_mmss(start_sec),
        "end": seconds_to_mmss(end_sec),
        "emotion": emotion
    }

def parse_mmss_to_seconds(mmss: str) -> float:
    parts = mmss.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid MM:SS format: {mmss}")
    minutes = int(parts[0])
    seconds = int(parts[1])
    return float(minutes * 60 + seconds)
