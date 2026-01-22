"""
Data Preprocessing Pipeline for CBT Prediction Platform

This script prepares raw data for the training pipeline by:
1. Reading raw data files from categorized folders
2. Normalizing column names and formats
3. Converting ALL timestamps to UTC (handling DST for America/Chicago)
4. Outputting files in the format expected by prepare_data.py

Input Structure (raw_data/):
    Final Merge _ CBT Files/
        user1_cbt.csv, user2_cbt.csv, ...
    Final Merge _ Heart Rate Files/
        user1_heart_rate.csv, user2_heart_rate.csv, ...
    Final Merge _ Wrist Temperature Files/
        user1_wrist_temp.csv, user2_wrist_temp.csv, ...
    Final Merge_ Environmental Files/
        user1_environmental.csv, user2_environmental.csv, ...

Output Structure (preprocessed_data/):
    user1/
        heart_rate.csv
        skin_temperature.csv
        environmental.csv
        cbt_labels.csv
    user2/
        ...
    all_users/
        heart_rate.csv          (combined, with user_id column)
        skin_temperature.csv
        environmental.csv
        cbt_labels.csv

ALL OUTPUT TIMESTAMPS ARE IN UTC FORMAT: 2025-04-05T00:00:00Z
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import warnings
from datetime import datetime
import pytz

warnings.filterwarnings('ignore')

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable


# ============================================
# CONFIGURATION
# ============================================

class Config:
    """Configuration for the preprocessing pipeline."""
    
    SCRIPT_DIR = Path(__file__).parent.resolve()
    RAW_DATA_DIR = SCRIPT_DIR / "raw_data"
    OUTPUT_DIR = SCRIPT_DIR / "preprocessed_data"
    
    # Folder names in raw_data (case-insensitive matching)
    CBT_FOLDER_PATTERN = "cbt"
    HR_FOLDER_PATTERN = "heart rate"
    WRIST_TEMP_FOLDER_PATTERN = "wrist temp"
    ENV_FOLDER_PATTERN = "environmental"
    
    # Output filenames (must match what DataLoader expects)
    HR_OUTPUT = "heart_rate.csv"
    SKIN_TEMP_OUTPUT = "skin_temperature.csv"
    ENV_OUTPUT = "environmental.csv"
    CBT_OUTPUT = "cbt_labels.csv"
    
    # Timezone for local data (CBT, Environmental, Wrist Temp are in local time)
    # Heart Rate from Fitbit is already in UTC
    LOCAL_TIMEZONE = "America/Chicago"
    
    # Standard output timestamp format
    TIMESTAMP_OUTPUT_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


# ============================================
# TIMESTAMP UTILITIES
# ============================================

def parse_flexible_datetime(date_str: str, time_str: Optional[str] = None) -> Optional[datetime]:
    """
    Parse datetime from various formats.
    
    Handles:
    - Date formats: YYYY-MM-DD, MM/DD/YYYY, MM/DD/YY, M/D/YYYY, M/D/YY
    - Time formats: HH:MM:SS, HH:MM, H:MM AM/PM, HH:MM AM/PM
    - Combined: YYYY-MM-DDTHH:MM, YYYY-MM-DD HH:MM:SS
    
    Args:
        date_str: Date string or combined datetime string
        time_str: Optional separate time string
    
    Returns:
        datetime object or None if parsing fails
    """
    if pd.isna(date_str) or date_str is None:
        return None
    
    date_str = str(date_str).strip()
    
    if time_str is not None and not pd.isna(time_str):
        time_str = str(time_str).strip()
        combined = f"{date_str} {time_str}"
    else:
        combined = date_str
    
    # List of formats to try (most specific first)
    formats = [
        # ISO formats (already has T separator)
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        
        # Standard formats with space separator
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        
        # US formats with 12-hour time
        "%m/%d/%Y %I:%M:%S %p",
        "%m/%d/%Y %I:%M %p",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y",
        
        # US formats with 2-digit year
        "%m/%d/%y %I:%M:%S %p",
        "%m/%d/%y %I:%M %p",
        "%m/%d/%y %H:%M:%S",
        "%m/%d/%y %H:%M",
        "%m/%d/%y",
        
        # Formats without leading zeros
        "%-m/%-d/%Y %I:%M %p",
        "%-m/%-d/%Y %I:%M:%S %p",
        "%-m/%-d/%y %I:%M %p",
        "%-m/%-d/%y %I:%M:%S %p",
    ]
    
    for fmt in formats:
        try:
            # Handle formats with %-m and %-d (no leading zeros) on Windows
            # Windows doesn't support %-m, so we need to handle it differently
            if "%-" in fmt:
                continue  # Skip these, we'll handle with regex below
            return datetime.strptime(combined, fmt)
        except ValueError:
            continue
    
    # Try parsing with regex for flexible formats
    # Pattern for dates like "4/5/2025" or "04/05/2025"
    date_pattern = r'^(\d{1,2})/(\d{1,2})/(\d{2,4})'
    time_12h_pattern = r'(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(AM|PM|am|pm)?'
    time_24h_pattern = r'(\d{1,2}):(\d{2})(?::(\d{2}))?$'
    
    try:
        date_match = re.match(date_pattern, combined)
        if date_match:
            month = int(date_match.group(1))
            day = int(date_match.group(2))
            year = int(date_match.group(3))
            
            # Handle 2-digit year
            if year < 100:
                year += 2000
            
            # Find time portion
            time_portion = combined[date_match.end():].strip()
            hour, minute, second = 0, 0, 0
            
            if time_portion:
                # Try 12-hour format first
                time_match = re.search(time_12h_pattern, time_portion, re.IGNORECASE)
                if time_match:
                    hour = int(time_match.group(1))
                    minute = int(time_match.group(2))
                    second = int(time_match.group(3)) if time_match.group(3) else 0
                    ampm = time_match.group(4)
                    
                    if ampm:
                        ampm = ampm.upper()
                        if ampm == 'PM' and hour != 12:
                            hour += 12
                        elif ampm == 'AM' and hour == 12:
                            hour = 0
                else:
                    # Try 24-hour format
                    time_match = re.search(time_24h_pattern, time_portion)
                    if time_match:
                        hour = int(time_match.group(1))
                        minute = int(time_match.group(2))
                        second = int(time_match.group(3)) if time_match.group(3) else 0
            
            return datetime(year, month, day, hour, minute, second)
    except (ValueError, AttributeError):
        pass
    
    # Try ISO format with T separator
    try:
        if 'T' in combined:
            # Remove Z suffix if present
            clean = combined.rstrip('Z')
            parts = clean.split('T')
            if len(parts) == 2:
                date_part = parts[0]
                time_part = parts[1]
                
                date_components = date_part.split('-')
                time_components = time_part.split(':')
                
                year = int(date_components[0])
                month = int(date_components[1])
                day = int(date_components[2])
                hour = int(time_components[0])
                minute = int(time_components[1])
                second = int(time_components[2]) if len(time_components) > 2 else 0
                
                return datetime(year, month, day, hour, minute, second)
    except (ValueError, IndexError):
        pass
    
    return None


def convert_local_to_utc(dt: datetime, local_tz_str: str = "America/Chicago") -> datetime:
    """
    Convert a naive local datetime to UTC.
    
    Handles DST transitions automatically.
    
    Args:
        dt: Naive datetime in local time
        local_tz_str: Timezone string (e.g., "America/Chicago")
    
    Returns:
        Datetime in UTC
    """
    if dt is None:
        return None
    
    local_tz = pytz.timezone(local_tz_str)
    
    try:
        # Localize the naive datetime (handles DST)
        local_dt = local_tz.localize(dt, is_dst=None)
    except pytz.exceptions.AmbiguousTimeError:
        # During fall-back, assume standard time
        local_dt = local_tz.localize(dt, is_dst=False)
    except pytz.exceptions.NonExistentTimeError:
        # During spring-forward, shift forward
        local_dt = local_tz.localize(dt, is_dst=True)
    
    # Convert to UTC
    utc_dt = local_dt.astimezone(pytz.UTC)
    
    return utc_dt.replace(tzinfo=None)  # Return naive UTC datetime


def format_utc_timestamp(dt: datetime) -> str:
    """
    Format datetime as UTC timestamp string.
    
    Args:
        dt: Datetime object (assumed to be UTC)
    
    Returns:
        String in format "2025-04-05T00:00:00Z"
    """
    if dt is None:
        return None
    return dt.strftime(Config.TIMESTAMP_OUTPUT_FORMAT)


def parse_and_convert_to_utc(
    date_str: str, 
    time_str: Optional[str] = None,
    is_utc: bool = False,
    local_tz: str = "America/Chicago"
) -> Optional[str]:
    """
    Parse a datetime string and convert to UTC timestamp.
    
    Args:
        date_str: Date string or combined datetime string
        time_str: Optional separate time string
        is_utc: If True, assume input is already UTC
        local_tz: Local timezone for conversion
    
    Returns:
        UTC timestamp string or None
    """
    dt = parse_flexible_datetime(date_str, time_str)
    
    if dt is None:
        return None
    
    if not is_utc:
        dt = convert_local_to_utc(dt, local_tz)
    
    return format_utc_timestamp(dt)


# ============================================
# UTILITIES
# ============================================

def extract_user_id(filename: str) -> str:
    """
    Extract user ID from filename.
    
    Expected format: user1_something.csv, user2_data.csv, etc.
    
    Args:
        filename: Name of the file
    
    Returns:
        User ID (e.g., "user1", "user2")
    """
    filename_lower = filename.lower()
    
    # Pattern: user followed by number at start of filename
    match = re.match(r'^(user\d+)', filename_lower)
    if match:
        return match.group(1)
    
    # Fallback: first part before underscore
    parts = Path(filename).stem.split('_')
    if parts:
        return parts[0].lower()
    
    return "unknown"


def find_folder(base_dir: Path, pattern: str) -> Optional[Path]:
    """
    Find a folder containing the pattern in its name.
    
    Args:
        base_dir: Directory to search in
        pattern: Pattern to match (case-insensitive)
    
    Returns:
        Path to matching folder, or None
    """
    pattern_lower = pattern.lower()
    
    for item in base_dir.iterdir():
        if item.is_dir() and pattern_lower in item.name.lower():
            return item
    
    return None


def load_csv_flexible(filepath: Path) -> pd.DataFrame:
    """
    Load CSV with flexible parsing.
    
    Handles various delimiters and encodings.
    
    Args:
        filepath: Path to CSV file
    
    Returns:
        DataFrame
    """
    try:
        # Try standard CSV first
        df = pd.read_csv(filepath)
        if len(df.columns) > 1:
            return df
    except:
        pass
    
    try:
        # Try with different encoding
        df = pd.read_csv(filepath, encoding='latin-1')
        if len(df.columns) > 1:
            return df
    except:
        pass
    
    try:
        # Try tab-separated
        df = pd.read_csv(filepath, sep='\t')
        if len(df.columns) > 1:
            return df
    except:
        pass
    
    # Last resort
    return pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip')


# ============================================
# HEART RATE PROCESSOR
# ============================================

class HeartRateProcessor:
    """
    Process Fitbit heart rate files.
    
    Input format (from Fitbit - already in UTC):
        timestamp, beats per minute
        2025-04-04T00:00:08Z, 67
    
    Output format:
        timestamp, beats per minute
        2025-04-04T00:00:08Z, 67
    """
    
    def __init__(self):
        self.data: Dict[str, pd.DataFrame] = {}  # user_id -> DataFrame
    
    def process_folder(self, folder: Path) -> "HeartRateProcessor":
        """Process all heart rate files in folder."""
        if folder is None or not folder.exists():
            print("  No heart rate folder found")
            return self
        
        files = list(folder.glob("*.csv")) + list(folder.glob("*.CSV"))
        print(f"  Found {len(files)} heart rate files in {folder.name}")
        
        for filepath in tqdm(files, desc="  Processing HR", disable=not HAS_TQDM):
            user_id = extract_user_id(filepath.name)
            df = self._process_file(filepath)
            
            if df is not None and len(df) > 0:
                if user_id in self.data:
                    self.data[user_id] = pd.concat([self.data[user_id], df], ignore_index=True)
                else:
                    self.data[user_id] = df
        
        # Sort and deduplicate per user
        for user_id in self.data:
            self.data[user_id] = (
                self.data[user_id]
                .sort_values("timestamp")
                .drop_duplicates(subset=["timestamp"], keep="first")
                .reset_index(drop=True)
            )
            print(f"    {user_id}: {len(self.data[user_id]):,} samples")
        
        return self
    
    def _process_file(self, filepath: Path) -> Optional[pd.DataFrame]:
        """Process a single heart rate file."""
        try:
            df = load_csv_flexible(filepath)
        except Exception as e:
            print(f"    Error loading {filepath.name}: {e}")
            return None
        
        if df.empty:
            return None
        
        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()
        
        # Find timestamp column
        ts_col = None
        for c in df.columns:
            if "timestamp" in c or "time" in c:
                ts_col = c
                break
        
        # Find heart rate column
        hr_col = None
        for c in df.columns:
            if "beat" in c or "bpm" in c or "heart" in c:
                hr_col = c
                break
        
        if ts_col is None or hr_col is None:
            print(f"    Warning: Missing columns in {filepath.name}")
            print(f"    Available: {list(df.columns)}")
            return None
        
        # Create output DataFrame
        result = pd.DataFrame()
        
        # Heart rate timestamps are already in UTC - just standardize format
        timestamps = []
        for ts in df[ts_col]:
            if pd.isna(ts):
                timestamps.append(None)
                continue
            
            ts_str = str(ts).strip()
            
            # If already has Z suffix, it's UTC
            if ts_str.endswith('Z'):
                dt = parse_flexible_datetime(ts_str)
                timestamps.append(format_utc_timestamp(dt) if dt else None)
            else:
                # Fitbit data should be UTC, but parse and format consistently
                dt = parse_flexible_datetime(ts_str)
                timestamps.append(format_utc_timestamp(dt) if dt else None)
        
        result["timestamp"] = timestamps
        result["beats per minute"] = pd.to_numeric(df[hr_col], errors="coerce")
        
        # Remove invalid rows
        result = result.dropna(subset=["timestamp", "beats per minute"])
        result = result[(result["beats per minute"] >= 30) & (result["beats per minute"] <= 220)]
        
        return result
    
    def save(self, output_dir: Path, per_user: bool = True, combined: bool = True) -> Dict[str, Path]:
        """
        Save processed heart rate data.
        
        Args:
            output_dir: Base output directory
            per_user: Save individual user files
            combined: Save combined file with user_id column
        
        Returns:
            Dict of saved file paths
        """
        saved = {}
        
        if per_user:
            for user_id, df in self.data.items():
                user_dir = output_dir / user_id
                user_dir.mkdir(parents=True, exist_ok=True)
                
                filepath = user_dir / Config.HR_OUTPUT
                df.to_csv(filepath, index=False)
                saved[f"{user_id}_hr"] = filepath
        
        if combined and self.data:
            all_users_dir = output_dir / "all_users"
            all_users_dir.mkdir(parents=True, exist_ok=True)
            
            combined_df = []
            for user_id, df in self.data.items():
                df_copy = df.copy()
                df_copy["user_id"] = user_id
                combined_df.append(df_copy)
            
            combined_df = pd.concat(combined_df, ignore_index=True)
            combined_df = combined_df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
            
            filepath = all_users_dir / Config.HR_OUTPUT
            combined_df.to_csv(filepath, index=False)
            saved["combined_hr"] = filepath
        
        return saved


# ============================================
# WRIST TEMPERATURE PROCESSOR
# ============================================

class WristTemperatureProcessor:
    """
    Process Fitbit wrist/skin temperature files.
    
    Input format (from Fitbit - LOCAL TIME):
        recorded_time, temperature
        2025-04-04T00:00, -2.095291443
    
    Output format (converted to UTC):
        timestamp, temperature
        2025-04-04T05:00:00Z, -2.095291443
    
    Note: Temperature is a deviation from baseline, not absolute temperature.
    """
    
    def __init__(self):
        self.data: Dict[str, pd.DataFrame] = {}
    
    def process_folder(self, folder: Path) -> "WristTemperatureProcessor":
        """Process all wrist temperature files in folder."""
        if folder is None or not folder.exists():
            print("  No wrist temperature folder found")
            return self
        
        files = list(folder.glob("*.csv")) + list(folder.glob("*.CSV"))
        print(f"  Found {len(files)} wrist temperature files in {folder.name}")
        
        for filepath in tqdm(files, desc="  Processing Skin Temp", disable=not HAS_TQDM):
            user_id = extract_user_id(filepath.name)
            df = self._process_file(filepath)
            
            if df is not None and len(df) > 0:
                if user_id in self.data:
                    self.data[user_id] = pd.concat([self.data[user_id], df], ignore_index=True)
                else:
                    self.data[user_id] = df
        
        # Sort and deduplicate per user
        for user_id in self.data:
            self.data[user_id] = (
                self.data[user_id]
                .sort_values("timestamp")
                .drop_duplicates(subset=["timestamp"], keep="first")
                .reset_index(drop=True)
            )
            print(f"    {user_id}: {len(self.data[user_id]):,} samples")
        
        return self
    
    def _process_file(self, filepath: Path) -> Optional[pd.DataFrame]:
        """Process a single wrist temperature file."""
        try:
            df = load_csv_flexible(filepath)
        except Exception as e:
            print(f"    Error loading {filepath.name}: {e}")
            return None
        
        if df.empty:
            return None
        
        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()
        
        # Find timestamp column
        ts_col = None
        for c in df.columns:
            if "recorded" in c or "timestamp" in c or "time" in c:
                ts_col = c
                break
        
        # Find temperature column
        temp_col = None
        for c in df.columns:
            if "temp" in c:
                temp_col = c
                break
        
        if ts_col is None or temp_col is None:
            print(f"    Warning: Missing columns in {filepath.name}")
            print(f"    Available: {list(df.columns)}")
            return None
        
        # Create output DataFrame
        result = pd.DataFrame()
        
        # Convert local timestamps to UTC
        timestamps = []
        for ts in df[ts_col]:
            utc_ts = parse_and_convert_to_utc(
                str(ts) if not pd.isna(ts) else None,
                is_utc=False,
                local_tz=Config.LOCAL_TIMEZONE
            )
            timestamps.append(utc_ts)
        
        result["timestamp"] = timestamps
        result["temperature"] = pd.to_numeric(df[temp_col], errors="coerce")
        
        result = result.dropna(subset=["timestamp"])
        
        return result
    
    def save(self, output_dir: Path, per_user: bool = True, combined: bool = True) -> Dict[str, Path]:
        """Save processed wrist temperature data."""
        saved = {}
        
        if per_user:
            for user_id, df in self.data.items():
                user_dir = output_dir / user_id
                user_dir.mkdir(parents=True, exist_ok=True)
                
                filepath = user_dir / Config.SKIN_TEMP_OUTPUT
                df.to_csv(filepath, index=False)
                saved[f"{user_id}_skin"] = filepath
        
        if combined and self.data:
            all_users_dir = output_dir / "all_users"
            all_users_dir.mkdir(parents=True, exist_ok=True)
            
            combined_df = []
            for user_id, df in self.data.items():
                df_copy = df.copy()
                df_copy["user_id"] = user_id
                combined_df.append(df_copy)
            
            combined_df = pd.concat(combined_df, ignore_index=True)
            combined_df = combined_df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
            
            filepath = all_users_dir / Config.SKIN_TEMP_OUTPUT
            combined_df.to_csv(filepath, index=False)
            saved["combined_skin"] = filepath
        
        return saved


# ============================================
# ENVIRONMENTAL PROCESSOR
# ============================================

class EnvironmentalProcessor:
    """
    Process Govee environmental monitor files.
    
    Input format (from Govee - LOCAL TIME):
        Timestamp for sample frequency every 1 min min, PM2.5(µg/m³), Temperature_Fahrenheit, Relative_Humidity
        2025-04-05 00:00:00, 0, 69.98, 50.6
    
    Output format (converted to UTC, temperature kept in Fahrenheit):
        timestamp, Temperature_Fahrenheit, Relative_Humidity
        2025-04-05T05:00:00Z, 69.98, 50.6
    
    Note: PM2.5 is dropped as it's not used.
    Note: Temperature is kept in Fahrenheit - DataLoader will convert if needed.
    """
    
    def __init__(self):
        self.data: Dict[str, pd.DataFrame] = {}
    
    def process_folder(self, folder: Path) -> "EnvironmentalProcessor":
        """Process all environmental files in folder."""
        if folder is None or not folder.exists():
            print("  No environmental folder found")
            return self
        
        files = list(folder.glob("*.csv")) + list(folder.glob("*.CSV"))
        print(f"  Found {len(files)} environmental files in {folder.name}")
        
        for filepath in tqdm(files, desc="  Processing Env", disable=not HAS_TQDM):
            user_id = extract_user_id(filepath.name)
            df = self._process_file(filepath)
            
            if df is not None and len(df) > 0:
                if user_id in self.data:
                    self.data[user_id] = pd.concat([self.data[user_id], df], ignore_index=True)
                else:
                    self.data[user_id] = df
        
        # Sort and deduplicate per user
        for user_id in self.data:
            self.data[user_id] = (
                self.data[user_id]
                .sort_values("timestamp")
                .drop_duplicates(subset=["timestamp"], keep="first")
                .reset_index(drop=True)
            )
            print(f"    {user_id}: {len(self.data[user_id]):,} samples")
        
        return self
    
    def _process_file(self, filepath: Path) -> Optional[pd.DataFrame]:
        """Process a single environmental file."""
        try:
            df = load_csv_flexible(filepath)
        except Exception as e:
            print(f"    Error loading {filepath.name}: {e}")
            return None
        
        if df.empty:
            return None
        
        # Normalize column names for detection
        original_columns = df.columns.tolist()
        normalized_columns = (
            df.columns
            .str.normalize("NFKC")
            .str.replace("\u00A0", " ", regex=False)
            .str.lower()
            .str.strip()
        )
        
        # Create column mapping
        col_map = {norm: orig for norm, orig in zip(normalized_columns, original_columns)}
        df.columns = normalized_columns
        
        # Find columns
        ts_col = None
        for c in df.columns:
            if c.startswith("timestamp") or c == "time" or c == "datetime":
                ts_col = c
                break
        
        temp_col = None
        for c in df.columns:
            if "temperature" in c and "pm" not in c:
                temp_col = c
                break
        
        humid_col = None
        for c in df.columns:
            if "humid" in c:
                humid_col = c
                break
        
        if ts_col is None:
            print(f"    Warning: No timestamp column in {filepath.name}")
            print(f"    Available: {original_columns}")
            return None
        
        if temp_col is None and humid_col is None:
            print(f"    Warning: No temp/humidity in {filepath.name}")
            return None
        
        # Build output with standardized column names
        result = pd.DataFrame()
        
        # Convert local timestamps to UTC
        timestamps = []
        for ts in df[ts_col]:
            utc_ts = parse_and_convert_to_utc(
                str(ts) if not pd.isna(ts) else None,
                is_utc=False,
                local_tz=Config.LOCAL_TIMEZONE
            )
            timestamps.append(utc_ts)
        
        result["timestamp"] = timestamps
        
        # Temperature (keep in Fahrenheit)
        if temp_col is not None:
            result["Temperature_Fahrenheit"] = pd.to_numeric(df[temp_col], errors="coerce")
        
        # Humidity
        if humid_col is not None:
            result["Relative_Humidity"] = pd.to_numeric(df[humid_col], errors="coerce")
        
        result = result.dropna(subset=["timestamp"])
        
        return result
    
    def save(self, output_dir: Path, per_user: bool = True, combined: bool = True) -> Dict[str, Path]:
        """Save processed environmental data."""
        saved = {}
        
        if per_user:
            for user_id, df in self.data.items():
                user_dir = output_dir / user_id
                user_dir.mkdir(parents=True, exist_ok=True)
                
                filepath = user_dir / Config.ENV_OUTPUT
                df.to_csv(filepath, index=False)
                saved[f"{user_id}_env"] = filepath
        
        if combined and self.data:
            all_users_dir = output_dir / "all_users"
            all_users_dir.mkdir(parents=True, exist_ok=True)
            
            combined_df = []
            for user_id, df in self.data.items():
                df_copy = df.copy()
                df_copy["user_id"] = user_id
                combined_df.append(df_copy)
            
            combined_df = pd.concat(combined_df, ignore_index=True)
            combined_df = combined_df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
            
            filepath = all_users_dir / Config.ENV_OUTPUT
            combined_df.to_csv(filepath, index=False)
            saved["combined_env"] = filepath
        
        return saved


# ============================================
# CBT PROCESSOR
# ============================================

class CBTProcessor:
    """
    Process CBT (Core Body Temperature) label files.
    
    Input format (manual measurements - LOCAL TIME):
        Date:, Time:, CBT (Deg F):
        4/5/2025, 8:59 PM, 98.5
    
    Output format (converted to UTC, temperature in Fahrenheit):
        timestamp, cbt_fahrenheit
        2025-04-06T01:59:00Z, 98.5
    
    Note: Date/time combined and converted to UTC.
    Note: Temperature kept in Fahrenheit.
    """
    
    def __init__(self):
        self.data: Dict[str, pd.DataFrame] = {}
    
    def process_folder(self, folder: Path) -> "CBTProcessor":
        """Process all CBT files in folder."""
        if folder is None or not folder.exists():
            print("  No CBT folder found")
            return self
        
        files = list(folder.glob("*.csv")) + list(folder.glob("*.CSV"))
        print(f"  Found {len(files)} CBT files in {folder.name}")
        
        for filepath in tqdm(files, desc="  Processing CBT", disable=not HAS_TQDM):
            user_id = extract_user_id(filepath.name)
            df = self._process_file(filepath)
            
            if df is not None and len(df) > 0:
                if user_id in self.data:
                    self.data[user_id] = pd.concat([self.data[user_id], df], ignore_index=True)
                else:
                    self.data[user_id] = df
        
        # Sort and deduplicate per user
        for user_id in self.data:
            self.data[user_id] = (
                self.data[user_id]
                .sort_values("timestamp")
                .drop_duplicates(subset=["timestamp"], keep="first")
                .reset_index(drop=True)
            )
            print(f"    {user_id}: {len(self.data[user_id])} measurements")
        
        return self
    
    def _process_file(self, filepath: Path) -> Optional[pd.DataFrame]:
        """Process a single CBT file."""
        try:
            df = load_csv_flexible(filepath)
        except Exception as e:
            print(f"    Error loading {filepath.name}: {e}")
            return None
        
        if df.empty:
            return None
        
        # Normalize column names for detection
        original_columns = df.columns.tolist()
        normalized = df.columns.str.lower().str.strip().str.rstrip(':')
        
        # Find columns by pattern
        date_col = None
        time_col = None
        cbt_col = None
        
        for i, c in enumerate(normalized):
            if c == "date":
                date_col = i
            elif c == "time":
                time_col = i
            elif "cbt" in c or "core" in c or "body" in c:
                cbt_col = i
        
        if date_col is None or time_col is None or cbt_col is None:
            print(f"    Warning: Missing columns in {filepath.name}")
            print(f"    Available: {original_columns}")
            print(f"    Normalized: {list(normalized)}")
            return None
        
        # Build output
        result = pd.DataFrame()
        
        # Convert local date/time to UTC
        timestamps = []
        for idx in range(len(df)):
            date_val = df.iloc[idx, date_col]
            time_val = df.iloc[idx, time_col]
            
            utc_ts = parse_and_convert_to_utc(
                str(date_val) if not pd.isna(date_val) else None,
                str(time_val) if not pd.isna(time_val) else None,
                is_utc=False,
                local_tz=Config.LOCAL_TIMEZONE
            )
            timestamps.append(utc_ts)
        
        result["timestamp"] = timestamps
        result["cbt_fahrenheit"] = pd.to_numeric(df.iloc[:, cbt_col], errors="coerce")
        
        # Validate CBT values are in reasonable range (95-105°F)
        result = result.dropna(subset=["timestamp", "cbt_fahrenheit"])
        result = result[
            (result["cbt_fahrenheit"] >= 95) & 
            (result["cbt_fahrenheit"] <= 105)
        ]
        
        return result
    
    def save(self, output_dir: Path, per_user: bool = True, combined: bool = True) -> Dict[str, Path]:
        """Save processed CBT data."""
        saved = {}
        
        if per_user:
            for user_id, df in self.data.items():
                user_dir = output_dir / user_id
                user_dir.mkdir(parents=True, exist_ok=True)
                
                filepath = user_dir / Config.CBT_OUTPUT
                df.to_csv(filepath, index=False)
                saved[f"{user_id}_cbt"] = filepath
        
        if combined and self.data:
            all_users_dir = output_dir / "all_users"
            all_users_dir.mkdir(parents=True, exist_ok=True)
            
            combined_df = []
            for user_id, df in self.data.items():
                df_copy = df.copy()
                df_copy["user_id"] = user_id
                combined_df.append(df_copy)
            
            combined_df = pd.concat(combined_df, ignore_index=True)
            combined_df = combined_df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
            
            filepath = all_users_dir / Config.CBT_OUTPUT
            combined_df.to_csv(filepath, index=False)
            saved["combined_cbt"] = filepath
        
        return saved


# ============================================
# MAIN PIPELINE
# ============================================

def run_pipeline(
    raw_data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    per_user: bool = True,
    combined: bool = True
) -> Dict[str, Path]:
    """
    Run the full preprocessing pipeline.
    
    Args:
        raw_data_dir: Directory containing raw data folders
        output_dir: Directory for preprocessed output
        per_user: Save individual user files
        combined: Save combined files with user_id column
    
    Returns:
        Dict of saved file paths
    """
    raw_data_dir = raw_data_dir or Config.RAW_DATA_DIR
    output_dir = output_dir or Config.OUTPUT_DIR
    
    print("=" * 60)
    print("CBT Prediction Platform - Data Preprocessing Pipeline")
    print("=" * 60)
    print(f"\nRaw data directory: {raw_data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Local timezone: {Config.LOCAL_TIMEZONE}")
    print(f"Per-user output: {per_user}")
    print(f"Combined output: {combined}")
    print()
    
    # Validate raw data directory
    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_data_dir}")
    
    # Find data folders
    print("Step 1: Discovering data folders...")
    hr_folder = find_folder(raw_data_dir, Config.HR_FOLDER_PATTERN)
    wrist_folder = find_folder(raw_data_dir, Config.WRIST_TEMP_FOLDER_PATTERN)
    env_folder = find_folder(raw_data_dir, Config.ENV_FOLDER_PATTERN)
    cbt_folder = find_folder(raw_data_dir, Config.CBT_FOLDER_PATTERN)
    
    print(f"  Heart Rate folder: {hr_folder.name if hr_folder else 'NOT FOUND'}")
    print(f"  Wrist Temp folder: {wrist_folder.name if wrist_folder else 'NOT FOUND'}")
    print(f"  Environmental folder: {env_folder.name if env_folder else 'NOT FOUND'}")
    print(f"  CBT folder: {cbt_folder.name if cbt_folder else 'NOT FOUND'}")
    print()
    
    all_saved = {}
    
    # Process Heart Rate
    print("Step 2: Processing heart rate data...")
    print("  (Already in UTC - standardizing format)")
    hr_processor = HeartRateProcessor().process_folder(hr_folder)
    hr_saved = hr_processor.save(output_dir, per_user=per_user, combined=combined)
    all_saved.update(hr_saved)
    print()
    
    # Process Wrist Temperature
    print("Step 3: Processing wrist temperature data...")
    print(f"  (Converting from {Config.LOCAL_TIMEZONE} to UTC)")
    wrist_processor = WristTemperatureProcessor().process_folder(wrist_folder)
    wrist_saved = wrist_processor.save(output_dir, per_user=per_user, combined=combined)
    all_saved.update(wrist_saved)
    print()
    
    # Process Environmental
    print("Step 4: Processing environmental data...")
    print(f"  (Converting from {Config.LOCAL_TIMEZONE} to UTC)")
    env_processor = EnvironmentalProcessor().process_folder(env_folder)
    env_saved = env_processor.save(output_dir, per_user=per_user, combined=combined)
    all_saved.update(env_saved)
    print()
    
    # Process CBT Labels
    print("Step 5: Processing CBT label data...")
    print(f"  (Converting from {Config.LOCAL_TIMEZONE} to UTC)")
    cbt_processor = CBTProcessor().process_folder(cbt_folder)
    cbt_saved = cbt_processor.save(output_dir, per_user=per_user, combined=combined)
    all_saved.update(cbt_saved)
    print()
    
    # Summary
    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    
    print(f"\nOutput directory: {output_dir}")
    print(f"\nAll timestamps converted to UTC format: YYYY-MM-DDTHH:MM:SSZ")
    
    # List users processed
    all_users = set()
    for processor in [hr_processor, wrist_processor, env_processor, cbt_processor]:
        all_users.update(processor.data.keys())
    
    print(f"\nUsers processed: {len(all_users)}")
    for user_id in sorted(all_users):
        hr_count = len(hr_processor.data.get(user_id, []))
        skin_count = len(wrist_processor.data.get(user_id, []))
        env_count = len(env_processor.data.get(user_id, []))
        cbt_count = len(cbt_processor.data.get(user_id, []))
        print(f"  {user_id}: HR={hr_count:,}, Skin={skin_count:,}, Env={env_count:,}, CBT={cbt_count}")
    
    if combined:
        print(f"\nCombined files saved to: {output_dir / 'all_users'}")
    
    if per_user:
        print(f"\nPer-user files saved to: {output_dir}/<user_id>/")
    
    return all_saved


def validate_output(output_dir: Path) -> bool:
    """
    Validate that output files have correct timestamp format.
    
    Args:
        output_dir: Directory to validate
    
    Returns:
        True if all files have valid UTC timestamps
    """
    all_users_dir = output_dir / "all_users"
    
    expected_files = [
        Config.HR_OUTPUT,
        Config.SKIN_TEMP_OUTPUT,
        Config.ENV_OUTPUT,
        Config.CBT_OUTPUT
    ]
    
    print("\nValidating output files...")
    all_ok = True
    
    # UTC timestamp pattern
    utc_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$')
    
    for filename in expected_files:
        filepath = all_users_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            
            # Find timestamp column
            ts_col = None
            for c in df.columns:
                if 'timestamp' in c.lower():
                    ts_col = c
                    break
            
            if ts_col:
                # Check first few timestamps
                sample = df[ts_col].dropna().head(5)
                valid_format = all(utc_pattern.match(str(ts)) for ts in sample)
                
                if valid_format:
                    print(f"  ✓ {filename}: {len(df):,} rows, UTC format OK")
                    print(f"    Sample: {sample.iloc[0]}")
                else:
                    print(f"  ✗ {filename}: Invalid timestamp format")
                    print(f"    Sample: {sample.iloc[0] if len(sample) > 0 else 'N/A'}")
                    all_ok = False
            else:
                print(f"  ✗ {filename}: No timestamp column found")
                all_ok = False
        else:
            print(f"  ✗ {filename}: NOT FOUND")
            all_ok = False
    
    return all_ok


def test_timestamp_parsing():
    """Test the timestamp parsing with various formats."""
    print("\nTesting timestamp parsing...")
    
    test_cases = [
        # (input_date, input_time, expected_local_datetime)
        ("4/5/2025", "8:59 PM", "2025-04-05 20:59:00"),
        ("04/05/2025", "8:59 PM", "2025-04-05 20:59:00"),
        ("4/5/25", "8:59 PM", "2025-04-05 20:59:00"),
        ("2025-04-05", "20:59:00", "2025-04-05 20:59:00"),
        ("2025-04-05T20:59", None, "2025-04-05 20:59:00"),
        ("2025-04-05 00:00:00", None, "2025-04-05 00:00:00"),
        ("4/5/2025", "12:00 AM", "2025-04-05 00:00:00"),
        ("4/5/2025", "12:00 PM", "2025-04-05 12:00:00"),
    ]
    
    for date_str, time_str, expected_local in test_cases:
        dt = parse_flexible_datetime(date_str, time_str)
        if dt:
            result = dt.strftime("%Y-%m-%d %H:%M:%S")
            status = "✓" if result == expected_local else "✗"
            print(f"  {status} '{date_str}' + '{time_str}' -> {result} (expected: {expected_local})")
        else:
            print(f"  ✗ '{date_str}' + '{time_str}' -> PARSE FAILED")
    
    # Test UTC conversion
    print("\nTesting UTC conversion (America/Chicago)...")
    
    # April 5, 2025 8:59 PM CDT (UTC-5) -> April 6, 2025 1:59 AM UTC
    dt = parse_flexible_datetime("4/5/2025", "8:59 PM")
    utc_dt = convert_local_to_utc(dt, "America/Chicago")
    utc_str = format_utc_timestamp(utc_dt)
    print(f"  April 5, 2025 8:59 PM CDT -> {utc_str}")
    
    # January 5, 2025 8:59 PM CST (UTC-6) -> January 6, 2025 2:59 AM UTC
    dt = parse_flexible_datetime("1/5/2025", "8:59 PM")
    utc_dt = convert_local_to_utc(dt, "America/Chicago")
    utc_str = format_utc_timestamp(utc_dt)
    print(f"  January 5, 2025 8:59 PM CST -> {utc_str}")


# ============================================
# CLI
# ============================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Preprocess raw data for CBT prediction training"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Raw data directory (default: preprocessing/raw_data)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: preprocessing/preprocessed_data)"
    )
    parser.add_argument(
        "--no-per-user",
        action="store_true",
        help="Don't save individual user files"
    )
    parser.add_argument(
        "--no-combined",
        action="store_true",
        help="Don't save combined files"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing output files"
    )
    parser.add_argument(
        "--test-parsing",
        action="store_true",
        help="Test timestamp parsing functions"
    )
    
    args = parser.parse_args()
    
    if args.test_parsing:
        test_timestamp_parsing()
    elif args.validate_only:
        output_dir = args.output_dir or Config.OUTPUT_DIR
        validate_output(output_dir)
    else:
        saved = run_pipeline(
            raw_data_dir=args.raw_dir,
            output_dir=args.output_dir,
            per_user=not args.no_per_user,
            combined=not args.no_combined
        )
        
        # Validate output
        output_dir = args.output_dir or Config.OUTPUT_DIR
        validate_output(output_dir)

