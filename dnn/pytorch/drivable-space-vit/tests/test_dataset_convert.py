import pandas as pd
import logging
import traceback

# Configure logging
logger = logging.getLogger(__name__)

def check_and_remove_duplicates(parquet_file):
    """
    Check for and remove duplicate entries in a Parquet file based on the 'stereo_front_left' column.
    
    Args:
        parquet_file: Path to the Parquet file to check
        
    Returns:
        tuple: (number of duplicates removed, total rows after deduplication)
    """
    try:
        # Read the Parquet file
        df = pd.read_parquet(parquet_file)
        original_rows = len(df)
        
        # Check for duplicates in the 'stereo_front_left' column
        duplicates = df.duplicated(subset=['stereo_front_left'], keep='first')
        num_duplicates = duplicates.sum()
        raise RuntimeError(f"num_duplicates: {num_duplicates}")
        if num_duplicates > 0:
            logger.info(f"Found {num_duplicates} duplicate entries in {parquet_file}")
            
            # Remove duplicates, keeping the first occurrence
            df_deduped = df.drop_duplicates(subset=['stereo_front_left'], keep='first')
            
            # Save the deduplicated data back to the file
            df_deduped.to_parquet(parquet_file, index=False)
            
            logger.info(f"Removed {num_duplicates} duplicate entries. Rows reduced from {original_rows} to {len(df_deduped)}")
            return num_duplicates, len(df_deduped)
        else:
            logger.info(f"No duplicates found in {parquet_file}")
            return 0, original_rows
            
    except Exception as e:
        logger.error(f"Error checking for duplicates in {parquet_file}: {str(e)}")
        logger.debug(traceback.format_exc())
        return 0, 0

check_and_remove_duplicates("../datasets/xiaocars/train_metadata.parquet")