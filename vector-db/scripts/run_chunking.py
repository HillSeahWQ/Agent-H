"""
Document chunking script with config-only (CAPS) variables, no CLI args.
"""
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from chunking.pdf_chunker import MultimodalPDFChunker
from utils.logger import get_logger

logger = get_logger(__name__)

# ======================================================================
# CONFIG (ALL CAPS)
# ======================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # vector-db
DATA_DIR = PROJECT_ROOT / "data" # create a 'data' folder at AGENT-H/vector-db/data
INPUT_DIR = DATA_DIR / "test" # AGENT-H/vector-db/data/INPUT_DIR
OUTPUT_DIR = DATA_DIR / "chunks"
OUTPUT_FILE = OUTPUT_DIR / "pdf_chunks.jsonl"

CHUNKING_CONFIG = {
    "PDF": {
        "image_coverage_threshold": 0.15,
        "vision_model": "gpt-4o-mini",
        "log_level": "INFO",
    }
}

# ======================================================================
# MAIN
# ======================================================================

def main():
    """Run chunking pipeline using config variables only."""

    input_dir = INPUT_DIR
    output_file = OUTPUT_FILE
    chunking_config = CHUNKING_CONFIG["PDF"].copy()

    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # === LOG CONFIGURATION ===
    logger.info("=" * 80)
    logger.info("DOCUMENT CHUNKING PIPELINE")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  Input directory: {input_dir}")
    logger.info(f"  Output file: {output_file}")
    logger.info(f"  Image threshold: {chunking_config['image_coverage_threshold']}")
    logger.info(f"  Vision model: {chunking_config['vision_model']}")
    logger.info("")

    # === VALIDATION ===
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return 1

    # === RUN CHUNKING ===
    try:
        logger.info("Initializing PDF chunker...")
        chunker = MultimodalPDFChunker(**chunking_config)

        pdf_files = list(input_dir.glob("**/*.pdf"))

        if not pdf_files:
            logger.warning(f"No PDF files found in {input_dir}")
            return 1

        logger.info(f"Found {len(pdf_files)} PDF files")
        logger.info("")

        all_contents = []
        all_metadatas = []

        for pdf_file in pdf_files:
            logger.info(f"Processing: {pdf_file.name}")
            try:
                contents, metadatas = chunker.chunk(pdf_file)
                all_contents.extend(contents)
                all_metadatas.extend(metadatas)
                logger.info(f"Generated {len(contents)} chunks")
            except Exception as e:
                logger.error(f"Failed: {e}")
                continue

        # Save chunks
        logger.info("")
        logger.info(f"Saving {len(all_contents)} chunks to {output_file}")
        MultimodalPDFChunker.save_chunks(all_contents, all_metadatas, output_file)

        logger.info("")
        logger.info("=" * 80)
        logger.info("[SUCCESS] - CHUNKING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total chunks: {len(all_contents)}")
        logger.info(f"Output saved to: {output_file}")
        logger.info("")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"[ERROR] - Chunking failed: {e}")
        logger.exception("Full error traceback:")
        return 1


if __name__ == "__main__":
    load_dotenv()
    sys.exit(main())