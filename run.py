from dotenv import load_dotenv
import logging
from src.pipeline import AnnotationPipeline

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    load_dotenv()
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        print("Starting annotation pipeline")
        pipeline = AnnotationPipeline(output_dir="processed_data")
        pipeline.process_datasets()
        logger.info("Pipeline completed successfully")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()