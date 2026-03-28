"""Main application entry point for AI Voice Detection System"""

import sys
import signal
from .utils.logging_config import setup_logging, get_logger
from .utils.config import get_config
from .services import initialize_services, shutdown_services


def main():
    """Main application entry point"""
    # Initialize logging
    setup_logging()
    logger = get_logger(__name__)
    
    config = get_config()
    logger.info("Starting AI Voice Detection System")
    logger.info(f"Configuration: API={config.api.host}:{config.api.port}, Debug={config.debug}")
    
    try:
        # Initialize all system services
        logger.info("Initializing system services...")
        initialize_services()
        logger.info("System services initialized successfully")
        
        # Set up graceful shutdown handlers
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            shutdown_services()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Import here to ensure services are initialized first
        from .api.app import create_app
        import uvicorn
        
        # Create FastAPI application
        app = create_app()
        
        # Run the application
        logger.info("Starting web server...")
        uvicorn.run(
            app,
            host=config.api.host,
            port=config.api.port,
            log_config=None,  # Use our custom logging
        )
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
        shutdown_services()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to start application: {e}", exc_info=True)
        shutdown_services()
        sys.exit(1)
    finally:
        # Ensure cleanup happens
        shutdown_services()


if __name__ == "__main__":
    main()
