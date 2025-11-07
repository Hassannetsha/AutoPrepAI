"""
Main application entry point for AutoPrepAI.
"""

from models import ModelManager
from processors import TextProcessor, IntentProcessor
from extractors import ParameterExtractor
from ui import StreamlitUI

def main():
    # Initialize components
    model_manager = ModelManager()
    text_processor = TextProcessor(model_manager)
    parameter_extractor = ParameterExtractor(model_manager)
    intent_processor = IntentProcessor(model_manager, text_processor, parameter_extractor)
    
    # Initialize and run UI
    ui = StreamlitUI(intent_processor)
    ui.run()

if __name__ == "__main__":
    main()