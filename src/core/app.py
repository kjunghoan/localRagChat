"""
Primary application entry point.
Contains all the chat logic

TODO:
    Add proper signal handling for Ctrl-C if needed
"""

from src.configs import AppConfig
from src.factories import ModelFactory, StorageFactory
from src.interfaces import ModelConfig
from src.storage import VectorStoreConfig
from src.memory import RollingChatMemory
from src.utils import create_logger
from src.utils.graceful_session_manager import GracefulSessionManager


class App:
    """
    TODO:
        Replace with proper ChatOrchestrator + clean separation
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.model = None
        self.memory = None
        self.storage = None
        self.logger = create_logger("App", config.debug.enabled)

    def setup(self):
        """Initialize all components"""
        if self.config.debug.enabled:
            print("🤖 Local RAG Chat starting:")
            print(f"   Model: {self.config.model.model.display_name}")
            print(
                f"   Quantization: {'Enabled' if self.config.model.use_quantization else 'Disabled'}"
            )
            print()

        # Setup directories
        self.config.ensure_directories()

        # Create model
        self.logger.factory("Creating model with factory...")
        self.model = self._create_model()

        # Create storage
        self.logger.storage("Creating Storage...")
        self.storage = self._create_storage()

        # Create memory
        self.logger.memory("Creating memory system...")
        self.memory = self._create_memory()
        self.session_manager = GracefulSessionManager(self.memory)

        self.logger.success("Setup Complete!\n")

    def _create_model(self):
        """Create model using factory"""
        model_config = ModelConfig(
            name=self.config.model.model.hf_name,
            use_quantization=self.config.model.use_quantization,
            torch_dtype=self.config.model.torch_dtype,
            device_map=self.config.model.device_map,
            max_tokens=self.config.chat.max_tokens,
            temperature=self.config.chat.temperature,
            do_sample=self.config.chat.do_sample,
        )

        model_type = self.config.model.model.model_type

        return ModelFactory.create_and_load(model_type, model_config)

    def _create_storage(self):
        """Create storage using factory"""
        storage_config = VectorStoreConfig.for_model(
            embedding_model=self.config.embedding_model,
            db_path=str(self.config.data_dir / "vector_store"),
        )
        return StorageFactory.create(self.config.storage_type, storage_config)

    def _create_memory(self):
        """Create memory system"""
        # TODO: Use MemoryFactory later on if needed
        return RollingChatMemory(
            active_limit=self.config.chat.active_limit,
            vector_store=self.storage,
        )

    def chat_turn(self, user_input: str) -> str:
        """Handle a single chat turn"""
        # Add user message to memory
        self.memory.add_message("user", user_input)

        # Get context and generate response
        max_messages = self.config.chat.context_messages
        conversation_history = self.memory.get_ai_context(max_messages=max_messages)
        response = self.model.generate_response(
            conversation_history, debug=self.config.debug.enabled
        )

        # Add response to memory
        self.memory.add_message("chatbot", response)

        return response

    def run(self):
        """Run the chat application"""
        try:
            self.setup()
            self._chat_loop()
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Error: {e}")
            if self.config.debug.enabled:
                import traceback

                traceback.print_exc()

    def _chat_loop(self):
        """Main chat loop"""
        self.logger.info("Chat started! Type 'exit' 'q' or <C-c> to end.\n")
        while True:
            user_input = input("\nUser: ").strip()
            try:
                user_input = input("\nUser: ").strip()
            except EOFError:
                self.logger.info("\n Recieved EOF, ending chat session.")
                break

            if user_input.lower() in ["exit", "quit", "q"]:
                self.logger.storage("Saving session...")
                self.memory.session_cleanup()
                print("Goodbye!")
                break

            if not user_input:
                continue

            try:
                response = self.chat_turn(user_input)
                print(f"\nChatbot: {response}")

                if self.config.debug.enabled:
                    stats = self.memory.get_memory_stats()
                    memory_display = (
                        f"Active {stats['active_messages']}/{stats['active_limit']}"
                    )
                    self.logger.memory(f"Memory stats: {memory_display}")

            except Exception as e:
                self.logger.error(f"Error during chat turn: {e}")
                if self.config.debug.enabled:
                    import traceback

                    traceback.print_exc()
