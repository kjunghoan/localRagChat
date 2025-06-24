# src/core/temp_app.py
"""
Temporary app class while we refactor.
Contains all the chat logic that was in main.py

TODO:
    Add proper signal handling for Ctrl-C if needed
"""

from src.config import AppConfig
from src.interfaces.model import ModelConfig
from src.factories.model import ModelFactory
from src.storage import ConversationStorage
from src.utils.logger import create_logger


class TempApp:
    """
    Temporary app class to bridge old and new systems.
    TODO:
        Replace with proper ChatOrchestrator + clean separation
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.model = None
        self.memory = None
        self.storage = None
        self.logger = create_logger("TempApp", config.debug.enabled)

    def setup(self):
        """Initialize all components"""
        if self.config.debug.enabled:
            print("🤖 Local RAG Chat starting with NEW FACTORY SYSTEM:")
            print(f"   Model: {self.config.model.name}")
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
        self.storage = ConversationStorage()

        # Create memory
        self.logger.memory("Creating memory system...")
        self.memory = self._create_memory()

        # print("✅ Setup complete!\n")
        self.logger.success("Setup Complete!\n")

    def _create_model(self):
        """Create model using factory"""
        model_config = ModelConfig(
            name=self.config.model.name,
            use_quantization=self.config.model.use_quantization,
            torch_dtype=self.config.model.torch_dtype,
            device_map=self.config.model.device_map,
            max_tokens=self.config.chat.max_tokens,
            temperature=self.config.chat.temperature,
            do_sample=self.config.chat.do_sample,
        )

        # Auto-detect model type
        """ Detect model type based on name."""
        if "mistral" in self.config.model.name.lower():
            model_type = "mistral"
        elif "dialogpt" in self.config.model.name.lower():
            model_type = "dialogpt"
        else:
            self.logger.warning(f"Unknown model type: {self.config.model.name}. Defaulting to Mistral.")
            model_type = "mistral"

        return ModelFactory.create_and_load(model_type, model_config)

    def _create_memory(self):
        """Create memory system"""
        # TODO: Use MemoryFactory when ready
        from src.memory import ConversationalMemory

        return ConversationalMemory(
            vram_limit=self.config.chat.context_messages,
            ram_limit=50,
            storage=self.storage,
        )

    def chat_turn(self, user_input: str) -> str:
        """Handle a single chat turn"""
        # Add user message
        self.memory.add_message("user", user_input)

        # Get context and generate response
        conversation_history = self.memory.get_ai_context()
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
        # Ensure memory is initialized
        if self.memory is None:
            raise RuntimeError("Memory not initialized. Call setup() first.")

        # print("Chat Started! Type 'exit' to end.\n")
        self.logger.info("Chat started! Type 'exit' 'q' or <C-c> to end.\n")

        if self.config.debug.enabled:
            self.logger.model(f"{self.model.__class__.__name__}")
            self.logger.info(f"{self.model.get_model_info()}")

        while True:
            user_input = input("\nUser: ").strip()

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
                        f"VRAM {stats['vram_messages']}/{stats['vram_limit']}"
                    )
                    if stats["ram_messages"] > 0:
                        memory_display += (
                            f", RAM {stats['ram_messages']}/{stats['ram_limit']}"
                        )
                    self.logger.memory(f"Memory stats: {memory_display}")

            except Exception as e:
                self.logger.error(f"Error during chat turn: {e}")
                if self.config.debug.enabled:
                    import traceback

                    traceback.print_exc()

