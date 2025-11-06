from src.config import Config
from src.models import MistralModel, DialoGPTModel
from src.models.base import ModelConfig
from src.storage.pgvector_store import PgVectorStore
from src.storage.vector_store_interface import VectorStoreConfig
from src.memory import RollingChatMemory
from src.utils import create_logger
from src.utils.graceful_session_manager import GracefulSessionManager


class App:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.memory = None
        self.storage = None
        self.logger = create_logger("App", config.log_level.value == "DEBUG")

    def setup(self):
        if self.config.log_level.value == "DEBUG":
            print("🤖 Local RAG Chat starting:")
            print(f"   Model: {self.config.model.display_name}")
            print(
                f"   Quantization: {'Enabled' if self.config.use_quantization else 'Disabled'}"
            )
            print()

        self.logger.factory("Creating model...")
        self.model = self._create_model()

        self.logger.storage("Creating Storage...")
        self.storage = self._create_storage()

        self.logger.memory("Creating memory system...")
        self.memory = self._create_memory()
        self.session_manager = GracefulSessionManager(self.memory)

        self.logger.success("Setup Complete!\n")

    def _create_model(self):
        model_config = ModelConfig(
            name=self.config.model.hf_name,
            use_quantization=self.config.use_quantization,
            torch_dtype=self.config.torch_dtype,
            device_map=self.config.device_map,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            do_sample=self.config.do_sample,
        )

        if self.config.model.model_type == "mistral":
            model = MistralModel(model_config)
        elif self.config.model.model_type == "dialogpt":
            model = DialoGPTModel(model_config)
        else:
            raise ValueError(f"Unknown model type: {self.config.model.model_type}")

        model.load()
        return model

    def _create_storage(self):
        storage_config = VectorStoreConfig.for_model(
            embedding_model=self.config.embedding_model
        )
        return PgVectorStore(storage_config)

    def _create_memory(self):
        return RollingChatMemory(
            active_limit=self.config.active_limit,
            vector_store=self.storage,
        )

    def chat_turn(self, user_input: str) -> str:
        self.memory.add_message("user", user_input)

        conversation_history = self.memory.get_ai_context(
            max_messages=self.config.context_messages
        )
        response = self.model.generate_response(
            conversation_history, debug=self.config.log_level.value == "DEBUG"
        )

        self.memory.add_message("chatbot", response)
        return response

    def run(self):
        try:
            self.setup()
            self._chat_loop()
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Error: {e}")
            if self.config.log_level.value == "DEBUG":
                import traceback

                traceback.print_exc()

    def _chat_loop(self):
        self.logger.info("Chat started! Type 'exit' 'q' or <C-c> to end.\n")
        while True:
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

                if self.config.log_level.value == "DEBUG":
                    stats = self.memory.get_memory_stats()
                    memory_display = (
                        f"Active {stats['active_messages']}/{stats['active_limit']}"
                    )
                    self.logger.memory(f"Memory stats: {memory_display}")

            except Exception as e:
                self.logger.error(f"Error during chat turn: {e}")
                if self.config.log_level.value == "DEBUG":
                    import traceback

                    traceback.print_exc()
