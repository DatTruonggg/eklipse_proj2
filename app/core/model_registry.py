class ModelRegistry:
    """
    A simple and safe registry for managing models and tokenizers.
    """

    def __init__(self):
        self.models = {}
        self.tokenizers = {}

    def register(self, model_name: str, model, tokenizer):
        """
        Register a model and its corresponding tokenizer.

        Args:
            model_name (str): Name identifier of the model.
            model: Model object (e.g., Huggingface model).
            tokenizer: Tokenizer object (e.g., Huggingface tokenizer).
        """
        if model_name in self.models:
            print(f"[WARN] Model '{model_name}' already registered. Overwriting...")
        self.models[model_name] = model
        self.tokenizers[model_name] = tokenizer
        print(f"[INFO] Registered model: {model_name}")

    def get_model(self, model_name: str):
        """
        Retrieve a model by its name.

        Args:
            model_name (str): Name of the registered model.

        Returns:
            Model object
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' is not registered.")
        return self.models[model_name]

    def get_tokenizer(self, model_name: str):
        """
        Retrieve a tokenizer by model name.

        Args:
            model_name (str): Name of the registered model.

        Returns:
            Tokenizer object
        """
        if model_name not in self.tokenizers:
            raise ValueError(f"Tokenizer for '{model_name}' is not registered.")
        return self.tokenizers[model_name]

    def list_models(self):
        """
        List all registered model names.

        Returns:
            List of model names.
        """
        return list(self.models.keys())

    def clear(self):
        """
        Clear all registered models and tokenizers.
        """
        self.models.clear()
        self.tokenizers.clear()
        print("[INFO] Cleared all registered models and tokenizers.")
