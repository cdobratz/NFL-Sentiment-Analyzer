"""
HuggingFace model integration service for sentiment analysis.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import asyncio
import json

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
    Pipeline,
)
import torch
from huggingface_hub import HfApi, Repository, login, logout
from huggingface_hub.utils import HfHubHTTPError

from ...models.mlops import ModelMetadata, ModelType, ModelStatus
from ...models.sentiment import SentimentLabel, SentimentResult
from ...core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class HuggingFaceModelService:
    """Service for managing HuggingFace models for sentiment analysis"""

    def __init__(self):
        """
        Initialize the HuggingFaceModelService instance by creating API clients, in-memory caches, default model mappings, and optionally authenticating to the HuggingFace Hub.

        Initializes:
        - hf_api: HuggingFace Hub API client.
        - models_cache: in-memory cache mapping model identifiers (name+version) to loaded pipelines.
        - tokenizers_cache: in-memory cache for tokenizers.
        - model_metadata_cache: in-memory cache for registered model metadata.
        - default_models: preset model IDs used for common sentiment and emotion tasks.

        If a HUGGINGFACE_TOKEN is available in settings, attempts to authenticate to the HuggingFace Hub.
        """
        self.hf_api = HfApi()
        self.models_cache: Dict[str, Pipeline] = {}
        self.tokenizers_cache: Dict[str, Any] = {}
        self.model_metadata_cache: Dict[str, ModelMetadata] = {}

        # Default models for NFL sentiment analysis
        self.default_models = {
            "sentiment_base": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "sentiment_sports": "nlptown/bert-base-multilingual-uncased-sentiment",
            "emotion_analysis": "j-hartmann/emotion-english-distilroberta-base",
        }

        # Initialize authentication if token is provided
        if hasattr(settings, "HUGGINGFACE_TOKEN") and settings.HUGGINGFACE_TOKEN:
            try:
                login(token=settings.HUGGINGFACE_TOKEN)
                logger.info("Successfully authenticated with HuggingFace Hub")
            except Exception as e:
                logger.warning(f"Failed to authenticate with HuggingFace Hub: {e}")

    async def load_model(
        self,
        model_name: str,
        model_version: Optional[str] = None,
        force_reload: bool = False,
    ) -> Pipeline:
        """
        Load a HuggingFace model for sentiment analysis

        Args:
            model_name: Name of the model (HF model ID or local path)
            model_version: Specific version/revision to load
            force_reload: Force reload even if cached

        Returns:
            Loaded sentiment analysis pipeline
        """
        cache_key = f"{model_name}:{model_version or 'latest'}"

        if not force_reload and cache_key in self.models_cache:
            return self.models_cache[cache_key]

        try:
            logger.info(f"Loading HuggingFace model: {model_name}")

            # Load model with specific revision if provided
            model_kwargs = {}
            if model_version:
                model_kwargs["revision"] = model_version

            # Create sentiment analysis pipeline (run in thread to avoid blocking)
            sentiment_pipeline = await asyncio.to_thread(
                pipeline,
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                device=0 if torch.cuda.is_available() else -1,
                **model_kwargs,
            )

            # Cache the loaded model
            self.models_cache[cache_key] = sentiment_pipeline

            # Load and cache tokenizer separately for advanced usage (run in thread)
            tokenizer = await asyncio.to_thread(
                AutoTokenizer.from_pretrained, model_name, **model_kwargs
            )
            self.tokenizers_cache[cache_key] = tokenizer

            logger.info(f"Successfully loaded model: {model_name}")
            return sentiment_pipeline

        except Exception as e:
            logger.error(f"Failed to load HuggingFace model {model_name}: {e}")
            raise

    async def predict_sentiment(
        self,
        text: str,
        model_name: str = "sentiment_base",
        model_version: Optional[str] = None,
        return_all_scores: bool = False,
    ) -> Dict[str, Any]:
        """
        Predict sentiment for a single text using a resolved HuggingFace model.

        Resolves model_name against the service's default model mapping, loads the model (optionally at a specific version), runs the pipeline, and returns a normalized prediction.

        Parameters:
            text (str): Text to analyze.
            model_name (str): Key from the service's default_models mapping or a HuggingFace model ID.
            model_version (Optional[str]): Specific model revision or version to load; when omitted the latest is used.
            return_all_scores (bool): If true, include scores for all labels in the pipeline output.

        Returns:
            Dict[str, Any]: Normalized sentiment prediction. For a single prediction this includes keys such as `label`, `sentiment_score`, `confidence`, and `raw_prediction`. If multiple label scores are returned, the dictionary may contain a `predictions` list with per-label entries.
        """
        try:
            # Resolve model name
            resolved_model_name = self.default_models.get(model_name, model_name)

            # Load model
            pipeline_model = await self.load_model(resolved_model_name, model_version)

            # Make prediction
            result = pipeline_model(text, return_all_scores=return_all_scores)

            # Normalize result format
            if isinstance(result, list) and len(result) > 0:
                result = result[0]

            # Convert to standardized format
            prediction = self._normalize_prediction(result, return_all_scores)

            return prediction

        except Exception as e:
            logger.error(f"Error predicting sentiment with HF model: {e}")
            raise

    async def batch_predict_sentiment(
        self,
        texts: List[str],
        model_name: str = "sentiment_base",
        model_version: Optional[str] = None,
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        """
        Predict sentiment for multiple input texts in batches and return normalized results.

        Parameters:
            texts (List[str]): Input texts to analyze; order of results matches input order.
            model_name (str): Named key from defaults or explicit model ID to resolve which model to load.
            model_version (Optional[str]): Optional model revision or version to load.
            batch_size (int): Number of texts to process per model inference call.

        Returns:
            List[Dict[str, Any]]: List of normalized prediction dictionaries corresponding to each input text.
        """
        try:
            resolved_model_name = self.default_models.get(model_name, model_name)
            pipeline_model = await self.load_model(resolved_model_name, model_version)

            # Process in batches
            results = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                batch_results = pipeline_model(batch)

                # Normalize each result
                for result in batch_results:
                    normalized = self._normalize_prediction(
                        result, return_all_scores=False
                    )
                    results.append(normalized)

            return results

        except Exception as e:
            logger.error(f"Error in batch sentiment prediction: {e}")
            raise

    async def register_model(
        self,
        model_name: str,
        model_path: str,
        model_type: ModelType = ModelType.SENTIMENT_ANALYSIS,
        metrics: Optional[Dict[str, float]] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> ModelMetadata:
        """
        Register a local model and, if authenticated, upload it to the HuggingFace Hub, returning the created metadata.

        Parameters:
            model_name (str): Desired name for the registered model (used to form the model ID).
            model_path (str): Filesystem path to the model artifacts to register/upload.
            model_type (ModelType): Domain-specific classification of the model; defaults to sentiment analysis.
            metrics (Optional[Dict[str, float]]): Optional custom performance metrics to attach to the metadata.
            description (Optional[str]): Optional human-readable description of the model.
            tags (Optional[List[str]]): Optional list of tags to associate with the model.

        Returns:
            ModelMetadata: Metadata for the registered model, including model_id, version, status, timestamps, and, when uploaded, deployment details.
        """
        try:
            # Create model metadata
            metadata = ModelMetadata(
                model_id=f"nfl-analyzer/{model_name}",
                model_name=model_name,
                version="1.0",
                model_type=model_type,
                status=ModelStatus.TRAINING,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                created_by="system",
                framework="transformers",
                model_path=model_path,
                custom_metrics=metrics or {},
                tags=tags or [],
                description=description,
            )

            # Upload to HuggingFace Hub if authenticated
            if hasattr(settings, "HUGGINGFACE_TOKEN") and settings.HUGGINGFACE_TOKEN:
                try:
                    # Create repository
                    repo_id = metadata.model_id
                    self.hf_api.create_repo(
                        repo_id=repo_id, exist_ok=True, private=True
                    )

                    # Upload model files
                    self.hf_api.upload_folder(
                        folder_path=model_path, repo_id=repo_id, repo_type="model"
                    )

                    # Update metadata with HF info
                    metadata.status = ModelStatus.DEPLOYED
                    metadata.deployment_config = {
                        "huggingface_repo": repo_id,
                        "uploaded_at": datetime.utcnow().isoformat(),
                    }

                    logger.info(
                        f"Successfully uploaded model to HuggingFace Hub: {repo_id}"
                    )

                except HfHubHTTPError as e:
                    logger.warning(f"Failed to upload to HuggingFace Hub: {e}")
                    metadata.status = ModelStatus.FAILED

            # Cache metadata
            self.model_metadata_cache[metadata.model_id] = metadata

            return metadata

        except Exception as e:
            logger.error(f"Error registering model: {e}")
            raise

    async def list_available_models(
        self, task: str = "sentiment-analysis", sort: str = "downloads", limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Retrieve a list of models from the HuggingFace Hub filtered and sorted by the given criteria.

        Parameters:
            task (str): Task filter to apply (e.g., "sentiment-analysis").
            sort (str): Sort key used by the Hub (e.g., "downloads").
            limit (int): Maximum number of models to return.

        Returns:
            List[dict]: A list of model information dictionaries. Each dictionary contains:
                - model_id: Model identifier on the Hub.
                - downloads: Number of downloads (or 0 if unavailable).
                - likes: Number of likes (or 0 if unavailable).
                - tags: List of model tags (or empty list if unavailable).
                - pipeline_tag: Primary pipeline tag (or None if unavailable).
                - created_at: Creation timestamp (or None if unavailable).
                - last_modified: Last modified timestamp (or None if unavailable).
        """
        try:
            models = self.hf_api.list_models(task=task, sort=sort, limit=limit)

            model_list = []
            for model in models:
                model_info = {
                    "model_id": model.modelId,
                    "downloads": getattr(model, "downloads", 0),
                    "likes": getattr(model, "likes", 0),
                    "tags": getattr(model, "tags", []),
                    "pipeline_tag": getattr(model, "pipeline_tag", None),
                    "created_at": getattr(model, "createdAt", None),
                    "last_modified": getattr(model, "lastModified", None),
                }
                model_list.append(model_info)

            return model_list

        except Exception as e:
            logger.error(f"Error listing HuggingFace models: {e}")
            return []

    async def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Retrieve detailed metadata for a HuggingFace model.

        Returns:
            A dictionary with model metadata:
            - `model_id` (str): Model identifier on HuggingFace.
            - `sha` (str): Commit SHA or revision identifier.
            - `downloads` (int): Number of downloads (0 if unavailable).
            - `likes` (int): Number of likes (0 if unavailable).
            - `tags` (List[str]): Model tags (empty list if unavailable).
            - `pipeline_tag` (Optional[str]): Primary pipeline tag (e.g., "text-classification").
            - `library_name` (Optional[str]): Primary library name (e.g., "transformers").
            - `created_at` (Optional[str]): Creation timestamp if available.
            - `last_modified` (Optional[str]): Last modified timestamp if available.
            - `card_data` (dict): Model card metadata (empty dict if unavailable).
            - `siblings` (List[str]): Filenames of sibling files in the model repository.
        """
        try:
            model_info = self.hf_api.model_info(model_id)

            return {
                "model_id": model_info.modelId,
                "sha": model_info.sha,
                "downloads": getattr(model_info, "downloads", 0),
                "likes": getattr(model_info, "likes", 0),
                "tags": getattr(model_info, "tags", []),
                "pipeline_tag": getattr(model_info, "pipeline_tag", None),
                "library_name": getattr(model_info, "library_name", None),
                "created_at": getattr(model_info, "createdAt", None),
                "last_modified": getattr(model_info, "lastModified", None),
                "card_data": getattr(model_info, "cardData", {}),
                "siblings": [s.rfilename for s in getattr(model_info, "siblings", [])],
            }

        except Exception as e:
            logger.error(f"Error getting model info for {model_id}: {e}")
            raise

    async def fine_tune_model(
        self,
        base_model: str,
        training_data: List[Dict[str, Any]],
        validation_data: Optional[List[Dict[str, Any]]] = None,
        output_dir: str = "./fine_tuned_model",
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        batch_size: int = 16,
    ) -> ModelMetadata:
        """
        Fine-tune a HuggingFace Transformers model for NFL-specific sentiment analysis.

        This creates and saves a fine-tuned model artifact in output_dir and records training metadata in the service cache. `training_data` and `validation_data` should be lists of labeled examples (e.g., dicts containing text and a sentiment label) suitable for building a sequence-classification dataset. The returned metadata reflects the created fine-tuned model and is updated to indicate validation is pending on successful save.

        Parameters:
            base_model: Identifier of the pretrained HuggingFace model to fine-tune (e.g., a model ID or local path).
            training_data: List of training examples (labeled text records) used for fine-tuning.
            validation_data: Optional list of validation examples used for evaluation during or after training.
            output_dir: Filesystem path where the fine-tuned model and tokenizer are saved.
            num_epochs: Number of training epochs to run.
            learning_rate: Learning rate used for fine-tuning.
            batch_size: Batch size used during training.

        Returns:
            ModelMetadata: Metadata for the created fine-tuned model; on successful completion the metadata.status is set to VALIDATING and the metadata is cached by the service.
        """
        try:
            # This is a simplified implementation
            # In practice, you'd use transformers.Trainer or similar
            logger.info(f"Starting fine-tuning of {base_model}")

            # Load base model and tokenizer (run in thread to avoid blocking)
            tokenizer = await asyncio.to_thread(
                AutoTokenizer.from_pretrained, base_model
            )
            model = await asyncio.to_thread(
                AutoModelForSequenceClassification.from_pretrained,
                base_model,
                num_labels=3,  # positive, negative, neutral
            )

            # Prepare training configuration
            training_config = {
                "base_model": base_model,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "training_samples": len(training_data),
                "validation_samples": len(validation_data) if validation_data else 0,
            }

            # Create metadata for the fine-tuned model
            model_name = f"nfl_sentiment_{base_model.split('/')[-1]}"
            metadata = ModelMetadata(
                model_id=f"nfl-analyzer/{model_name}",
                model_name=model_name,
                version="1.0-finetuned",
                model_type=ModelType.SENTIMENT_ANALYSIS,
                status=ModelStatus.TRAINING,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                created_by="system",
                framework="transformers",
                model_path=output_dir,
                training_dataset=f"nfl_sentiment_training_{len(training_data)}",
                validation_dataset=f"nfl_sentiment_validation_{len(validation_data) if validation_data else 0}",
                epochs=num_epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                description=f"Fine-tuned {base_model} for NFL sentiment analysis",
            )

            # TODO: Implement actual fine-tuning logic here
            # This would involve:
            # 1. Data preprocessing and tokenization
            # 2. Dataset creation
            # 3. Training loop with transformers.Trainer
            # 4. Model evaluation
            # 5. Saving the fine-tuned model

            # For now, simulate training completion
            await asyncio.sleep(1)  # Simulate training time

            # Save model and tokenizer
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            # Update metadata
            metadata.status = ModelStatus.VALIDATING
            metadata.updated_at = datetime.utcnow()

            # Cache metadata
            self.model_metadata_cache[metadata.model_id] = metadata

            logger.info(f"Fine-tuning completed for {model_name}")
            return metadata

        except Exception as e:
            logger.error(f"Error fine-tuning model: {e}")
            raise

    def _normalize_prediction(
        self, result: Dict[str, Any], return_all_scores: bool = False
    ) -> Dict[str, Any]:
        """
        Convert a HuggingFace pipeline output into the service's standardized prediction structure.

        Parameters:
            result (dict|list[dict]): Raw prediction(s) returned by a HuggingFace pipeline (single prediction dict or a list of prediction dicts).
            return_all_scores (bool): Whether normalized predictions should include scores for all labels.

        Returns:
            dict: If a single prediction was provided, returns a normalized prediction dict with keys such as `label`, `sentiment_score`, `confidence`, and `raw_prediction`. If a list was provided, returns `{"predictions": [<normalized prediction dicts>]}`.
        """
        # Handle different result formats
        if isinstance(result, list):
            # Multiple predictions returned
            normalized = {"predictions": []}
            for pred in result:
                normalized["predictions"].append(
                    self._normalize_single_prediction(pred)
                )
            return normalized
        else:
            # Single prediction
            return self._normalize_single_prediction(result)

    def _normalize_single_prediction(self, pred: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a single HuggingFace model prediction into the service's standardized sentiment format.

        This maps common HF label names (including `LABEL_0`/`LABEL_1`/`LABEL_2`) to the internal SentimentLabel enum, converts the model `score` into a signed `sentiment_score` (positive for positive sentiment, negative for negative sentiment, zero for neutral), and preserves the original prediction.

        Parameters:
            pred (Dict[str, Any]): Prediction object from a HuggingFace pipeline, expected to contain at least `label` and `score`.

        Returns:
            Dict[str, Any]: A dictionary with keys:
                - `label` (str): Standardized sentiment label value.
                - `sentiment_score` (float): Signed sentiment score (positive for positive sentiment, negative for negative, 0.0 for neutral).
                - `confidence` (float): Original model score.
                - `raw_prediction` (Dict[str, Any]): The original prediction object.
        """
        # Map HuggingFace labels to our standard labels
        label_mapping = {
            "POSITIVE": SentimentLabel.POSITIVE,
            "NEGATIVE": SentimentLabel.NEGATIVE,
            "NEUTRAL": SentimentLabel.NEUTRAL,
            "LABEL_0": SentimentLabel.NEGATIVE,  # Common in some models
            "LABEL_1": SentimentLabel.NEUTRAL,
            "LABEL_2": SentimentLabel.POSITIVE,
        }

        label = pred.get("label", "NEUTRAL")
        score = pred.get("score", 0.0)

        # Map to standard label
        standard_label = label_mapping.get(label, SentimentLabel.NEUTRAL)

        # Convert score to sentiment score (-1 to 1)
        if standard_label == SentimentLabel.POSITIVE:
            sentiment_score = score
        elif standard_label == SentimentLabel.NEGATIVE:
            sentiment_score = -score
        else:
            sentiment_score = 0.0

        return {
            "label": standard_label.value,
            "sentiment_score": sentiment_score,
            "confidence": score,
            "raw_prediction": pred,
        }

    async def cleanup_cache(self):
        """
        Clear in-memory model and tokenizer caches to free resources.

        This removes all entries from the service's model and tokenizer caches.
        """
        self.models_cache.clear()
        self.tokenizers_cache.clear()
        logger.info("Cleared HuggingFace model cache")

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Return current in-memory cache state for loaded model pipelines and tokenizers.

        Returns:
            dict: Mapping with keys:
                - cached_models: list of cache keys for model pipelines.
                - cached_tokenizers: list of cache keys for tokenizers.
                - cache_size: number of cached model pipelines.
        """
        return {
            "cached_models": list(self.models_cache.keys()),
            "cached_tokenizers": list(self.tokenizers_cache.keys()),
            "cache_size": len(self.models_cache),
        }
