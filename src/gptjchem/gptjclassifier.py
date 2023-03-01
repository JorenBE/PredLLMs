from typing import Optional
from gptchem.extractor import ClassificationExtractor
from gptchem.formatter import ClassificationFormatter
from gptchem.gpt_classifier import GPTClassifier
from numpy.typing import ArrayLike
import pandas as pd
from gptjchem.gptj import train, create_dataloaders_from_frames, load_model, tokenizer
import torch
from tqdm import tqdm
from more_itertools import chunked
import gc


class GPTJClassifier(GPTClassifier):
    def __init__(
        self,
        property_name: str,
        querier_settings: Optional[dict] = None,
        extractor: ClassificationExtractor = ClassificationExtractor(),
        batch_size: int = 4,
        tune_settings: Optional[dict] = None,
        inference_batch_size: int = 4,
        inference_max_new_tokens: int = 200,
    ):
        self.property_name = property_name
        self.querier_settings = querier_settings
        self.extractor = extractor
        self.batch_size = batch_size
        self.tune_settings = tune_settings or {}
        self.inference_batch_size = inference_batch_size
        self.inference_max_new_tokens = inference_max_new_tokens

        self.formatter = ClassificationFormatter(
            representation_column="repr",
            label_column="prop",
            property_name=property_name,
            num_classes=None,
        )
        self.model = load_model()

    def _prepare_df(self, X: ArrayLike, y: ArrayLike):
        rows = []
        for i in range(len(X)):
            rows.append({"repr": X[i], "prop": y[i]})
        return pd.DataFrame(rows)

    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        """Fine tune a GPT-3 model on a dataset.

        Args:
            X (ArrayLike): Input data (typically array of molecular representations)
            y (ArrayLike): Target data (typically array of property values)
        """
        df = self._prepare_df(X, y)
        formatted = self.formatter(df)

        dl = create_dataloaders_from_frames(formatted, None, batch_size=self.batch_size)
        train(self.model, dl["train"], **self.tune_settings)
        dl = None
        gc.collect()

    def predict(self, X: ArrayLike, temperature=0.7, do_sample=False) -> ArrayLike:
        """Predict property values for a set of molecular representations.

        Args:
            X (ArrayLike): Input data (typically array of molecular representations)

        Returns:
            ArrayLike: Predicted property values
        """
        df = self._prepare_df(X, [0] * len(X))
        formatted = self.formatter(df)
        completions = []

        self.model.eval()
        device = self.model.device
        with torch.no_grad():
            for chunk in tqdm(
                chunked(range(len(formatted)), self.inference_batch_size),
                total=len(formatted) // self.inference_batch_size,
            ):
                rows = formatted.iloc[chunk]
                prompt = tokenizer(
                    rows["prompt"].to_list(),
                    truncation=False,
                    padding=True,
                    max_length=self.inference_max_new_tokens,
                    return_tensors="pt",
                )
                prompt = {key: value.to(device) for key, value in prompt.items()}
                out = self.model.generate(
                    **prompt,
                    temperature=temperature,
                    max_new_tokens=self.inference_max_new_tokens,
                    do_sample=do_sample,
                )
                completions.extend([tokenizer.decode(out[i]) for i in range(len(out))])
        print(completions)
        extracted = [
            self.extractor.extract(completions[i].split("###")[1]) for i in range(len(completions))
        ]

        return extracted
