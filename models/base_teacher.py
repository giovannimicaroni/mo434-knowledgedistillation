from __future__ import annotations

import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseTeacherModel(nn.Module, ABC):
    def __init__(self, weights_path: str | None = None):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = self._build_model()
        self.classifier = getattr(self.model, 'classifier', None) # Da para separar em métodos de get e set se for melhor
        self.features = getattr(self.model, 'features', None)
        self.avgpool = getattr(self.model, 'avgpool', None)

        if weights_path is not None:
            self._load_weights(weights_path)

        self.to(self.device)

    @abstractmethod
    def _build_model(self) -> nn.Module:
        '''Abstract method to instantiate the model architecture.
        Must be implemented by each subclass.'''
        pass

    def _load_weights(self, weights_path: str) -> None:
        """Standard method for loading checkpoint weights."""
        print(f"Loading teacher weights from {weights_path}...")
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
    

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run only the feature extractor portion of the teacher."""
        x = x.to(self.device)
        features_module = getattr(self.model, 'features', None)

        if features_module is not None:
            return features_module(x)

        return self.model(x)

    def forward_classifier(self, features: torch.Tensor) -> torch.Tensor:
        """Run classifier head from extracted features."""
        x = features
        avgpool_module = getattr(self.model, 'avgpool', None)
        classifier_module = getattr(self.model, 'classifier', None)

        if avgpool_module is not None:
            x = avgpool_module(x)

        x = torch.flatten(x, 1)

        if classifier_module is not None:
            return classifier_module(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass through feature extractor and classifier."""
        x = x.to(self.device)
        features = self.extract_features(x)
        return self.forward_classifier(features)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Inference helper that disables gradients."""
        x = x.to(self.device)
        with torch.no_grad():
            return self.forward(x)

    def freeze_feature_extractor(self) -> None:
        """Freeze feature extractor layers for fine-tuning only the classifier."""
        features_module = getattr(self.model, 'features', None)
        avgpool_module = getattr(self.model, 'avgpool', None)

        if features_module is not None:
            for param in features_module.parameters():
                param.requires_grad = False

        if avgpool_module is not None:
            for param in avgpool_module.parameters():
                param.requires_grad = False

    def freeze_classifier(self) -> None:
        """Freeze classifier head."""
        classifier_module = getattr(self.model, 'classifier', None)

        if classifier_module is not None:
            for param in classifier_module.parameters():
                param.requires_grad = False

    def unfreeze_classifier(self) -> None:
        """Enable gradients on classifier head."""
        classifier_module = getattr(self.model, 'classifier', None)

        if classifier_module is not None:
            for param in classifier_module.parameters():
                param.requires_grad = True

    def get_teacher_parameters(self) -> dict:
        # 1. Total and Trainable parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # 2. Feature Extractor parameters
        feature_params = 0
        if self.features is not None:
            feature_params = sum(p.numel() for p in self.features.parameters())
            
        # 3. Classifier parameters
        classifier_params = 0
        if self.classifier is not None:
            classifier_params = sum(p.numel() for p in self.classifier.parameters())
            
        # 4. AvgPool parameters (rarely has parameters, but good to cover)
        avgpool_params = 0
        if self.avgpool is not None:
            avgpool_params = sum(p.numel() for p in self.avgpool.parameters())

        return {
            "total": total_params,
            "trainable": trainable_params,
            "frozen": total_params - trainable_params,
            "features_module": feature_params,
            "classifier_module": classifier_params,
            "avgpool_module": avgpool_params
        }