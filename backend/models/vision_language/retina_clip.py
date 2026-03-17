"""
retina_clip.py — Retina Vision-Language Foundation Model
=========================================================
Retina-CLIP: Contrastive Language-Image Pretraining for retinal fundus images.

Inspired by OpenAI CLIP — adapted for clinical ophthalmic domain.

Aligns:
    Retinal Images  ←→  Clinical Text Descriptions

After training, enables:
    1. Zero-shot disease classification
       "Does this retina show diabetic retinopathy?" → YES/NO
    
    2. Semantic image search
       "Find retinas with hard exudates near the macula" → ranked results
    
    3. Report grounding
       Highlight regions matching clinical terms
    
    4. Cross-modal retrieval
       Image → similar clinical notes
       Clinical query → matching fundus images

Training data philosophy:
    (fundus image, clinical report) pairs from:
        - EHR + fundus photography records
        - Publicly annotated datasets (ODIR, APTOS, REFUGE, IDRiD)
        - Synthetic descriptions via GPT-4V labeling

Architecture:
    ┌─────────────────┐    ┌──────────────────────┐
    │  Retina Image   │    │  Clinical Text        │
    │  Encoder (ViT)  │    │  Encoder (BERT/BioGPT)│
    └────────┬────────┘    └──────────┬────────────┘
             │                        │
             ▼                        ▼
    ┌────────────────┐    ┌────────────────────┐
    │  Image Proj.   │    │  Text Projection   │
    └────────┬───────┘    └──────────┬─────────┘
             │                       │
             ▼                       ▼
    ┌──────────────────────────────────────┐
    │   Joint Embedding Space (512-dim)    │
    │   Contrastive similarity learning    │
    └──────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple, Union
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Clinical Vocabulary and Prompt Templates
# ─────────────────────────────────────────────────────────────────────────────

# Ophthalmic clinical prompt templates for zero-shot classification
CLINICAL_PROMPT_TEMPLATES = [
    "A fundus photograph showing {}.",
    "Retinal fundus image with {}.",
    "Ophthalmoscopy image demonstrating {}.",
    "Fundus photography revealing {}.",
    "Clinical retinal image: {}.",
    "A retinal photograph of a patient with {}.",
    "Fundus image showing signs of {}.",
    "Ophthalmic photograph with {}.",
]

# Disease descriptions for zero-shot classification
DISEASE_TEXT_DESCRIPTIONS = {
    "normal": [
        "no retinal pathology",
        "normal fundus appearance",
        "healthy retina",
        "no signs of diabetic retinopathy",
        "clear macula and optic disc",
    ],
    "mild_dr": [
        "mild diabetic retinopathy",
        "microaneurysms only",
        "early signs of diabetic retinopathy",
        "few microaneurysms scattered throughout the fundus",
    ],
    "moderate_dr": [
        "moderate diabetic retinopathy",
        "microaneurysms, dot and blot hemorrhages, and hard exudates",
        "moderate non-proliferative diabetic retinopathy",
        "multiple hemorrhages and exudates",
    ],
    "severe_dr": [
        "severe non-proliferative diabetic retinopathy",
        "more than 20 intraretinal hemorrhages in each quadrant",
        "venous beading and intraretinal microvascular abnormalities",
        "severe NPDR with 4-2-1 rule",
    ],
    "proliferative_dr": [
        "proliferative diabetic retinopathy",
        "neovascularization of the disc and elsewhere",
        "fibrovascular proliferation and vitreous hemorrhage",
        "advanced diabetic retinopathy with neovascularization",
    ],
    "amd_early": [
        "early age-related macular degeneration",
        "drusen in the macula",
        "small hard drusen near the fovea",
    ],
    "amd_intermediate": [
        "intermediate age-related macular degeneration",
        "medium-sized drusen and retinal pigment epithelium changes",
        "drusenoid pigment epithelial detachment",
    ],
    "amd_late": [
        "late age-related macular degeneration",
        "geographic atrophy or neovascular AMD",
        "choroidal neovascularization and subretinal fluid",
        "disciform scar from wet AMD",
    ],
    "glaucoma": [
        "glaucomatous optic neuropathy",
        "increased cup-to-disc ratio with rim thinning",
        "optic disc cupping with peripapillary atrophy",
        "neural rim loss consistent with glaucoma",
        "vertical cup to disc ratio greater than 0.7",
    ],
    "hypertensive_retinopathy": [
        "hypertensive retinopathy",
        "arteriovenous nicking and flame-shaped hemorrhages",
        "cotton-wool spots and retinal arteriolar narrowing",
    ],
}

# Lesion-level descriptions
LESION_TEXT_DESCRIPTIONS = {
    "microaneurysm":        "microaneurysms visible as small red dots",
    "hemorrhage":           "dot and blot hemorrhages",
    "flame_hemorrhage":     "flame-shaped superficial hemorrhages",
    "hard_exudate":         "hard exudates appearing as bright yellow-white deposits",
    "soft_exudate":         "soft exudates or cotton-wool spots",
    "neovascularization":   "neovascularization at the disc or elsewhere",
    "drusen":               "drusen deposits in the macular region",
    "optic_disc_edema":     "optic disc swelling or papilledema",
    "laser_scar":           "laser photocoagulation scars",
    "vessel_tortuosity":    "increased retinal vascular tortuosity",
}


# ─────────────────────────────────────────────────────────────────────────────
# Image Encoder Projection Head
# ─────────────────────────────────────────────────────────────────────────────

class ImageProjectionHead(nn.Module):
    """Projects vision encoder output to joint embedding space."""

    def __init__(self, input_dim: int = 1024, output_dim: int = 512,
                 hidden_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.ln = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(self.projection(x))


# ─────────────────────────────────────────────────────────────────────────────
# Clinical Text Encoder
# ─────────────────────────────────────────────────────────────────────────────

class ClinicalTextEncoder(nn.Module):
    """
    Clinical text encoder using a pretrained biomedical language model.

    Supported backbones:
        - 'pubmedbert': microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract
        - 'biogpt':     microsoft/biogpt
        - 'bert':       bert-base-uncased (fallback)

    Produces 512-dim clinical text embeddings aligned with retinal image space.
    """

    def __init__(self, model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                 output_dim: int = 512, max_length: int = 128,
                 freeze_backbone: bool = True):
        super().__init__()
        self.max_length = max_length
        self.output_dim = output_dim

        # Lazy import to avoid hard dependency at import time
        try:
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.backbone = AutoModel.from_pretrained(model_name)
            backbone_dim = self.backbone.config.hidden_size
        except Exception:
            # Fallback: use a simple transformer
            print(f"[ClinicalTextEncoder] Could not load {model_name}, using lightweight fallback.")
            self.tokenizer = None
            self.backbone = self._build_fallback_encoder()
            backbone_dim = 512

        if freeze_backbone and hasattr(self, 'backbone'):
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim),
            nn.GELU(),
            nn.Linear(backbone_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def _build_fallback_encoder(self):
        """Lightweight transformer fallback when pretrained model unavailable."""
        return nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True),
            num_layers=6,
        )

    def tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Provide token ids directly.")
        return self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_ids:      (B, L) token ids
            attention_mask: (B, L) mask

        Returns:
            (B, output_dim) clinical text embeddings
        """
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Use CLS token or mean pooling
        if hasattr(output, 'last_hidden_state'):
            hidden = output.last_hidden_state  # (B, L, D)
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1)
            else:
                pooled = hidden.mean(1)
        else:
            pooled = output.mean(1)

        return self.projection(pooled)


# ─────────────────────────────────────────────────────────────────────────────
# CLIP Contrastive Loss
# ─────────────────────────────────────────────────────────────────────────────

class CLIPContrastiveLoss(nn.Module):
    """
    Symmetric InfoNCE contrastive loss between image and text embeddings.

    Learnable temperature parameter τ (log scale, initialized to log(1/0.07)).
    """

    def __init__(self, init_temperature: float = 0.07):
        super().__init__()
        self.log_temperature = nn.Parameter(
            torch.tensor(np.log(1.0 / init_temperature))
        )

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)

    def forward(self, image_emb: torch.Tensor,
                text_emb: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            image_emb: (B, D) normalized image embeddings
            text_emb:  (B, D) normalized text embeddings

        Returns:
            loss: scalar
            metrics: dict with accuracy, mean_similarity
        """
        # L2 normalize
        image_emb = F.normalize(image_emb, dim=-1)
        text_emb  = F.normalize(text_emb, dim=-1)

        # Cosine similarity matrix scaled by temperature
        logits = torch.matmul(image_emb, text_emb.T) * self.temperature
        B = logits.shape[0]
        labels = torch.arange(B, device=logits.device)

        # Symmetric cross-entropy
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        loss = (loss_i2t + loss_t2i) / 2.0

        # Metrics
        with torch.no_grad():
            i2t_acc = (logits.argmax(dim=1) == labels).float().mean()
            t2i_acc = (logits.argmax(dim=0) == labels).float().mean()
            mean_diag = logits.diag().mean()

        return loss, {
            "i2t_accuracy": i2t_acc.item(),
            "t2i_accuracy": t2i_acc.item(),
            "mean_similarity": mean_diag.item(),
            "temperature": self.temperature.item(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Full Retina-CLIP Model
# ─────────────────────────────────────────────────────────────────────────────

class RetinaCLIP(nn.Module):
    """
    Retina Vision-Language Foundation Model.

    Aligns retinal images with clinical text in a shared embedding space.

    Capabilities after training:
        • Zero-shot disease classification
        • Semantic retinal image search
        • Grounded report generation
        • Cross-modal retrieval

    Usage:
        model = RetinaCLIP(foundation_encoder)

        # Compute similarities
        similarities = model(images, text_ids, attention_masks)

        # Zero-shot classification
        probs = model.zero_shot_classify(image, disease_names)

        # Encode for retrieval
        img_emb = model.encode_image(image)
        txt_emb = model.encode_text(text_ids, masks)
    """

    def __init__(self, foundation_encoder: nn.Module,
                 encoder_embed_dim: int = 1024,
                 joint_embed_dim: int = 512,
                 text_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                 freeze_text_encoder: bool = True):
        super().__init__()

        # Vision branch
        self.vision_encoder = foundation_encoder
        self.image_projection = ImageProjectionHead(
            input_dim=encoder_embed_dim,
            output_dim=joint_embed_dim,
        )

        # Language branch
        self.text_encoder = ClinicalTextEncoder(
            model_name=text_model_name,
            output_dim=joint_embed_dim,
            freeze_backbone=freeze_text_encoder,
        )

        # Loss
        self.loss_fn = CLIPContrastiveLoss(init_temperature=0.07)

        self.joint_embed_dim = joint_embed_dim

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode retinal images to unit-sphere joint embedding space.
        Returns: (B, joint_embed_dim)
        """
        features = self.vision_encoder(images)       # (B, encoder_embed_dim)
        projected = self.image_projection(features)  # (B, joint_embed_dim)
        return F.normalize(projected, dim=-1)

    def encode_text(self, input_ids: torch.Tensor,
                    attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode clinical text to unit-sphere joint embedding space.
        Returns: (B, joint_embed_dim)
        """
        features = self.text_encoder(input_ids, attention_mask)  # (B, joint_embed_dim)
        return F.normalize(features, dim=-1)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Training forward pass.

        Returns:
            loss: CLIP contrastive loss
            metrics: dict with accuracy and similarity stats
        """
        img_emb = self.encode_image(images)
        txt_emb = self.encode_text(input_ids, attention_mask)
        return self.loss_fn(img_emb, txt_emb)

    @torch.no_grad()
    def zero_shot_classify(
        self,
        image: torch.Tensor,
        class_names: List[str],
        templates: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Zero-shot disease classification using text prompts.

        Args:
            image: (1, 3, H, W) retinal image
            class_names: list of disease names
            templates: list of prompt templates

        Returns:
            dict mapping class_name → probability
        """
        if templates is None:
            templates = CLINICAL_PROMPT_TEMPLATES

        device = next(self.parameters()).device
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(device)

        # Build text embeddings for each class (averaged over templates)
        text_embeddings = []
        for cls_name in class_names:
            descriptions = DISEASE_TEXT_DESCRIPTIONS.get(cls_name, [cls_name])
            all_prompts = [
                tmpl.format(desc)
                for tmpl in templates
                for desc in descriptions
            ]

            if self.text_encoder.tokenizer is not None:
                tokens = self.text_encoder.tokenize(all_prompts)
                tokens = {k: v.to(device) for k, v in tokens.items()}
                embs = self.encode_text(
                    tokens['input_ids'], tokens.get('attention_mask')
                )
            else:
                # Skip if no tokenizer
                embs = torch.zeros(len(all_prompts), self.joint_embed_dim, device=device)

            text_embeddings.append(embs.mean(0))  # Average over templates

        text_emb = torch.stack(text_embeddings)   # (num_classes, D)
        text_emb = F.normalize(text_emb, dim=-1)

        img_emb = self.encode_image(image)         # (1, D)

        logits = (img_emb @ text_emb.T).squeeze(0)  # (num_classes,)
        probs = F.softmax(logits * 100.0, dim=0)    # Scale before softmax

        return {cls: prob.item() for cls, prob in zip(class_names, probs)}

    @torch.no_grad()
    def similarity(self, image: torch.Tensor,
                   texts: List[str]) -> Dict[str, float]:
        """
        Compute similarity scores between a retinal image and clinical descriptions.

        Returns: dict mapping text → cosine similarity
        """
        device = next(self.parameters()).device
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(device)

        img_emb = self.encode_image(image)  # (1, D)

        if self.text_encoder.tokenizer is None:
            return {t: 0.0 for t in texts}

        tokens = self.text_encoder.tokenize(texts)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        txt_embs = self.encode_text(tokens['input_ids'], tokens.get('attention_mask'))

        sims = (img_emb @ txt_embs.T).squeeze(0)
        return {text: sim.item() for text, sim in zip(texts, sims)}

    @torch.no_grad()
    def retrieve_top_k(
        self,
        query_image: torch.Tensor,
        database_embeddings: torch.Tensor,
        k: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve top-k most similar images from an embedding database.

        Args:
            query_image:         (1, 3, H, W)
            database_embeddings: (N, D) pre-computed embeddings
            k: number of results

        Returns:
            indices: (k,) top-k indices
            scores:  (k,) cosine similarities
        """
        device = next(self.parameters()).device
        query_emb = self.encode_image(query_image.to(device))
        sims = query_emb @ database_embeddings.T.to(device)
        top_scores, top_indices = sims.topk(k, dim=-1)
        return top_indices.squeeze(0), top_scores.squeeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# Retina-CLIP Trainer
# ─────────────────────────────────────────────────────────────────────────────

class RetinaCLIPTrainer:
    """
    Training loop for Retina-CLIP vision-language alignment.

    Dataset expected to return:
        (image_tensor, input_ids, attention_mask)

    Each pair is a (fundus image, clinical description) aligned pair.
    """

    def __init__(self, model: RetinaCLIP, dataloader,
                 lr: float = 1e-4, weight_decay: float = 0.01,
                 device: torch.device = None,
                 save_dir: str = "checkpoints/clip"):

        self.model = model
        self.dataloader = dataloader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.save_dir = save_dir

        # Only train projection heads + temperature (keep encoders frozen initially)
        trainable_params = [
            *model.image_projection.parameters(),
            *model.text_encoder.projection.parameters(),
            model.loss_fn.log_temperature,
        ]
        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=lr, weight_decay=weight_decay
        )
        self.scaler = torch.cuda.amp.GradScaler()

    def train_epoch(self, epoch: int) -> Dict:
        self.model.train()
        total_loss = 0.0
        total_i2t_acc = 0.0
        total_t2i_acc = 0.0

        for batch in self.dataloader:
            images, input_ids, attn_masks = batch
            images = images.to(self.device)
            input_ids = input_ids.to(self.device)
            attn_masks = attn_masks.to(self.device)

            with torch.cuda.amp.autocast():
                loss, metrics = self.model(images, input_ids, attn_masks)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            total_i2t_acc += metrics['i2t_accuracy']
            total_t2i_acc += metrics['t2i_accuracy']

        N = len(self.dataloader)
        return {
            "loss": total_loss / N,
            "i2t_accuracy": total_i2t_acc / N,
            "t2i_accuracy": total_t2i_acc / N,
        }

    def save(self, epoch: int, metrics: Dict):
        import os
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            **metrics,
        }, f"{self.save_dir}/retina_clip_epoch_{epoch:04d}.pt")

    def run(self, epochs: int = 50):
        print("🔭 Starting Retina-CLIP Vision-Language Training")
        for epoch in range(epochs):
            metrics = self.train_epoch(epoch)
            print(
                f"Epoch [{epoch+1:3d}/{epochs}]  "
                f"Loss: {metrics['loss']:.4f}  "
                f"I→T Acc: {metrics['i2t_accuracy']:.3f}  "
                f"T→I Acc: {metrics['t2i_accuracy']:.3f}"
            )
            if (epoch + 1) % 10 == 0:
                self.save(epoch, metrics)
        print("✅ Retina-CLIP training complete.")
