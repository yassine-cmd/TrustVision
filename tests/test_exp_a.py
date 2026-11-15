"""
Unit and Integration Tests for Member A (CLIP & Attention) and Shared Contracts.
"""

import ast
import json
import math
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


class TestNotebookAndImports(unittest.TestCase):
    """Verifies that all notebook cells and modules are syntactically sound and have all imports."""

    def test_notebook_syntax_and_transforms_import(self):
        nb_path = Path("notebooks/exp_A.ipynb")
        self.assertTrue(nb_path.exists(), "notebooks/exp_A.ipynb must exist")

        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        self.assertGreaterEqual(len(code_cells), 5, "Expected at least 5 code cells")

        # Verify Cell 1 imports transforms, requests, shutil, zipfile explicitly
        cell_1_code = "".join(code_cells[0]["source"])
        self.assertIn("from torchvision import transforms", cell_1_code)
        self.assertIn("import requests", cell_1_code)
        self.assertIn("import shutil", cell_1_code)
        self.assertIn("import zipfile", cell_1_code)
        self.assertIn('install_if_missing("kagglehub")', cell_1_code)
        self.assertIn('install_if_missing("requests")', cell_1_code)

        # Verify Cell 3 also has fallback import guards
        cell_3_code = "".join(code_cells[1]["source"])
        self.assertIn("requests", cell_3_code)
        self.assertIn("zipfile", cell_3_code)
        self.assertIn("shutil", cell_3_code)

        # Verify zero mock data in entire notebook
        full_nb_text = json.dumps(nb)
        self.assertNotIn("mock_", full_nb_text, "Found 'mock_' in exp_A.ipynb! Zero mock policy violated.")
        self.assertNotIn("torch.rand(", full_nb_text, "Found 'torch.rand(' in exp_A.ipynb! Zero mock policy violated.")

        # Verify all code cells parse with ast (no syntax errors)
        for idx, cell in enumerate(code_cells):
            code_str = "".join(cell["source"])
            clean_lines = [
                line for line in code_str.splitlines()
                if not line.strip().startswith("!") and not line.strip().startswith("%")
            ]
            clean_code = "\n".join(clean_lines)
            try:
                ast.parse(clean_code)
            except SyntaxError as e:
                self.fail(f"SyntaxError in Notebook Code Cell {idx}: {e}")

    def test_src_modules_importable(self):
        from src.data import NEUCLSDataset, get_dataloaders, NEU_CLASSES
        from src.model import TrustVisionCNN
        from src.eval import compute_ece, evaluate_all
        from src.clip_attn import ManualScaledDotProductAttention, DEFAULT_CLASSES, extract_tensor_features

        self.assertEqual(len(NEU_CLASSES), 6)
        self.assertEqual(len(DEFAULT_CLASSES), 6)


class TestTensorFeatureExtraction(unittest.TestCase):
    """Guarantees that BaseModelOutputWithPooling and other transformer output objects never fail .norm()."""

    def test_extract_tensor_features_from_pooling_output(self):
        from src.clip_attn import extract_tensor_features

        # Simulate BaseModelOutputWithPooling
        class MockBaseModelOutputWithPooling:
            def __init__(self, tensor):
                self.pooler_output = tensor

        raw_tensor = torch.randn(6, 512)
        mock_output = MockBaseModelOutputWithPooling(raw_tensor)

        extracted = extract_tensor_features(mock_output)
        self.assertIsInstance(extracted, torch.Tensor)
        self.assertEqual(extracted.shape, (6, 512))

        # This was the exact failing line reported by user:
        normed = extracted / extracted.norm(dim=-1, keepdim=True)
        self.assertEqual(normed.shape, (6, 512))

    def test_extract_tensor_features_from_bare_tensor(self):
        from src.clip_attn import extract_tensor_features

        raw_tensor = torch.randn(4, 512)
        extracted = extract_tensor_features(raw_tensor)
        self.assertIsInstance(extracted, torch.Tensor)
        normed = extracted / extracted.norm(dim=-1, keepdim=True)
        self.assertEqual(normed.shape, (4, 512))


class TestPrefixMatcher(unittest.TestCase):
    """Guarantees RS_1.bmp, PS_1.bmp, Cr_1.bmp, etc. are matched correctly."""

    def test_prefix_matching(self):
        from src.data import match_neu_class, NEU_CLASSES

        cases = [
            ("RS_1.bmp", "rolled-in_scale"),
            ("RS_300.bmp", "rolled-in_scale"),
            ("PS_42.bmp", "pitted_surface"),
            ("Cr_10.bmp", "crazing"),
            ("In_123.bmp", "inclusion"),
            ("Pa_99.bmp", "patches"),
            ("Sc_200.bmp", "scratches"),
            ("crazing_1.jpg", "crazing"),
            ("rolled-in_scale_5.jpg", "rolled-in_scale"),
        ]

        for filename, expected_class in cases:
            p = Path(filename)
            c_idx = match_neu_class(p)
            self.assertIsNotNone(c_idx, f"Failed to match {filename}")
            self.assertEqual(NEU_CLASSES[c_idx], expected_class, f"Mismatch for {filename}")


class TestAttentionMath(unittest.TestCase):
    """Verifies the pure PyTorch Scaled Dot-Product Attention mathematical properties."""

    def test_attention_shapes_and_row_stochasticity(self):
        from src.clip_attn import ManualScaledDotProductAttention

        embed_dim = 768
        d_k = 64
        seq_len = 50
        batch_size = 2

        attn = ManualScaledDotProductAttention(embed_dim=embed_dim, d_k=d_k)
        x = torch.randn(batch_size, seq_len, embed_dim)
        context, weights = attn(x, return_attention_weights=True)

        self.assertEqual(context.shape, (batch_size, seq_len, d_k))
        self.assertEqual(weights.shape, (batch_size, seq_len, seq_len))

        row_sums = weights.sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5))

        expected_scale = 1.0 / math.sqrt(d_k)
        self.assertAlmostEqual(attn.scale, expected_scale, places=6)


class TestRealDatasetIntegrity(unittest.TestCase):
    """Verifies real dataset integrity (1,800 images, exactly 300 per class, zero mock)."""

    @unittest.skipUnless(Path("data/NEU-CLS").exists(), "NEU-CLS dataset not present")
    def test_dataset_and_split_integrity(self):
        from src.data import get_dataloaders, NEUCLSDataset, NEU_CLASSES

        # Direct dataset count
        ds = NEUCLSDataset(root_dir="data/NEU-CLS", img_size=224)
        self.assertEqual(len(ds), 1800)
        for c_idx in range(6):
            count = sum(1 for s in ds.samples if s[1] == c_idx)
            self.assertEqual(count, 300, f"Class {NEU_CLASSES[c_idx]} must have 300 real images")

        train_l, cal_l, test_l, test_files = get_dataloaders(
            root_dir="data/NEU-CLS",
            batch_size=32,
            seed=42,
            img_size=224,
        )

        self.assertEqual(len(test_files), 360, "Test set must have exactly 360 images")
        images, labels = next(iter(train_l))
        self.assertEqual(images.shape, (32, 3, 224, 224))
        self.assertGreaterEqual(images.min().item(), 0.0)
        self.assertLessEqual(images.max().item(), 1.0)


class TestPatchAttentionExtraction(unittest.TestCase):
    """Verifies that ViT patch attention extraction works without unexpected argument errors."""

    def test_extract_cls_patch_attention_shape_and_eager(self):
        from src.clip_attn import load_clip, extract_cls_patch_attention

        clip_model, _ = load_clip()
        dummy_images = torch.rand(2, 3, 224, 224)
        
        # Single layer
        maps = extract_cls_patch_attention(clip_model, dummy_images, layer_idx=-1, head_idx=None)
        self.assertEqual(maps.shape, (2, 7, 7))
        self.assertGreaterEqual(maps.min(), 0.0)

        # Multi-layer (all 12 layers)
        maps_all = extract_cls_patch_attention(clip_model, dummy_images, layer_idx=None, head_idx=None)
        self.assertEqual(maps_all.shape, (2, 12, 7, 7))
        self.assertGreaterEqual(maps_all.min(), 0.0)


class TestModelWithTemperatureDevice(unittest.TestCase):
    """Verifies that ModelWithTemperature sets temperature on the specified device without mismatch."""

    def test_temperature_scaling_on_device(self):
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
        from src.eval import ModelWithTemperature

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = nn.Linear(10, 6)
        x = torch.randn(20, 10)
        y = torch.randint(0, 6, (20,))
        loader = DataLoader(TensorDataset(x, y), batch_size=5)

        m = ModelWithTemperature(model)
        temp_val = m.set_temperature(loader, device=device)
        self.assertGreater(temp_val, 0.0)


class TestFGSMZeroGradHygiene(unittest.TestCase):
    """Verifies that FGSM attack leaves no residual gradients on target model parameters."""

    def test_fgsm_attack_clears_grad(self):
        import torch.nn as nn
        from src.eval import fgsm_attack

        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 96 * 96, 6))
        images = torch.rand(2, 3, 96, 96)
        labels = torch.tensor([0, 1])

        _ = fgsm_attack(model, images, labels, eps=0.03)
        for param in model.parameters():
            if param.grad is not None:
                self.assertTrue(torch.all(param.grad == 0))


class TestXAISimilarityAndSplitsLock(unittest.TestCase):
    """Verifies XAI similarity helper functions and strict splits SHA-256 agreement."""

    def test_compute_xai_similarity_scale_invariance(self):
        from src.eval import compute_xai_similarity
        
        # Simulated Grad-CAM in [0, 1]
        grad_cam = torch.rand(7, 7)
        # Scaled attention map (like clip_attn_test.npy, sum ~ 0.35)
        attn_map = grad_cam * 0.05 + 0.001

        sim_cos = compute_xai_similarity(grad_cam, attn_map, method="cosine")
        self.assertGreater(sim_cos, 0.95, "Cosine similarity must be scale invariant")

        sim_l1 = compute_xai_similarity(grad_cam, attn_map, method="normalized_l1")
        self.assertGreater(sim_l1, 0.95, "Normalized L1 similarity must be scale invariant")

    def test_splits_sha256_matches_s0_clip_json(self):
        import hashlib
        
        with open("configs/splits.json", "rb") as f:
            computed_hash = hashlib.sha256(f.read()).hexdigest()
        
        with open("results/s0_clip.json", "r", encoding="utf-8") as f:
            s0 = json.load(f)
        
        expected_hash = s0["splits_sha256"]
        self.assertEqual(computed_hash, expected_hash, f"Splits hash mismatch: {computed_hash} vs {expected_hash}")


    def test_compute_xai_similarity_resizes_to_vit_grid(self):
        from src.eval import compute_xai_similarity

        base = torch.rand(7, 7)
        high_res = F.interpolate(
            base.view(1, 1, 7, 7), size=(96, 96), mode="bilinear", align_corners=False
        ).view(96, 96)

        sim_cos = compute_xai_similarity(high_res, base, method="cosine")
        self.assertGreater(sim_cos, 0.99, "Resized Grad-CAM must align with the 7x7 ViT grid")

        sim_l1 = compute_xai_similarity(high_res, base, method="normalized_l1")
        self.assertGreater(sim_l1, 0.95, "Resized normalized L1 similarity must stay high")

    def test_s0_clip_json_schema(self):
        with open("results/s0_clip.json", "r", encoding="utf-8") as f:
            s0 = json.load(f)

        self.assertEqual(s0["stage"], "S0_CLIP")
        self.assertEqual(s0["model"], "openai/clip-vit-base-patch32")
        self.assertIn("max_numerical_discrepancy", s0["theme1_mechanistic_attention"])

        metrics = s0["theme4_zero_shot"]["selected_test_metrics"]
        self.assertEqual(
            set(metrics.keys()),
            {"accuracy", "macro_f1", "ece", "nll", "brier", "confusion_matrix", "bin_stats"},
        )
        self.assertEqual(set(metrics["bin_stats"].keys()), {"accuracies", "confidences", "counts"})

        linear_probe = s0["theme4_linear_probe"]
        self.assertIn("loss_curve", linear_probe)
        self.assertIn("rob_acc_003", linear_probe)
        self.assertIn("tau95", linear_probe)
        self.assertIn("clip_linear_probe_rob_acc_003", s0["cross_effects_input"])


class TestSplitContract(unittest.TestCase):
    """Guards the restored 3-seed split contract and continual-learning metadata."""

    def test_locked_seeds_and_metadata(self):
        with open("configs/splits.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)

        self.assertEqual(set(cfg["seeds"].keys()), {"42", "123", "999"})
        self.assertEqual(cfg["continual_learning"]["task1_classes"], [0, 1, 2])
        self.assertEqual(cfg["continual_learning"]["task2_classes"], [3, 4, 5])
        self.assertAlmostEqual(cfg["split_distribution"]["train_pct"], 0.7)

        for seed in ("42", "123", "999"):
            self.assertEqual(len(cfg["seeds"][seed]["train"]), 1260)
            self.assertEqual(len(cfg["seeds"][seed]["cal"]), 180)
            self.assertEqual(len(cfg["seeds"][seed]["test"]), 360)


class TestEvaluateAllContract(unittest.TestCase):
    """Ensures evaluate_all returns the full unified metric contract and XAI score."""

    def test_unified_metrics_and_xai(self):
        import numpy as np
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from src.eval import evaluate_all

        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 24 * 24, 6))
        x = torch.rand(12, 3, 24, 24)
        y = torch.randint(0, 6, (12,))
        loader = DataLoader(TensorDataset(x, y), batch_size=6)

        out = evaluate_all(model, loader, device=torch.device("cpu"))
        expected = {
            "accuracy", "macro_f1", "ece", "nll", "brier", "confusion_matrix",
            "bin_stats", "acc", "rob_acc_003", "xai_sim", "num_samples", "forgetting",
        }
        self.assertTrue(expected.issubset(out.keys()))
        self.assertIsNone(out["xai_sim"])
        self.assertIsNone(out["forgetting"])
        self.assertEqual(len(out["confusion_matrix"]), 6)
        self.assertGreaterEqual(out["rob_acc_003"], 0.0)
        self.assertLessEqual(out["rob_acc_003"], 1.0)

        out_xai = evaluate_all(
            model,
            loader,
            device=torch.device("cpu"),
            xai_saliency_maps=np.random.rand(12, 48, 48),
            xai_reference_maps=np.random.rand(12, 7, 7),
        )
        self.assertIsInstance(out_xai["xai_sim"], float)
        self.assertGreaterEqual(out_xai["xai_sim"], -1.0)
        self.assertLessEqual(out_xai["xai_sim"], 1.0)


if __name__ == "__main__":
    unittest.main()
