"""
Unit tests for the model_config module.

Tests cover:
- ModelConfig dataclass validation
- ModelManager initialization
- PEFT configuration setup
- Model information retrieval
- Factory function behavior
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_config import (
    ModelConfig,
    ModelManager,
    PeftMethod,
    create_model_manager,
    PRESET_CONFIGS,
)


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_model_config_defaults(self):
        """Test ModelConfig has correct defaults."""
        config = ModelConfig()

        assert config.model_name == "gpt2"
        assert config.peft_method == "lora"
        assert config.lora_r == 8
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.05
        assert config.target_modules is None
        assert config.bits is None
        assert config.device_map == "auto"
        assert config.use_cache is False

    def test_model_config_custom_values(self):
        """Test ModelConfig with custom values."""
        config = ModelConfig(
            model_name="gpt2-medium",
            peft_method="qlora",
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["c_attn"],
            bits=4,
            device_map="cuda:0",
        )

        assert config.model_name == "gpt2-medium"
        assert config.peft_method == "qlora"
        assert config.lora_r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.1
        assert config.target_modules == ["c_attn"]
        assert config.bits == 4
        assert config.device_map == "cuda:0"

    def test_model_config_normalizes_peft_method(self):
        """Test that peft_method is normalized to lowercase."""
        config = ModelConfig(peft_method="LORA")
        assert config.peft_method == "lora"

    def test_model_config_torch_dtype_string(self):
        """Test that torch_dtype is converted from string."""
        config = ModelConfig(torch_dtype="float16")
        assert config.torch_dtype == torch.float16

    def test_model_config_torch_dtype_bfloat16(self):
        """Test that torch_dtype handles bfloat16."""
        config = ModelConfig(torch_dtype="bfloat16")
        assert config.torch_dtype == torch.bfloat16

    def test_model_config_torch_dtype_auto(self):
        """Test that torch_dtype handles 'auto'."""
        config = ModelConfig(torch_dtype="auto")
        assert config.torch_dtype == "auto"


class TestPeftMethod:
    """Tests for PeftMethod enum."""

    def test_peft_method_values(self):
        """Test PeftMethod enum has correct values."""
        assert PeftMethod.LORA.value == "lora"
        assert PeftMethod.QLORA.value == "qlora"
        assert PeftMethod.NONE.value == "none"


class TestModelManager:
    """Tests for ModelManager class."""

    def test_model_manager_init(self):
        """Test ModelManager initialization."""
        config = ModelConfig()
        manager = ModelManager(config)

        assert manager.config == config
        assert manager.model is None
        assert manager.tokenizer is None
        assert manager.peft_config is None
        assert manager.is_peft_model is False

    def test_count_parameters_no_model(self):
        """Test count_parameters returns 0 when no model loaded."""
        config = ModelConfig()
        manager = ModelManager(config)

        assert manager.count_parameters() == 0
        assert manager.count_parameters(trainable_only=True) == 0

    def test_count_parameters_with_model(self):
        """Test count_parameters with a mock model."""
        config = ModelConfig()
        manager = ModelManager(config)

        # Create mock model with parameters
        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.numel = MagicMock(return_value=1000)
        mock_param.requires_grad = True
        mock_model.parameters = MagicMock(return_value=[mock_param, mock_param])

        manager.model = mock_model

        assert manager.count_parameters() == 2000
        assert manager.count_parameters(trainable_only=True) == 2000

    def test_get_model_info_no_model(self):
        """Test get_model_info when model not loaded."""
        config = ModelConfig()
        manager = ModelManager(config)

        info = manager.get_model_info()

        assert "error" in info
        assert info["error"] == "Model not loaded"

    def test_get_model_info_with_model(self):
        """Test get_model_info with a loaded model."""
        config = ModelConfig()
        manager = ModelManager(config)

        # Create mock model
        mock_model = MagicMock()
        mock_model.config.model_type = "gpt2"
        mock_model.config.vocab_size = 50257
        mock_model.config.hidden_size = 768
        mock_model.config.num_hidden_layers = 12
        mock_model.config.num_attention_heads = 12
        mock_model.config.pad_token_id = None
        mock_model.config.eos_token_id = 50256

        mock_param = MagicMock()
        mock_param.numel = MagicMock(return_value=124439808)
        mock_param.requires_grad = True
        mock_model.parameters = MagicMock(return_value=[mock_param])

        manager.model = mock_model

        info = manager.get_model_info()

        assert info["model_name"] == "gpt2"
        assert info["model_type"] == "gpt2"
        assert info["vocab_size"] == 50257
        assert info["total_parameters"] == 124439808

    def test_get_target_modules_gpt2(self):
        """Test _get_target_modules for GPT-2."""
        config = ModelConfig(model_name="gpt2")
        manager = ModelManager(config)

        modules = manager._get_target_modules()

        assert "c_attn" in modules
        assert "c_proj" in modules

    def test_get_target_modules_custom(self):
        """Test _get_target_modules with custom modules."""
        config = ModelConfig(
            model_name="gpt2",
            target_modules=["custom_module"]
        )
        manager = ModelManager(config)

        modules = manager._get_target_modules()

        assert modules == ["custom_module"]


class TestModelManagerLoadModel:
    """Tests for ModelManager model loading."""

    @patch("src.model_config.AutoModelForCausalLM.from_pretrained")
    def test_load_model_basic(self, mock_from_pretrained):
        """Test basic model loading."""
        mock_model = MagicMock()
        mock_model.config.pad_token_id = None
        mock_model.config.eos_token_id = 50256
        mock_model.config.use_cache = True
        mock_from_pretrained.return_value = mock_model

        config = ModelConfig(model_name="gpt2")
        manager = ModelManager(config)

        model = manager.load_model()

        assert manager.model is not None
        mock_from_pretrained.assert_called_once()

    @patch("src.model_config.AutoModelForCausalLM.from_pretrained")
    def test_load_model_sets_pad_token(self, mock_from_pretrained):
        """Test that loading model sets pad_token_id to eos_token_id."""
        mock_model = MagicMock()
        mock_model.config.pad_token_id = None
        mock_model.config.eos_token_id = 50256
        mock_model.config.use_cache = True
        mock_from_pretrained.return_value = mock_model

        config = ModelConfig(model_name="gpt2")
        manager = ModelManager(config)

        manager.load_model()

        assert mock_model.config.pad_token_id == 50256


class TestModelManagerLoadTokenizer:
    """Tests for ModelManager tokenizer loading."""

    @patch("src.model_config.AutoTokenizer.from_pretrained")
    def test_load_tokenizer_basic(self, mock_from_pretrained):
        """Test basic tokenizer loading."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "pad"
        mock_tokenizer.pad_token_id = 1
        mock_from_pretrained.return_value = mock_tokenizer

        config = ModelConfig(model_name="gpt2")
        manager = ModelManager(config)

        tokenizer = manager.load_tokenizer()

        assert manager.tokenizer is not None
        mock_from_pretrained.assert_called_once()

    @patch("src.model_config.AutoTokenizer.from_pretrained")
    def test_load_tokenizer_sets_pad_token(self, mock_from_pretrained):
        """Test that loading tokenizer sets pad_token when None."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "eos"
        mock_tokenizer.pad_token_id = None
        mock_tokenizer.eos_token_id = 50256
        mock_from_pretrained.return_value = mock_tokenizer

        config = ModelConfig(model_name="gpt2")
        manager = ModelManager(config)

        manager.load_tokenizer()

        assert mock_tokenizer.pad_token == "eos"


class TestModelManagerPeftSetup:
    """Tests for ModelManager PEFT setup."""

    @patch("src.model_config.get_peft_model")
    @patch("src.model_config.LoraConfig")
    def test_setup_peft_creates_lora_config(self, mock_lora_config, mock_get_peft):
        """Test that setup_peft creates LoraConfig correctly."""
        mock_model = MagicMock()
        mock_peft_model = MagicMock()
        mock_peft_model.print_trainable_parameters = MagicMock()
        mock_get_peft.return_value = mock_peft_model

        config = ModelConfig(model_name="gpt2", peft_method="lora")
        manager = ModelManager(config)
        manager.model = mock_model

        result = manager.setup_peft()

        assert manager.is_peft_model is True
        mock_lora_config.assert_called_once()

    def test_setup_peft_none_method(self):
        """Test that setup_peft returns model when peft_method is 'none'."""
        mock_model = MagicMock()

        config = ModelConfig(model_name="gpt2", peft_method="none")
        manager = ModelManager(config)
        manager.model = mock_model

        result = manager.setup_peft()

        assert manager.is_peft_model is False
        assert result == mock_model

    def test_setup_peft_raises_without_model(self):
        """Test that setup_peft raises error when model not loaded."""
        config = ModelConfig(model_name="gpt2", peft_method="lora")
        manager = ModelManager(config)

        with pytest.raises(ValueError, match="Model must be loaded"):
            manager.setup_peft()


class TestCreateModelManager:
    """Tests for create_model_manager factory function."""

    def test_create_model_manager_defaults(self):
        """Test create_model_manager with defaults."""
        manager = create_model_manager()

        assert isinstance(manager, ModelManager)
        assert manager.config.model_name == "gpt2"
        assert manager.config.peft_method == "lora"

    def test_create_model_manager_custom(self):
        """Test create_model_manager with custom values."""
        manager = create_model_manager(
            model_name="gpt2-medium",
            peft_method="qlora",
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.1,
        )

        assert manager.config.model_name == "gpt2-medium"
        assert manager.config.peft_method == "qlora"
        assert manager.config.lora_r == 16
        assert manager.config.lora_alpha == 32
        assert manager.config.lora_dropout == 0.1


class TestPresetConfigs:
    """Tests for preset configurations."""

    def test_preset_configs_exist(self):
        """Test that expected preset configs exist."""
        assert "gpt2_small" in PRESET_CONFIGS
        assert "gpt2_medium" in PRESET_CONFIGS
        assert "distilgpt2" in PRESET_CONFIGS
        assert "full_finetune" in PRESET_CONFIGS

    def test_preset_config_values(self):
        """Test preset config values are reasonable."""
        gpt2_small = PRESET_CONFIGS["gpt2_small"]
        assert gpt2_small["model_name"] == "gpt2"
        assert gpt2_small["peft_method"] == "lora"
        assert gpt2_small["lora_r"] == 8

        full_finetune = PRESET_CONFIGS["full_finetune"]
        assert full_finetune["peft_method"] == "none"
