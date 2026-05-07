import os

from gesture_agent.providers import SiliconFlowClient
from gesture_agent.settings import get_env, load_env_file


def test_load_env_file_without_override(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "SILICONFLOW_API_KEY=from_file\n"
        "SILICONFLOW_MODEL='quoted-model'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SILICONFLOW_API_KEY", "from_shell")

    load_env_file(str(env_file))

    assert os.environ["SILICONFLOW_API_KEY"] == "from_shell"
    assert os.environ["SILICONFLOW_MODEL"] == "quoted-model"


def test_get_env_treats_empty_as_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("EMPTY_VALUE=\n", encoding="utf-8")

    assert get_env("EMPTY_VALUE", "fallback") == "fallback"


def test_siliconflow_uses_text_model_by_default(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "SILICONFLOW_API_KEY=test-key\n"
        "SILICONFLOW_MODEL=text-model\n"
        "SILICONFLOW_VISION_MODEL=vision-model\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    monkeypatch.delenv("SILICONFLOW_MODEL", raising=False)
    monkeypatch.delenv("SILICONFLOW_VISION_MODEL", raising=False)

    client = SiliconFlowClient.from_env()

    assert client.model == "text-model"


def test_siliconflow_uses_vision_model_for_image_case(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "SILICONFLOW_API_KEY=test-key\n"
        "SILICONFLOW_MODEL=text-model\n"
        "SILICONFLOW_VISION_MODEL=vision-model\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    monkeypatch.delenv("SILICONFLOW_MODEL", raising=False)
    monkeypatch.delenv("SILICONFLOW_VISION_MODEL", raising=False)

    client = SiliconFlowClient.from_env(use_vision_model=True)

    assert client.model == "vision-model"


def test_explicit_model_overrides_vision_model(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "SILICONFLOW_API_KEY=test-key\n"
        "SILICONFLOW_MODEL=text-model\n"
        "SILICONFLOW_VISION_MODEL=vision-model\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    monkeypatch.delenv("SILICONFLOW_MODEL", raising=False)
    monkeypatch.delenv("SILICONFLOW_VISION_MODEL", raising=False)

    client = SiliconFlowClient.from_env(model="manual-model", use_vision_model=True)

    assert client.model == "manual-model"
