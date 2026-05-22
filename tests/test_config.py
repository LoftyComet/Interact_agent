import os

from gesture_agent.cli import main
from gesture_agent.providers import SiliconFlowClient
from gesture_agent.settings import get_env, load_agent_config, load_env_file


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


def test_load_agent_config_reads_runtime_and_prompt(tmp_path) -> None:
    config = tmp_path / "agent_config.json"
    config.write_text(
        """
{
  // 检索配置支持注释。
  "retrieval": {
    "top_k": 2
  },
  "data": {
    "structured_knowledge": "data/custom_structured.json"
  },
  /*
    运行配置也支持块注释。
  */
  "runtime": {
    "stream": true,
    "default_interactive": false
  },
  "model": {
    "enable_thinking": null
  },
  "prompt": {
    "extra_system_prompt": ["第一行", "第二行"],
    "extra_response_instructions": ["只使用 output_frame 标题"]
  }
}
""".strip(),
        encoding="utf-8",
    )

    loaded = load_agent_config(config)

    assert loaded.top_k == 2
    assert loaded.structured_knowledge == "data/custom_structured.json"
    assert loaded.stream is True
    assert loaded.default_interactive is False
    assert loaded.enable_thinking is None
    assert loaded.prompt.extra_system_prompt == "第一行\n第二行"
    assert loaded.prompt.extra_response_instructions == ["只使用 output_frame 标题"]


def test_cli_uses_agent_config_for_dry_run(tmp_path, capsys) -> None:
    config = tmp_path / "agent_config.json"
    config.write_text(
        """
{
  "retrieval": {
    "top_k": 1
  },
  "runtime": {
    "dry_run": true,
    "show_structure": true,
    "show_context": true,
    "default_interactive": false
  }
}
""".strip(),
        encoding="utf-8",
    )

    exit_code = main(["--config", str(config), "什么是单击？"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "问题结构" in captured.out
    assert "检索资料" in captured.out
    assert captured.out.count("score=") == 1


def test_cli_defaults_to_interactive_when_no_question(tmp_path, monkeypatch) -> None:
    config = tmp_path / "agent_config.json"
    config.write_text('{"runtime": {"default_interactive": true}}', encoding="utf-8")
    monkeypatch.setattr("builtins.input", lambda _prompt="": (_ for _ in ()).throw(EOFError()))

    assert main(["--config", str(config)]) == 0
