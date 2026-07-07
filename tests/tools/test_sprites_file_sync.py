import pytest


def test_iter_sprites_sync_files_should_include_only_skills(monkeypatch, tmp_path):
    from tools.environments.file_sync import iter_sprites_sync_files
    import tools.credential_files as credential_files

    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text("name: demo\n", encoding="utf-8")

    monkeypatch.setattr(credential_files, "get_credential_file_mounts", lambda: [])
    monkeypatch.setattr(
        credential_files,
        "iter_skills_files",
        lambda container_base: [
            {
                "host_path": str(skill_file),
                "container_path": f"{container_base}/demo/SKILL.md",
            }
        ],
    )

    assert iter_sprites_sync_files("/skills") == [
        (str(skill_file), "/skills/demo/SKILL.md")
    ]


def test_iter_sprites_sync_files_should_refuse_credentials(monkeypatch, tmp_path):
    from tools.environments.file_sync import iter_sprites_sync_files
    import tools.credential_files as credential_files

    credential = tmp_path / "secret.json"
    credential.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        credential_files,
        "get_credential_file_mounts",
        lambda: [
            {
                "host_path": str(credential),
                "container_path": "/root/.hermes/credentials/secret.json",
            }
        ],
    )
    monkeypatch.setattr(credential_files, "iter_skills_files", lambda container_base: [])

    with pytest.raises(RuntimeError, match="refuses to sync credential files"):
        iter_sprites_sync_files("/skills")
