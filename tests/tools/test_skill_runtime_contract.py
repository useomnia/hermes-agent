"""Skill viewing, management, and execution share a backend path contract."""
import json
from pathlib import Path

import pytest

from agent import skill_utils
from tools import skills_tool, skill_manager_tool


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    root = tmp_path / 'profile' / 'skills'
    root.mkdir(parents=True)
    monkeypatch.setenv('HERMES_HOME', str(root.parent))
    monkeypatch.setenv('TERMINAL_ENV', 'local')
    monkeypatch.setattr(skills_tool, 'SKILLS_DIR', root)
    monkeypatch.setattr(skill_manager_tool, 'SKILLS_DIR', root)
    monkeypatch.setattr(skill_utils, 'get_external_skills_dirs', lambda: [])
    monkeypatch.setattr(skill_utils, 'get_all_skills_dirs', lambda: [root])
    return root


def write_skill(root, relative, body='Use the helper.'):
    directory = root / relative
    directory.mkdir(parents=True)
    (directory / 'SKILL.md').write_text(
        '---\nname: helper\ndescription: Test helper\n---\n# Usage\n' + body
    )
    return directory


def test_skill_view_publishes_runtime_directory_but_internal_load_keeps_host(catalog, monkeypatch):
    directory = write_skill(catalog, 'ops/helper', 'python ${HERMES_SKILL_DIR}/scripts/run.py')
    scripts = directory / 'scripts'
    scripts.mkdir()
    (scripts / 'run.py').write_text("print('helper executed')\n")
    monkeypatch.setenv('TERMINAL_ENV', 'sprites')
    result = json.loads(skills_tool.skill_view('ops/helper'))
    assert result['skill_dir'] == '/skills/ops/helper'
    assert '/skills/ops/helper/scripts/run.py' in result['content']
    from tools.credential_files import iter_skills_files
    assert any(f['container_path'] == result['skill_dir'] + '/scripts/run.py'
               for f in iter_skills_files('/skills'))
    internal = json.loads(skills_tool.skill_view('ops/helper', preprocess=False))
    assert internal['skill_dir'] == str(directory)


def test_categorized_skill_management_edits_host_file(catalog):
    directory = write_skill(catalog, 'ops/helper')
    result = json.loads(skill_manager_tool.skill_manage(
        action='patch', name='ops/helper', old_string='Use the helper.', new_string='Run the helper.'
    ))
    assert result['success'], result
    assert 'Run the helper.' in (directory / 'SKILL.md').read_text()


def test_directory_request_lists_available_support_files(catalog):
    directory = write_skill(catalog, 'helper')
    (directory / 'references').mkdir()
    (directory / 'references' / 'guide.md').write_text('Guide')
    result = json.loads(skills_tool.skill_view('helper', file_path='references'))
    assert result['success'] is False
    assert 'references/guide.md' in result['available_files']['references']


def test_identical_same_root_copy_prefers_shallow_skill(catalog):
    directory = write_skill(catalog, 'helper')
    write_skill(catalog, 'ops/helper')
    result = json.loads(skills_tool.skill_view('helper'))
    assert result['success'], result
    assert result['skill_dir'] == str(directory)


def test_distinct_same_root_skills_still_refuse_collision(catalog):
    write_skill(catalog, 'helper', 'Custom implementation')
    write_skill(catalog, 'ops/helper', 'Built-in implementation')
    result = json.loads(skills_tool.skill_view('helper'))
    assert result['success'] is False
    assert 'Ambiguous' in result['error']


@pytest.mark.parametrize('value', ['~/brand/config', '$HOME/brand/config', '${HOME}/brand/config'])
def test_skill_config_uses_toolbox_home(catalog, monkeypatch, value):
    monkeypatch.setenv('TERMINAL_ENV', 'sprites')
    monkeypatch.setattr(skill_utils, '_load_raw_config', lambda: {})
    result = skill_utils.resolve_skill_config_values([{'key': 'path', 'default': value}])
    assert result['path'] == '/home/brand/config'


def test_local_skill_directory_is_unchanged(catalog):
    directory = write_skill(catalog, 'helper')
    assert json.loads(skills_tool.skill_view('helper'))['skill_dir'] == str(directory)


def test_external_skill_runtime_path_matches_sync_destination(catalog, tmp_path, monkeypatch):
    from agent.skill_path_mapping import map_skill_dir_for_backend
    from tools.credential_files import iter_skills_files
    external = tmp_path / 'external'
    directory = write_skill(external, 'ops/helper')
    monkeypatch.setattr(skill_utils, 'get_external_skills_dirs', lambda: [external])
    monkeypatch.setenv('TERMINAL_ENV', 'sprites')
    runtime = map_skill_dir_for_backend(directory)
    assert runtime == '/skills/external_skills/0/ops/helper'
    assert any(f['container_path'] == runtime + '/SKILL.md' for f in iter_skills_files('/skills'))


def test_other_brand_skill_path_is_not_translated(catalog, tmp_path, monkeypatch):
    from agent.skill_path_mapping import map_skill_dir_for_backend
    other = write_skill(tmp_path / 'other-profile' / 'skills', 'helper')
    monkeypatch.setenv('TERMINAL_ENV', 'sprites')
    assert map_skill_dir_for_backend(other) == str(other)


def test_equal_depth_identical_copies_remain_ambiguous(catalog):
    write_skill(catalog, 'one/helper')
    write_skill(catalog, 'two/helper')
    assert json.loads(skills_tool.skill_view('helper'))['success'] is False


def test_identical_cross_root_copies_remain_ambiguous(catalog, tmp_path, monkeypatch):
    write_skill(catalog, 'helper')
    external = tmp_path / 'external'
    write_skill(external, 'nested/helper')
    monkeypatch.setattr(skill_utils, 'get_external_skills_dirs', lambda: [external])
    assert json.loads(skills_tool.skill_view('helper'))['success'] is False


def test_slash_command_advertises_runtime_support_paths(catalog, monkeypatch):
    from agent.skill_commands import _build_skill_message
    directory = write_skill(catalog, 'ops/helper')
    monkeypatch.setenv('TERMINAL_ENV', 'sprites')
    result = _build_skill_message(
        {'content': 'python ${HERMES_SKILL_DIR}/scripts/run.py',
         'linked_files': {'scripts': ['scripts/run.py']}}, directory, 'Activated'
    )
    assert '[Skill directory: /skills/ops/helper]' in result
    assert '/skills/ops/helper/scripts/run.py' in result
    assert str(directory) not in result
    assert 'name="ops/helper"' in result


def test_inline_shell_keeps_host_paths_while_instructions_use_toolbox(catalog, monkeypatch):
    from agent.skill_preprocessing import preprocess_skill_content
    directory = write_skill(catalog, 'helper')
    (directory / 'host.txt').write_text('HOST_CONTENT')
    monkeypatch.setenv('TERMINAL_ENV', 'sprites')
    content = '!`cat "${HERMES_SKILL_DIR}/host.txt"`\nRun ${HERMES_SKILL_DIR}/scripts/helper.py'
    rendered = preprocess_skill_content(content, directory, skills_cfg={'inline_shell': True})
    assert rendered == 'HOST_CONTENT\nRun /skills/helper/scripts/helper.py'


def test_docker_skill_path_uses_the_existing_mount(catalog, monkeypatch):
    from agent.skill_path_mapping import map_skill_dir_for_backend
    directory = write_skill(catalog, 'ops/helper')
    monkeypatch.setenv('TERMINAL_ENV', 'docker')
    assert map_skill_dir_for_backend(directory) == '/root/.hermes/skills/ops/helper'


def test_skill_config_preserves_non_home_environment_variables(catalog, monkeypatch):
    monkeypatch.setenv('TERMINAL_ENV', 'sprites')
    monkeypatch.setenv('ASSET_ROOT', '/assets')
    monkeypatch.setattr(skill_utils, '_load_raw_config', lambda: {})
    assert skill_utils.resolve_skill_config_values([
        {'key': 'path', 'default': '${ASSET_ROOT}/config'},
    ]) == {'path': '/assets/config'}


def test_categorized_external_skill_can_be_managed(catalog, tmp_path, monkeypatch):
    external = tmp_path / 'external'
    directory = write_skill(external, 'ops/helper')
    monkeypatch.setattr(skill_utils, 'get_all_skills_dirs', lambda: [catalog, external])
    assert skill_manager_tool._find_skill('ops/helper') == {'path': directory}


def test_missing_reference_still_returns_real_inventory(catalog):
    directory = write_skill(catalog, 'helper')
    (directory / 'references').mkdir()
    (directory / 'references' / 'actual.md').write_text('Actual instructions')
    result = json.loads(skills_tool.skill_view('helper', file_path='references/missing.md'))
    assert result['success'] is False
    assert result['available_files']['references'] == ['references/actual.md']


@pytest.mark.parametrize('name', ['omnio/helper', 'omnio:helper'])
def test_stale_category_suggests_installed_name_without_silent_fallback(catalog, name):
    write_skill(catalog, 'marketing/research/helper')
    result = json.loads(skills_tool.skill_view(name))
    assert result['success'] is False
    assert result['matching_skills'][0]['name'] == 'helper'
    assert json.loads(skills_tool.skill_view(result['matching_skills'][0]['name']))['success']
