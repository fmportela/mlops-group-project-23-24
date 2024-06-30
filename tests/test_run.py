from pathlib import Path

import pytest
from kedro.config import OmegaConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.hooks import _create_hook_manager


@pytest.fixture
def config_loader():
    return OmegaConfigLoader(conf_source=str(Path.cwd()))


@pytest.fixture
def project_context(config_loader):
    return KedroContext(
        env="base",
        package_name="mlops_group_project",
        project_path=Path.cwd(),
        config_loader=config_loader,
        hook_manager=_create_hook_manager(),
    )


# The tests below are here for the demonstration purpose
# and should be replaced with the ones testing the project
# functionality
class TestProjectContext:
    def test_project_path(self, project_context):
        assert project_context.project_path == Path.cwd()
