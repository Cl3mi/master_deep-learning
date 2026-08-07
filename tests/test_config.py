from glys_rul import config


def test_repo_root_contains_pyproject():
    assert (config.REPO_ROOT / "pyproject.toml").is_file()


def test_scale_range_is_ordered():
    assert config.SCALE_VMIN < config.SCALE_VMAX


def test_reports_dir_is_under_repo_root():
    assert config.REPORTS_DIR.is_relative_to(config.REPO_ROOT)


def test_region_names_match_expected_count():
    assert len(config.REGION_NAMES) == config.EXPECTED_REGIONS
