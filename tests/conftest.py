import numpy as np
import pytest

from body3d import bodies, landmarks


@pytest.fixture(scope="session")
def faces():
    return bodies.faces()


@pytest.fixture(scope="session")
def template():
    return bodies.template()


@pytest.fixture(scope="session")
def mixed():
    """Two bodies per mesh-BMI band, one of each sex."""
    return bodies.sample_stratified(2, seed=5)


@pytest.fixture(scope="session")
def lm_idx():
    return landmarks.load()


@pytest.fixture
def rng():
    return np.random.default_rng(0)
