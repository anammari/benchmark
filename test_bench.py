"""test_bench.py
Runs hub models in benchmark mode using pytest-benchmark. Run setup separately first.

Usage:
  python test.py --setup_only
  pytest test_bench.py

See pytest-benchmark help (pytest test_bench.py -h) for additional options
e.g. --benchmark-autosave
     --benchmark-compare
     -k <filter expression>
     ...
"""
import os
import pytest
import time
import torch
from torchbenchmark import workdir, list_models


def pytest_generate_tests(metafunc, display_len=24):
    # This is where the list of models to test can be configured
    # e.g. by using info in metafunc.config
    all_models = list(list_models())
    short_names = []
    for _, name in all_models:
        short = os.path.split(name)[1]
        if len(short) > display_len:
            short = short[:display_len] + "..."
        short_names.append(short)

    if metafunc.cls and metafunc.cls.__name__ == "TestBenchNetwork":
        metafunc.parametrize('model', all_models,
                             ids=short_names, scope="class")
        metafunc.parametrize('device', ['cpu', 'cuda'], scope='class')
        metafunc.parametrize('compiler', ['jit', 'eager'], scope='class')


@pytest.fixture(scope='class')
def hub_model(request, model, device, compiler):
    """Constructs a model object for pytests to use.
    Any pytest function that consumes a 'modeldef' arg will invoke this
    automatically, and reuse it for each test that takes that combination
    of arguments within the module.

    If reusing the module between tests isn't safe, change 'scope' parameter.
    """
    Model, model_path = model
    with workdir(model_path):
        use_jit = compiler == 'jit'
        return Model(device=device, jit=use_jit)


def cuda_timer():
    torch.cuda.synchronize()
    return time.perf_counter()


@pytest.mark.benchmark(
    warmup=True,
    warmup_iterations=3,
    disable_gc=True,
    timer=cuda_timer if (torch.has_cuda and
                         torch.cuda.is_available()) else time.perf_counter,
    group='hub',
)
class TestBenchNetwork:
    """
    This test class will get instantiated once for each 'model_stuff' provided
    by the fixture above, for each device listed in the device parameter.
    """
    def test_train(self, hub_model, benchmark):
        try:
            benchmark(hub_model.train)
        except NotImplementedError:
            print('Method train is not implemented, skipping...')

    def test_eval(self, hub_model, benchmark):
        try:
            benchmark(hub_model.eval)
        except NotImplementedError:
            print('Method eval is not implemented, skipping...')
