def test_imports():
    import src.atc_classifier as pkg
    from src.atc_classifier.config import Config
    from src.atc_classifier.models.model import ATCModel
    assert hasattr(pkg, "__version__")
    assert ATCModel is not None
    assert Config is not None
