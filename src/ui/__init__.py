"""UI module for NER-OCR pipeline workbench."""


def create_app() -> object:
    """Lazy import to avoid loading heavy dependencies on module import."""
    from .app import create_app as _create_app

    return _create_app()


def launch_workbench(*args, **kwargs) -> None:  # noqa: ANN002
    """Lazy import to avoid loading heavy dependencies on module import."""
    from .app import launch_workbench as _launch_workbench

    return _launch_workbench(*args, **kwargs)


__all__ = ["create_app", "launch_workbench"]
