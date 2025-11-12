"""Custom Streamlit component that renders the animated React landing page."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import streamlit.components.v1 as components


def _get_component_func():
    build_dir = Path(__file__).resolve().parent / "frontend" / "dist"
    if not build_dir.exists():
        raise RuntimeError(
            "React assets are missing. Run `cd landing_component/frontend && npm run build` first."
        )
    return components.declare_component("iris_landing", path=str(build_dir))


_component_func = _get_component_func()


def iris_landing_component(
    *,
    title: str,
    subtitle: str,
    highlight: str,
    bullets: Iterable[str],
    metrics: Iterable[Mapping[str, str]],
    cta_label: str,
):
    """Render the landing component and return True when the CTA is clicked."""

    return bool(
        _component_func(
            title=title,
            subtitle=subtitle,
            highlight=highlight,
            bullets=list(bullets),
            metrics=list(metrics),
            ctaLabel=cta_label,
            default=False,
        )
    )


__all__ = ["iris_landing_component"]
