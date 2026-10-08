"""Workflow templates + compiler.

A template is an API-format ComfyUI workflow (``workflow.json``) plus ``params.yaml`` that maps
semantic parameter names (STYLE_PROMPT, SEED…) to ``(node id, input name)`` targets. Application
code only ever speaks semantic names; node IDs stay inside the template directory.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

_TYPES: dict[str, type] = {"str": str, "int": int, "float": float, "bool": bool}


class TemplateError(ValueError):
    pass


class ParamTarget(BaseModel):
    node: str
    input: str


class ParamSpec(BaseModel):
    targets: list[ParamTarget]
    type: str = "str"
    required: bool = False
    default: Any = None
    min: float | None = None
    max: float | None = None
    description: str = ""
    optional_node: bool = False


class TemplateSpec(BaseModel):
    name: str
    version: int
    description: str = ""
    resource_class: str = "GPU_HEAVY"
    input_kind: str = "video"  # video | image | none
    output_kinds: list[str] = Field(default_factory=lambda: ["images", "videos", "gifs"])
    parameters: dict[str, ParamSpec]


@dataclass
class WorkflowTemplate:
    spec: TemplateSpec
    workflow: dict[str, Any]
    path: Path

    @property
    def node_classes(self) -> set[str]:
        return {n["class_type"] for n in self.workflow.values()}


@dataclass
class CompiledWorkflow:
    template: str
    template_version: int
    workflow: dict[str, Any]
    applied: dict[str, Any]
    ignored: dict[str, Any] = field(default_factory=dict)


def _parse_params(raw: dict[str, Any]) -> dict[str, ParamSpec]:
    params = {}
    for name, spec in raw.items():
        spec = dict(spec)
        if "targets" not in spec:
            spec["targets"] = [{"node": str(spec.pop("node")), "input": spec.pop("input")}]
        params[name] = ParamSpec.model_validate(spec)
    return params


def load_template(directory: Path) -> WorkflowTemplate:
    workflow = json.loads((directory / "workflow.json").read_text(encoding="utf-8"))
    meta = yaml.safe_load((directory / "params.yaml").read_text(encoding="utf-8"))
    meta["parameters"] = _parse_params(meta.get("parameters") or {})
    spec = TemplateSpec.model_validate(meta)
    if not isinstance(workflow, dict) or not all(
        isinstance(n, dict) and "class_type" in n and "inputs" in n for n in workflow.values()
    ):
        raise TemplateError(f"{directory}: workflow.json is not ComfyUI API format "
                            "(export with 'Save (API Format)')")
    for pname, p in spec.parameters.items():
        if p.type not in _TYPES:
            raise TemplateError(f"{spec.name}: parameter {pname} has unknown type {p.type}")
        for t in p.targets:
            node = workflow.get(t.node)
            if node is None:
                raise TemplateError(f"{spec.name}: parameter {pname} targets missing node {t.node}")
            if t.input not in node["inputs"]:
                raise TemplateError(
                    f"{spec.name}: parameter {pname} targets missing input "
                    f"{t.node}.{t.input} ({node['class_type']})")
    return WorkflowTemplate(spec, workflow, directory)


class TemplateRegistry:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._cache: dict[str, WorkflowTemplate] = {}

    def names(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir()
                      if (p / "workflow.json").exists() and (p / "params.yaml").exists())

    def get(self, name: str) -> WorkflowTemplate:
        if name not in self._cache:
            directory = self.root / name
            if not (directory / "workflow.json").exists():
                raise TemplateError(
                    f"workflow template {name!r} is not installed in {self.root}. "
                    "Export it from ComfyUI in API format; see docs/comfyui.md")
            self._cache[name] = load_template(directory)
        return self._cache[name]


def _coerce(name: str, spec: ParamSpec, value: Any) -> Any:
    try:
        coerced = _TYPES[spec.type](value)
    except (TypeError, ValueError) as exc:
        raise TemplateError(f"parameter {name}: cannot convert {value!r} to {spec.type}") from exc
    if spec.type in ("int", "float"):
        if spec.min is not None and coerced < spec.min:
            raise TemplateError(f"parameter {name}={coerced} below minimum {spec.min}")
        if spec.max is not None and coerced > spec.max:
            raise TemplateError(f"parameter {name}={coerced} above maximum {spec.max}")
    return coerced


def compile_workflow(template: WorkflowTemplate, params: dict[str, Any]) -> CompiledWorkflow:
    """Substitute semantic parameters. Unknown parameters are reported, never silently dropped."""
    workflow = copy.deepcopy(template.workflow)
    applied: dict[str, Any] = {}
    for name, spec in template.spec.parameters.items():
        if name in params and params[name] is not None:
            value = _coerce(name, spec, params[name])
        elif spec.default is not None:
            value = _coerce(name, spec, spec.default)
        elif spec.required:
            raise TemplateError(f"{template.spec.name}: required parameter {name} missing")
        else:
            if spec.optional_node:
                for target in spec.targets:
                    workflow.pop(target.node, None)
                    for node in workflow.values():
                        node["inputs"] = {k: v for k, v in node["inputs"].items()
                                          if not (isinstance(v, list) and v
                                                  and str(v[0]) == target.node)}
            continue
        for t in spec.targets:
            workflow[t.node]["inputs"][t.input] = value
        applied[name] = value
    ignored = {k: v for k, v in params.items() if k not in template.spec.parameters}
    return CompiledWorkflow(template.spec.name, template.spec.version, workflow, applied, ignored)


def validate_against_object_info(template: WorkflowTemplate,
                                 object_info: dict[str, Any]) -> list[str]:
    """Problems that would make ComfyUI reject this template on this installation."""
    problems = []
    # Inputs a required parameter fills at render time (e.g. the uploaded INPUT_VIDEO) hold a
    # placeholder in the template, so their installed-file choices are not checked here.
    runtime = {(t.node, t.input) for p in template.spec.parameters.values()
               if (p.required or p.optional_node) and p.default is None for t in p.targets}
    for node_id, node in template.workflow.items():
        cls = node["class_type"]
        info = object_info.get(cls)
        if info is None:
            problems.append(f"node {node_id}: class {cls} not installed")
            continue
        declared = {**(info.get("input", {}).get("required") or {}),
                    **(info.get("input", {}).get("optional") or {})}
        for key, value in node["inputs"].items():
            if key not in declared:
                problems.append(f"node {node_id} ({cls}): unknown input {key!r}")
                continue
            spec = declared[key] or [None]
            choices = spec[0]
            if choices == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
                choices = spec[1].get("options")
            if (node_id, key) in runtime:
                continue
            if isinstance(choices, list) and not isinstance(value, list) and value not in choices:
                # An empty list means ComfyUI has no files at all for this loader (e.g. no checkpoints).
                hint = f"e.g. {choices[:3]}" if choices else "none installed"
                problems.append(f"node {node_id} ({cls}): {key}={value!r} not among installed "
                                f"options ({hint})")
    return problems
