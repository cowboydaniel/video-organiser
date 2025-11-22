"""Operations manager for applying rules to media files."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Mapping

from .metadata import MetadataReader
from .rules import DestinationPlan, RuleEngine


ProgressCallback = Callable[[int, int], None]


@dataclass(slots=True)
class OperationReport:
    """Result of attempting to apply a plan."""

    plan: DestinationPlan
    success: bool
    error: Exception | None = None


class Organizer:
    """Perform dry-run previews and commit file moves/copies."""

    def __init__(self, rule_engine: RuleEngine, metadata_reader: MetadataReader | None = None) -> None:
        self.rule_engine = rule_engine
        self.metadata_reader = metadata_reader or MetadataReader()

    def build_plans(
        self, sources: Iterable[Path], custom_tags: Mapping[str, str] | None = None
    ) -> List[DestinationPlan]:
        plans: List[DestinationPlan] = []
        for source in sources:
            metadata = self.metadata_reader.read(source)
            plan = self.rule_engine.resolve(source, metadata, custom_tags)
            plans.append(plan)
        return plans

    def preview(self, sources: Iterable[Path], custom_tags: Mapping[str, str] | None = None) -> List[DestinationPlan]:
        """Return destination plans without touching the filesystem."""

        return self.build_plans(sources, custom_tags)

    def commit(
        self,
        plans: Iterable[DestinationPlan],
        *,
        copy_files: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> List[OperationReport]:
        """Execute plans and roll back on failure."""

        executed: list[DestinationPlan] = []
        reports: list[OperationReport] = []
        plans_list = list(plans)
        total = len(plans_list)

        for index, plan in enumerate(plans_list, start=1):
            try:
                self._apply_plan(plan, copy_files)
            except Exception as exc:  # pragma: no cover - surfaced to UI/CLI
                reports.append(OperationReport(plan=plan, success=False, error=exc))
                self._rollback(executed, copy_files)
                break
            else:
                executed.append(plan)
                reports.append(OperationReport(plan=plan, success=True))
            if progress_callback is not None:
                progress_callback(index, total)

        return reports

    def _apply_plan(self, plan: DestinationPlan, copy_files: bool) -> None:
        destination_parent = plan.destination.parent
        destination_parent.mkdir(parents=True, exist_ok=True)

        if plan.destination.exists():
            raise FileExistsError(plan.destination)

        if copy_files:
            shutil.copy2(plan.source, plan.destination)
        else:
            plan.source.replace(plan.destination)

    def _rollback(self, executed: list[DestinationPlan], copy_files: bool) -> None:
        for plan in reversed(executed):
            try:
                if copy_files:
                    if plan.destination.exists():
                        plan.destination.unlink()
                else:
                    if plan.destination.exists():
                        plan.destination.replace(plan.source)
            except OSError:
                continue
