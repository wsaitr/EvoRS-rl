"""Deterministic non-overlapping spatial mosaic for small-object perception."""
from __future__ import annotations

from numbers import Real

from PIL import Image

from rs_hmcomm.agents.action_protocol import CallTool
from rs_hmcomm.backends.base import VisualInput
from rs_hmcomm.contracts import Evidence

from .base import ToolContext, ToolInputError, ToolResult


class SpatialMosaicTool:
    """Magnify every region of one scene without duplicating image content."""

    name = "spatial_mosaic"
    description = (
        "Split one scene into an N x N non-overlapping grid and enlarge the "
        "tiles in their original spatial arrangement."
    )

    def execute(self, call: CallTool, context: ToolContext) -> ToolResult:
        if call.tool_name != self.name:
            raise ToolInputError(
                f"SpatialMosaicTool cannot execute a '{call.tool_name}' call"
            )
        unknown = set(call.arguments) - {
            "image_ref",
            "grid_size",
            "tile_side",
            "gap",
        }
        if unknown:
            raise ToolInputError(
                f"spatial_mosaic unknown fields: {sorted(unknown)}"
            )
        image_ref = call.arguments.get("image_ref")
        if not isinstance(image_ref, str) or not image_ref.strip():
            raise ToolInputError("image_ref must be a non-empty string")
        grid_size = self._integer(
            call.arguments.get("grid_size", 3), "grid_size", 2, 4
        )
        tile_side = self._integer(
            call.arguments.get("tile_side", 336), "tile_side", 112, 768
        )
        gap = self._integer(call.arguments.get("gap", 2), "gap", 0, 16)
        try:
            source = context.visuals_by_id[image_ref]
        except KeyError as exc:
            raise ToolInputError(f"unknown image_ref: {image_ref}") from exc
        if not isinstance(source.image, Image.Image):
            raise ToolInputError("spatial_mosaic source must be a PIL image")

        mosaic = self.build(
            source.image,
            grid_size=grid_size,
            tile_side=tile_side,
            gap=gap,
        )
        visual_id = f"visual_{call.call_id}"
        visual = VisualInput(
            visual_id=visual_id,
            label=f"spatial_mosaic_{grid_size}x{grid_size}",
            image=mosaic,
        )
        evidence = Evidence(
            evidence_id=f"evidence_{call.call_id}",
            kind="spatial_mosaic",
            summary=(
                f"A {grid_size}x{grid_size} non-overlapping magnified view "
                f"of {image_ref}."
            ),
            created_by=context.agent_id,
            created_step=context.step,
            payload={
                "tool_name": self.name,
                "call_id": call.call_id,
                "source_image_ref": image_ref,
                "source_size": list(source.image.size),
                "grid_size": grid_size,
                "tile_side": tile_side,
                "gap": gap,
                "output_size": list(mosaic.size),
                "content_overlap": False,
            },
        )
        return ToolResult(
            call_id=call.call_id,
            tool_name=self.name,
            ok=True,
            observation=(
                f"Created {grid_size}x{grid_size} spatial mosaic as {visual_id}."
            ),
            evidence=(evidence,),
            visuals=(visual,),
            metadata={
                "grid_size": grid_size,
                "tile_side": tile_side,
                "gap": gap,
                "content_overlap": False,
            },
        )

    @staticmethod
    def build(
        image: Image.Image,
        *,
        grid_size: int,
        tile_side: int,
        gap: int = 2,
    ) -> Image.Image:
        """Return a spatially ordered mosaic with exact, non-overlapping tiles."""

        source = image.convert("RGB")
        width, height = source.size
        canvas_side = grid_size * tile_side + (grid_size - 1) * gap
        canvas = Image.new("RGB", (canvas_side, canvas_side), (8, 8, 8))
        resampling = getattr(Image, "Resampling", Image).BICUBIC
        for row in range(grid_size):
            top = round(row * height / grid_size)
            bottom = round((row + 1) * height / grid_size)
            for col in range(grid_size):
                left = round(col * width / grid_size)
                right = round((col + 1) * width / grid_size)
                crop = source.crop((left, top, right, bottom))
                crop.thumbnail((tile_side, tile_side), resampling)
                panel = Image.new("RGB", (tile_side, tile_side), (0, 0, 0))
                panel.paste(
                    crop,
                    ((tile_side - crop.width) // 2, (tile_side - crop.height) // 2),
                )
                canvas.paste(
                    panel,
                    (col * (tile_side + gap), row * (tile_side + gap)),
                )
        return canvas

    @staticmethod
    def _integer(value: object, name: str, low: int, high: int) -> int:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ToolInputError(f"{name} must be an integer")
        parsed = int(value)
        if parsed != value or not low <= parsed <= high:
            raise ToolInputError(f"{name} must be within [{low}, {high}]")
        return parsed


__all__ = ["SpatialMosaicTool"]
