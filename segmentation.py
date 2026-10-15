
import os
import json
import argparse

import cv2
import numpy as np
import torch

from PIL import Image
from transformers import (
    AutoImageProcessor,
    SegformerForSemanticSegmentation
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

OUTPUT_DIR = "output"

MIN_AREA_RATIO = 0.01

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Simplified plant-care profiles.
# Lighting categories are estimates, not measured lux values.

PLANTS = {
    "snake_plant": {
        "preferred_surfaces": [
            "floor", "table", "shelf", "cabinet"
        ],
        "light": ["low", "medium", "high"],
        "space": "small"
    },

    "money_plant": {
        "preferred_surfaces": [
            "table", "shelf", "cabinet"
        ],
        "light": ["medium", "high"],
        "space": "small"
    },

    "areca_palm": {
        "preferred_surfaces": ["floor"],
        "light": ["medium", "high"],
        "space": "large"
    },

    "peace_lily": {
        "preferred_surfaces": [
            "floor", "table", "shelf"
        ],
        "light": ["low", "medium"],
        "space": "medium"
    },

    "aloe_vera": {
        "preferred_surfaces": [
            "table", "shelf", "window"
        ],
        "light": ["high"],
        "space": "medium"
    }
}

# ADE20K semantic labels that may serve as candidate surfaces.
# Note: a detected region is not proof of a safe, empty surface.

SURFACE_LABELS = {
    "floor",
    "table",
    "shelf",
    "cabinet",
    "desk",
    "counter",
    "window"
}


# ============================================================
# LOAD SEGMENTATION MODEL
# ============================================================

print(f"Loading segmentation model on {DEVICE}...")

processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME
)

model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_NAME
)

model.to(DEVICE)
model.eval()

print("Segmentation model loaded successfully.")


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(image_path):
    """Load an image and return RGB and OpenCV BGR versions."""

    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(image_path).convert("RGB")

    rgb = np.asarray(image)

    bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )

    return image, rgb, bgr


# ============================================================
# SEMANTIC SEGMENTATION
# ============================================================

def segment_room(image):
    """
    Segment an RGB PIL image.

    Returns:
        mask: H x W semantic class ID mask
        labels: class ID to label mapping
        class_coverage: percentage of image per detected class
    """

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)

    # Resize predictions to the original image dimensions.
    logits = torch.nn.functional.interpolate(
        outputs.logits,
        size=(image.height, image.width),
        mode="bilinear",
        align_corners=False
    )

    mask = (
        logits.argmax(dim=1)[0]
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    labels = {
        int(class_id): str(label)
        for class_id, label
        in model.config.id2label.items()
    }

    total_pixels = mask.size

    class_coverage = {}

    for class_id in np.unique(mask):
        label = labels.get(
            int(class_id),
            f"class_{class_id}"
        )

        percentage = (
            np.count_nonzero(mask == class_id)
            / total_pixels
        ) * 100

        class_coverage[label] = round(
            float(percentage), 2
        )

    return mask, labels, class_coverage


# ============================================================
# LIGHTING ESTIMATION
# ============================================================

def analyze_lighting(rgb_image):
    """
    Estimate relative image brightness.

    This is NOT a sunlight or lux measurement.
    Camera exposure, shadows and white balance affect the result.
    """

    gray = cv2.cvtColor(
        rgb_image,
        cv2.COLOR_RGB2GRAY
    )

    brightness = float(np.mean(gray))

    if brightness < 70:
        light_level = "low"

    elif brightness < 160:
        light_level = "medium"

    else:
        light_level = "high"

    return {
        "brightness": round(brightness, 2),
        "light_level": light_level,
        "method": "Mean grayscale brightness",
        "warning": (
            "Relative image brightness is not a "
            "measurement of plant-available sunlight."
        )
    }


# ============================================================
# CANDIDATE SURFACE DETECTION
# ============================================================

def get_placement_candidates(mask, labels):
    """
    Find connected regions assigned to candidate surface classes.

    Returns bounding boxes and image-space center points.
    These are 2D image regions, not verified 3D placement points.
    """

    height, width = mask.shape

    candidates = []

    for class_id, label in labels.items():

        surface = label.lower().strip()

        if surface not in SURFACE_LABELS:
            continue

        binary = (
            mask == class_id
        ).astype(np.uint8)

        count, component_map, stats, centers = (
            cv2.connectedComponentsWithStats(
                binary,
                connectivity=8
            )
        )

        for index in range(1, count):

            x, y, w, h, area = stats[index]

            area_ratio = area / (height * width)

            if area_ratio < MIN_AREA_RATIO:
                continue

            candidates.append({
                "surface": surface,
                "class_id": int(class_id),
                "bbox": [
                    int(x),
                    int(y),
                    int(x + w),
                    int(y + h)
                ],
                "center": [
                    int(centers[index][0]),
                    int(centers[index][1])
                ],
                "area_pixels": int(area),
                "area_ratio": round(
                    float(area_ratio), 4
                )
            })

    candidates.sort(
        key=lambda item: item["area_pixels"],
        reverse=True
    )

    return candidates


# ============================================================
# PLANT PLACEMENT SCORING
# ============================================================

def score_candidate(
    candidate,
    plant_info,
    light_level
):
    """
    Calculate a heuristic compatibility score from 0 to 100.

    This is a rule-based score, not a trained prediction probability.
    """

    score = 0.0
    reasons = []

    surface = candidate["surface"]

    # Surface suitability.
    if surface in plant_info["preferred_surfaces"]:
        score += 50
        reasons.append(
            "Surface is in the plant's preferred list."
        )
    else:
        reasons.append(
            "Surface is not in the plant's preferred list."
        )

    # Approximate lighting compatibility.
    if light_level in plant_info["light"]:
        score += 30
        reasons.append(
            "Estimated brightness category is compatible."
        )
    else:
        reasons.append(
            "Estimated brightness category may be unsuitable."
        )

    # Larger visible regions receive a limited bonus.
    # This does not establish actual physical free space.
    area_bonus = min(
        candidate["area_ratio"] * 100,
        20
    )

    score += area_bonus

    return {
        "score": round(min(score, 100), 2),
        "reasons": reasons
    }


def rank_candidates(
    candidates,
    plant_info,
    light_level
):
    """Rank candidate image regions for a selected plant."""

    ranked = []

    for candidate in candidates:

        evaluation = score_candidate(
            candidate,
            plant_info,
            light_level
        )

        ranked.append({
            **candidate,
            "compatibility_score": evaluation["score"],
            "reasons": evaluation["reasons"]
        })

    ranked.sort(
        key=lambda item: item["compatibility_score"],
        reverse=True
    )

    return ranked


# ============================================================
# SEGMENTATION VISUALIZATION
# ============================================================

def create_segmentation_visualization(
    rgb_image,
    mask,
    labels
):
    """
    Create a colored semantic mask and blend it with the room image.
    """

    height, width = mask.shape

    color_mask = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    rng = np.random.default_rng(42)

    class_ids = np.unique(mask)

    for class_id in class_ids:

        # Assign a stable visualization color per class.
        color = rng.integers(
            40, 240, size=3
        ).tolist()

        color_mask[mask == class_id] = color

    blended = cv2.addWeighted(
        rgb_image,
        0.55,
        color_mask,
        0.45,
        0
    )

    return blended


def draw_recommendations(
    rgb_image,
    ranked_candidates
):
    """Draw candidate boxes and the top-ranked image region."""

    output = cv2.cvtColor(
        rgb_image.copy(),
        cv2.COLOR_RGB2BGR
    )

    for index, candidate in enumerate(
        ranked_candidates[:5]
    ):

        x1, y1, x2, y2 = candidate["bbox"]

        # Green for the highest-ranked region.
        # Orange for alternatives.
        color = (
            (0, 200, 0)
            if index == 0
            else (0, 165, 255)
        )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        label = (
            f'{candidate["surface"]}: '
            f'{candidate["compatibility_score"]:.0f}'
        )

        text_y = max(y1 - 8, 20)

        cv2.putText(
            output,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

    return output


# ============================================================
# COMPLETE AI PIPELINE
# ============================================================

def analyze_plant_placement(
    image_path,
    plant_name,
    output_dir=OUTPUT_DIR
):
    """
    Run segmentation, candidate extraction, lighting estimation,
    and plant-specific recommendation.

    Returns a JSON-serializable dictionary.
    """

    plant_name = plant_name.lower().strip()

    if plant_name not in PLANTS:
        return {
            "success": False,
            "error": (
                f"Unknown plant '{plant_name}'. "
                f"Available plants: {list(PLANTS.keys())}"
            )
        }

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    image, rgb_image, bgr_image = load_image(
        image_path
    )

    # Step 1: Semantic segmentation.
    mask, labels, class_coverage = segment_room(
        image
    )

    # Step 2: Approximate lighting analysis.
    lighting = analyze_lighting(
        rgb_image
    )

    # Step 3: Extract possible placement regions.
    candidates = get_placement_candidates(
        mask,
        labels
    )

    # Step 4: Rank regions according to plant requirements.
    plant_info = PLANTS[plant_name]

    ranked = rank_candidates(
        candidates,
        plant_info,
        lighting["light_level"]
    )

    # Step 5: Generate diagnostic visualizations.
    segmentation_image = create_segmentation_visualization(
        rgb_image,
        mask,
        labels
    )

    segmentation_path = os.path.join(
        output_dir,
        "segmentation_result.jpg"
    )

    cv2.imwrite(
        segmentation_path,
        cv2.cvtColor(
            segmentation_image,
            cv2.COLOR_RGB2BGR
        )
    )

    recommendation_path = None

    if ranked:

        recommendation_image = draw_recommendations(
            rgb_image,
            ranked
        )

        recommendation_path = os.path.join(
            output_dir,
            "placement_result.jpg"
        )

        cv2.imwrite(
            recommendation_path,
            recommendation_image
        )

    # Step 6: Return structured results.
    result = {
        "success": bool(ranked),
        "plant": plant_name,
        "device": str(DEVICE),
        "lighting": lighting,
        "detected_classes": class_coverage,
        "candidate_count": len(candidates),
        "recommended_location": (
            ranked[0] if ranked else None
        ),
        "alternatives": ranked[1:4],
        "segmentation_image": segmentation_path,
        "recommendation_image": recommendation_path,
        "limitations": [
            "Segmentation regions are 2D, not 3D surfaces.",
            "Detected furniture may not have free space.",
            "Brightness is not a lux measurement.",
            "Scores are heuristic, not validated probabilities.",
            "Actual AR anchoring requires a separate AR framework."
        ]
    }

    return result


# ============================================================
# COMMAND-LINE ENTRY POINT
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "PLANTSCAPE AI Plant Placement "
            "Segmentation and Recommendation"
        )
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to a room image"
    )

    parser.add_argument(
        "--plant",
        default="snake_plant",
        choices=list(PLANTS.keys()),
        help="Plant to recommend a location for"
    )

    parser.add_argument(
        "--output",
        default=OUTPUT_DIR,
        help="Directory for generated results"
    )

    args = parser.parse_args()

    result = analyze_plant_placement(
        image_path=args.image,
        plant_name=args.plant,
        output_dir=args.output
    )

    print(
        json.dumps(
            result,
            indent=2
        )
    )


if __name__ == "__main__":
    main()
