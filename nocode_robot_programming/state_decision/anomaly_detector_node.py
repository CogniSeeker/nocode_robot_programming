"""Always-on DINO-attention anomaly detector for demonstrated robot tasks.

The node deliberately has no state-selection or branching logic.  A retrain
request selects a task, all demonstrations for that task become the nominal
training set, and every new image is classified as either ``continue`` or
``anomaly``.
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path
import sys
import threading
import time
import traceback

from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import rclpy
from sensor_msgs.msg import Image
from std_msgs.msg import String
import torch
import torch.nn.functional as F

from lfd_msgs.srv import StringService
from skills_manager.ros_utils import SpinningRosNode
import trajectory_data

from nocode_robot_programming.state_decision.dino_model import (
    DINOFeaturePresenceAttnGated,
)
from nocode_robot_programming.state_decision.utils import Filename


IMAGE_TOPIC = "/modified_img"
RESULT_TOPIC = "/target_state"
RETRAIN_SERVICE = "/state_decider_retrain"

NORMAL_RESULT = "continue"
ANOMALY_RESULT = "anomaly"
MODEL_INPUT_SIZE = 224
DEFAULT_PERCENTILE_KEEP = 0.1
MAX_TRAIN_TIME = 300.0
WARNING_WHEN_IMAGE_OLDER_THAN = 0.2


def _prepare_images(images: np.ndarray) -> torch.Tensor:
    """Convert saved/live grayscale images to ``[N, 224, 224]`` in ``[0, 1]``."""
    array = np.asarray(images)
    if array.ndim == 2:
        array = array[None, ...]
    elif array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    elif array.ndim == 4 and array.shape[-1] == 1:
        array = array[..., 0]

    if array.ndim != 3:
        raise ValueError(
            f"Expected grayscale images shaped [N,H,W], got {array.shape}"
        )

    tensor = torch.from_numpy(np.ascontiguousarray(array))
    if tensor.dtype == torch.uint8:
        tensor = tensor.float().div_(255.0)
    elif tensor.dtype == torch.uint16:
        tensor = tensor.float().div_(65535.0)
    else:
        tensor = tensor.float()
        if tensor.numel() and (tensor.min() < 0 or tensor.max() > 1):
            raise ValueError("Floating-point images must be in the range [0, 1]")

    # Use the same geometry for saved demonstrations and live images.  Resizing
    # here also lets demonstrations with different camera resolutions coexist.
    return F.interpolate(
        tensor.unsqueeze(1),
        size=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(1)


def load_task_demonstrations(
    task_name: str,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Load nominal images from demo files belonging to ``task_name``.

    Each demonstrated trajectory part is an internal prototype class.  The
    class identity is used only to make the open-set model cover multimodal
    demonstrations; it is never published as a state or branch decision.
    """
    trajectory_dir = Path(trajectory_data.package_path) / "trajectories"
    if not trajectory_dir.is_dir():
        raise FileNotFoundError(
            f"Trajectory directory does not exist: {trajectory_dir}"
        )

    demo_files = []
    for path in sorted(trajectory_dir.rglob("*.npz")):
        parsed = Filename(path.stem)
        if parsed.task == task_name and parsed.is_demo:
            demo_files.append((path, parsed.part_name))

    if not demo_files:
        raise FileNotFoundError(
            f"No demonstrations found for task {task_name!r} under {trajectory_dir}"
        )

    class_names = list(dict.fromkeys(part_name for _, part_name in demo_files))
    class_ids = {name: index for index, name in enumerate(class_names)}
    image_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []

    for path, part_name in demo_files:
        with np.load(path, allow_pickle=False) as data:
            if "img" not in data:
                raise KeyError(f"Demonstration {path} has no 'img' array")
            images = _prepare_images(data["img"])

        if images.shape[0] == 0:
            raise ValueError(f"Demonstration {path} contains no images")
        image_batches.append(images)
        label_batches.append(
            torch.full((images.shape[0],), class_ids[part_name], dtype=torch.long)
        )

    if not image_batches:
        raise ValueError(f"Demonstrations for task {task_name!r} contain no images")

    return torch.cat(image_batches), torch.cat(label_batches), class_names


class DINOAttentionAnomalyDetectorNode(SpinningRosNode):
    """Train on demonstrations and continuously gate live images for anomalies."""

    def __init__(self, percentile_keep: float = DEFAULT_PERCENTILE_KEEP):
        super().__init__()
        if not 0.0 <= percentile_keep <= 1.0:
            raise ValueError("percentile_keep must be between 0 and 1")

        self.percentile_keep = percentile_keep
        self.bridge = CvBridge()
        self.model: DINOFeaturePresenceAttnGated | None = None

        self.create_service(
            StringService,
            RETRAIN_SERVICE,
            self.train_call,
            callback_group=self.callback_group,
        )
        self.create_subscription(Image, IMAGE_TOPIC, self.image_callback, 1)
        self.result_pub = self.create_publisher(String, RESULT_TOPIC, 5)

        self._image_lock = threading.Lock()
        self._latest_image: torch.Tensor | None = None
        self._latest_image_timestamp = 0.0
        self._image_sequence = 0
        self.new_image = threading.Event()

        self._training_lock = threading.Lock()
        self._requested_task: str | None = None
        self._training_error: Exception | None = None
        self.training_request = threading.Event()
        self.training_finished = threading.Event()

    def train_call(self, request, response):
        task_name = request.text.strip()
        if not task_name:
            response.success = False
            return response

        with self._training_lock:
            self._requested_task = task_name
            self._training_error = None
            self.training_finished.clear()
            self.training_request.set()

        if not self.training_finished.wait(MAX_TRAIN_TIME):
            self.get_logger().error(
                f"Training did not finish within {MAX_TRAIN_TIME:.0f} seconds"
            )
            response.success = False
            return response

        if self._training_error is not None:
            self.get_logger().error(f"Failed to train task {task_name!r}")
            response.success = False
            return response

        response.success = True
        return response

    def train_requested_task(self) -> None:
        with self._training_lock:
            task_name = self._requested_task
        if task_name is None:
            raise RuntimeError("A training request has no task name")

        images, labels, class_names = load_task_demonstrations(task_name)
        model = DINOFeaturePresenceAttnGated(
            input_size=MODEL_INPUT_SIZE,
            percentile_keep=self.percentile_keep,
        )
        labels = labels.to(model.device)
        model.train(X=images, y=labels, y_cls=class_names)

        # Replace the active detector only after training succeeds, so a failed
        # retrain cannot discard a working model.
        self.model = model
        print(
            f"Trained DINOattn anomaly detector for {task_name!r}: "
            f"{images.shape[0]} frames, {len(class_names)} demo prototype(s)",
            flush=True,
        )

    def image_callback(self, message: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding="mono8")
            prepared = _prepare_images(image)[0]
            with self._image_lock:
                self._latest_image = prepared
                self._latest_image_timestamp = time.time()
                self._image_sequence += 1
            self.new_image.set()
        except CvBridgeError as error:
            self.get_logger().error(f"Could not convert {IMAGE_TOPIC} image: {error}")
        except Exception:
            self.get_logger().error(
                f"Image callback failed:\n{traceback.format_exc()}"
            )

    def take_latest_image(self) -> tuple[torch.Tensor | None, float, int]:
        with self._image_lock:
            return (
                self._latest_image,
                self._latest_image_timestamp,
                self._image_sequence,
            )

    def predict(self, image: torch.Tensor) -> tuple[str, str]:
        if self.model is None:
            return "nomodel", "waiting for a retrain request"

        model_image = image.to(self.model.device, non_blocking=True)
        raw_prediction = self.model.predict(model_image)
        result = ANOMALY_RESULT if raw_prediction == ANOMALY_RESULT else NORMAL_RESULT

        scores = getattr(self.model, "last_scores", None)
        note = " ".join(f"{name}={score:.3f}" for name, score in scores or [])
        self.result_pub.publish(String(data=result))
        return result, note


def _parse_args(args: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Always-on DINO-attention anomaly detector"
    )
    parser.add_argument(
        "--percentile-keep",
        type=float,
        default=DEFAULT_PERCENTILE_KEEP,
        help=(
            "Lower training-score quantile used as the anomaly threshold "
            f"(default: {DEFAULT_PERCENTILE_KEEP})"
        ),
    )
    ros_args = sys.argv if args is None else args
    return parser.parse_args(rclpy.utilities.remove_ros_args(args=ros_args)[1:])


def main(args: list[str] | None = None) -> None:
    parsed_args = _parse_args(args)
    rclpy.init(args=args)
    node = DINOAttentionAnomalyDetectorNode(parsed_args.percentile_keep)
    last_sequence = 0
    last_result: str | None = None
    last_error: str | None = None

    print(
        f"DINOattn anomaly detector is running; waiting on {RETRAIN_SERVICE}",
        flush=True,
    )

    try:
        while rclpy.ok():
            if node.training_request.is_set():
                node.training_request.clear()
                print("\n============ Training in progress ============", flush=True)
                try:
                    node.train_requested_task()
                    node._training_error = None
                    print("============ Training finished ============", flush=True)
                except Exception as error:
                    node._training_error = error
                    print(f"[training ERROR]\n{traceback.format_exc()}", flush=True)
                finally:
                    node.training_finished.set()

            image, timestamp, sequence = node.take_latest_image()
            if image is None or sequence == last_sequence:
                node.new_image.wait(timeout=0.1)
                node.new_image.clear()
                continue

            last_sequence = sequence
            started = time.perf_counter()
            try:
                age = time.time() - timestamp
                result, note = node.predict(image)
                if age > WARNING_WHEN_IMAGE_OLDER_THAN:
                    stale_note = f"image age={age:.2f}s"
                    note = f"{note} | {stale_note}" if note else stale_note
                elapsed = max(time.perf_counter() - started, 1e-9)
                now = datetime.datetime.now().strftime("%H:%M:%S")
                status = f"[{now}] {result} | {1.0 / elapsed:.1f} frames/s"
                if note:
                    status += f" | {note}"
                if result != last_result:
                    print(f"\n{status}", flush=True)
                    last_result = result
                else:
                    print(f"\r\033[K{status}", end="", flush=True)
                last_error = None
            except Exception:
                error = traceback.format_exc()
                if error != last_error:
                    print(f"[prediction ERROR]\n{error}", flush=True)
                    last_error = error
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
