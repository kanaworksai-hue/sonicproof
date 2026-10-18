import shutil
import subprocess

import numpy as np
import pytest

from sonicproof import visual
from sonicproof.errors import InputError, SetupError
from sonicproof.visual import MAX_FRAMES, has_video_stream, load_image, load_video, video_frame_times


def test_frame_times_follow_one_fps_and_cap_long_clips():
    assert video_frame_times(5.0) == [0.5, 1.5, 2.5, 3.5, 4.5]
    assert video_frame_times(0.4) == [0.2]
    long = video_frame_times(600.0)
    assert len(long) == MAX_FRAMES
    assert long[0] > 0 and long[-1] < 600 and np.all(np.diff(long) > 0)


def test_load_image_is_rgb(tmp_path):
    Image = pytest.importorskip("PIL.Image")

    p = tmp_path / "frame.png"
    Image.new("RGBA", (32, 24), (255, 0, 0, 128)).save(p)
    im = load_image(p)
    assert im.mode == "RGB" and im.size == (32, 24)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_load_video_samples_frames(tmp_path):
    p = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=4",
         "-pix_fmt", "yuv420p", str(p)],
        check=True,
    )  # fmt: skip
    fr = load_video(p)
    assert fr.frames.shape[1:] == (240, 320, 3) and fr.frames.dtype == np.uint8
    assert len(fr.frames) == 4 and fr.duration == pytest.approx(4.0, abs=0.1)
    assert has_video_stream(p)
    assert not has_video_stream(tmp_path / "missing.wav")


def test_bad_image_is_an_input_error(tmp_path):
    pytest.importorskip("PIL")
    p = tmp_path / "x.png"
    p.write_bytes(b"not a png")
    with pytest.raises(InputError) as e:
        load_image(p)
    assert e.value.code == "unreadable"
    with pytest.raises(InputError) as e:
        load_image(tmp_path / "missing.png")
    assert e.value.code == "not_found"


def test_no_ffprobe_is_a_setup_error_not_no_video(tmp_path, monkeypatch):
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"\0" * 64)
    monkeypatch.setattr(visual.shutil, "which", lambda name: None)
    with pytest.raises(SetupError) as e:
        has_video_stream(p)
    assert e.value.code == "needs_ffmpeg"
    with pytest.raises(SetupError):
        load_video(p)
    assert not has_video_stream(tmp_path / "sound.wav")  # not a video by name: no ffprobe needed


def _make(path, *args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args, str(path)], check=True)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
@pytest.mark.parametrize(
    ("size", "shape"), [("321x241", (240, 320)), ("240x427", (428, 240)), ("2x2", (2, 2))]
)
def test_odd_and_portrait_sizes(tmp_path, size, shape):
    p = tmp_path / "clip.mkv"
    _make(p, "-f", "lavfi", "-i", f"testsrc=size={size}:rate=10:duration=2", "-c:v", "ffv1")
    fr = load_video(p)
    assert fr.frames.shape[1:3] == shape and len(fr.frames) == 2


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_rotated_phone_video_comes_out_portrait(tmp_path):
    src, rot = tmp_path / "a.mp4", tmp_path / "rot.mp4"
    _make(src, "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=2", "-pix_fmt", "yuv420p")
    try:
        _make(rot, "-display_rotation", "90", "-i", str(src), "-c", "copy")
    except subprocess.CalledProcessError:
        pytest.skip("this ffmpeg cannot write display rotation")
    fr = load_video(rot)
    assert fr.frames.shape[1:3] == (320, 240)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_audio_only_container_has_no_video(tmp_path):
    p = tmp_path / "tone.mp4"
    _make(p, "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:a", "aac")
    assert not has_video_stream(p)
    with pytest.raises(InputError):
        load_video(p)
