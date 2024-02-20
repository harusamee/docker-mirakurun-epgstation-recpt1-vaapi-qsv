import os
import subprocess
import argparse
import datetime
from multiprocessing.pool import ThreadPool

import numpy as np
import cv2
from ultralytics import YOLO

DURATION=1.001

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", nargs="+", default=[])
    parser.add_argument("-o", "--output")
    parser.add_argument("--target-cls", default={"airplane", "bus", "train", "truck"})
    parser.add_argument("--ffmpeg-video-encoder", default=None, type=str)
    parser.add_argument("--min-part-length", default=5.0, type=float)

    return parser.parse_args()

def get_timestamp(input_video):
    command = ["ffprobe",
               "-select_streams", "0",
               "-show_entries", "frame=pict_type,pkt_dts_time",
               "-of", "csv=p=0",
               "-loglevel", "quiet",
               input_video]

    timestamp = []
    ph = subprocess.Popen(command, stdout=subprocess.PIPE)

    while True:
        line = ph.stdout.readline()
        if not line: break
        try:
            tsstr, pict_type, _ = line.split(b",")
            if pict_type == b"I":
                timestamp.append(float(tsstr))
        except:
            pass

    if len(timestamp) > 0:
        return np.array(timestamp) - timestamp[0]
    return np.array([])

def detect_objects(input_video):
    model = YOLO("yolov8n.pt")
    if not os.path.exists("yolov8n_openvino_model"):
        model.export(format="openvino")
    ov_model = YOLO('yolov8n_openvino_model/')

    command = ["ffmpeg",
            "-skip_frame", "nokey",
            "-i", input_video,
            "-an",
            "-fps_mode", "vfr",
            "-vf", "yadif=0:-1:1,scale=iw*sar:ih,setsar=1",
            "-pix_fmt", "bgr24",
            "-f", "rawvideo",
            "pipe:1",
            "-loglevel", "quiet",
            "-nostats",
            "-hide_banner"]

    cv2_video = cv2.VideoCapture(input_video)
    sar = cv2_video.get(cv2.CAP_PROP_SAR_NUM) / cv2_video.get(cv2.CAP_PROP_SAR_DEN)
    height = int(cv2_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cv2_video.get(cv2.CAP_PROP_FRAME_WIDTH) * sar)
    frame_bytes = height * width * 3

    ph = subprocess.Popen(command, stdout=subprocess.PIPE)
    count = 0
    detected = []
    while True:
        data = ph.stdout.read(frame_bytes)
        if len(data) != frame_bytes:
            break
        image = np.frombuffer(data, dtype=np.uint8)
        image.shape = height, width, 3

        results = ov_model.predict(source=image, verbose=False)
        names = []
        for result in results:
            for cls in result.boxes.cls:
                name = result.names[cls.item()]
                names.append(name)
        detected.append(set(names))
        count += 1

    ph.stdout.close()
    ph.kill()

    return detected


def blur_bool_1d(input: np.ndarray|list, count: int):
    if not isinstance(input, np.ndarray) or input.ndim != 1:
        raise ValueError("Input must be a 1D numpy array or list")
    if not isinstance(count, int) or count < 0:
        raise ValueError("count must be a non-negative integer")

    if count == 0:
        return input

    for _ in range(count):
        input[0:-1] = input[0:-1] | input[1:]
        input[1:] = input[1:] | input[0:-1]

    return input

def analyze_video(input_video):
    npz_path, _ = os.path.splitext(input_video)
    npz_path_w_ext = f"{npz_path}.npz"

    if not os.path.exists(npz_path_w_ext):
        pool = ThreadPool()
        detect_result = pool.apply_async(detect_objects, (input_video,))
        timestamp_result = pool.apply_async(get_timestamp, (input_video,))
        detected = detect_result.get()
        timestamp = timestamp_result.get()
        pool.close()
        pool.terminate()

        np.savez(npz_path, detected=detected, timestamp=timestamp)
    else:
        print("# Using previous detection result.")
        npz_file = np.load(npz_path_w_ext, allow_pickle=True)
        detected = npz_file["detected"]
        timestamp = npz_file["timestamp"]

    return detected, timestamp

def detected_to_bool_1d(detected: list[set], target_cls: set):
    return np.array([bool(target_cls & d) for d in detected])

def get_stream_io(i, suffix):
    if i == 0:
        input_stream = f"[{i}{suffix}][{i+1}{suffix}]"
    else:
        input_stream = f"[0to{i}{suffix}][{i+1}{suffix}]"
    output_stream = f"[0to{i+1}{suffix}]"

    return input_stream, output_stream

def get_parts(input_video, detected, timestamp, min_part_length):
    from enum import Enum
    class SearchEdge(Enum):
        RISING = 0
        FALLING = 1

    prev = False
    search_edge = SearchEdge.RISING
    rising_ts = 0
    parts = []
    for curr, ts in zip(detected, timestamp):
        if search_edge == SearchEdge.RISING and prev == False and curr == True:
            search_edge = SearchEdge.FALLING
            rising_ts = ts
        if search_edge == SearchEdge.FALLING and prev == True and curr == False:
            search_edge = SearchEdge.RISING
            part_length = ts - rising_ts
            if part_length >= min_part_length:
                prev_offset = 0 if len(parts) == 0 else parts[-1][3]
                xfade_offset = prev_offset + part_length - DURATION
                parts.append((input_video, rising_ts, part_length, xfade_offset))
        prev = curr

    return parts

def get_crossfade_commands(parts, encoder, name, ext):
    # https://stackoverflow.com/questions/64696381/ffmpeg-desynced-audio-when-using-xfade-and-acrossfade-together
    filters_video = []
    filters_audio = []
    for i, (_, _, t, _) in enumerate(parts):
        filters_video.append(f"[{i}:v] scale=720:-1,tpad=stop=300:stop_mode=clone,trim=0:{t} [{i}v]")
        filters_audio.append(f"[{i}:a] apad=pad_dur=10,atrim=0:{t} [{i}a]")

    command_video = ["ffmpeg"]
    command_audio = ["ffmpeg"]
    for i, (input_video, ss, t, offset) in enumerate(parts):
        command_video += ["-ss", str(ss), "-t", str(t), "-i", input_video]
        command_audio += ["-ss", str(ss), "-t", str(t), "-i", input_video]

        if i != len(parts) - 1:
            iv, ov = get_stream_io(i, "v")
            ia, oa = get_stream_io(i, "a")
            xfade = f"{iv} xfade=duration={DURATION}:offset={offset} {ov}"
            acrossfade = f"{ia} acrossfade=duration={DURATION} {oa}"
            filters_video.append(xfade)
            filters_audio.append(acrossfade)
        else:
            if len(parts) == 1:
                ov = "[0v]"
                oa = "[0a]"
            else:
                _, ov = get_stream_io(i - 1, "v")
                _, oa = get_stream_io(i - 1, "a")
            command_video += ["-filter_complex"] + ["'" + ";".join(filters_video) + "'"]
            command_video += ["-map", ov, "-y", "-an"]
            command_audio += ["-filter_complex"] + ["'" + ";".join(filters_audio) + "'"]
            command_audio += ["-map", oa, "-y", "-vn"]
            if encoder is not None:
                command_video += ["-c:v", encoder]
            command_audio += ["-c:a", "aac"]
            command_video += ["-f", "mp4", f"{name}.m4v"]
            command_audio += [f"{name}.m4a"]

    command_mux = f"ffmpeg -i {name}.m4v -i {name}.m4a -c copy -f mp4 -y {name}{ext}"
    command_rm = f"rm {name}.m4v {name}.m4a"

    return [" ".join(command_video), " ".join(command_audio), command_mux, command_rm]

def get_conversion_command_and_video_fn(input_video, encoder):
    input_video_basename, input_video_ext = os.path.splitext(input_video)
    if input_video_ext in [".m2ts", ".ts"]:
        input_video_mp4 = input_video_basename + ".mp4"
        if not os.path.exists(input_video_mp4):
            encoder = encoder or "hevc"
            command = f"ffmpeg -i {input_video} -vf yadif -c:v {encoder} -q 23 -b:a 192k -y {input_video_mp4}"
            return [command], input_video_mp4
    return [], input_video

def main(args):
    temp_dir = 'temp_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    print(f"mkdir -p {temp_dir}")

    video_lengths = {}
    for i, input_video in enumerate(args.input):
        print(f"# Analyzing {input_video}")
        detected, timestamp = analyze_video(input_video)
        detected = detected_to_bool_1d(detected, args.target_cls)
        detected = blur_bool_1d(detected, 1)

        commands, input_video = get_conversion_command_and_video_fn(input_video, args.ffmpeg_video_encoder)

        parts = get_parts(input_video, detected, timestamp, args.min_part_length)
        print(f"# {input_video} has {len(parts)} parts")
        if len(parts) > 0:
            temp_name = os.path.join(temp_dir, str(i))
            commands += get_crossfade_commands(parts, args.ffmpeg_video_encoder, temp_name, ".mp4")
            video_lengths[i] = parts[-1][3] + DURATION

            for command in commands:
                print(command)

    parts = []
    offset = 0
    for i, length in video_lengths.items():
        temp_name = os.path.join(temp_dir, f"{i}.mp4")
        offset += length - DURATION
        parts.append((temp_name, 0, length, offset))

    output, output_ext = os.path.splitext(args.output)
    for command in get_crossfade_commands(parts, args.ffmpeg_video_encoder, output, output_ext):
        print(command)


args = parse_args()
main(args)