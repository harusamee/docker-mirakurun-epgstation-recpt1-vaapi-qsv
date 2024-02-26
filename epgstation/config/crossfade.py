import os
import subprocess
import argparse
import datetime

import numpy as np

from classify import analyze_video

DURATION=1.001

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-video")
    parser.add_argument("-j", "--input-json")
    parser.add_argument("-o", "--output")
    parser.add_argument("--target-cls", default={"airplane", "bus", "train", "truck"})
    parser.add_argument("--min-confidence", default=0.6)
    parser.add_argument("--min-area", default=0.15)
    parser.add_argument("--ffmpeg-video-encoder", default="hevc_qsv", type=str)
    parser.add_argument("--min-part-length", default=5.0, type=float)
    parser.add_argument("--remove-temp-dir", action="store_true")
    parser.add_argument("--execute-commands", action="store_true")

    return parser.parse_args()

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


def get_stream_io(i, suffix):
    if i == 0:
        input_stream = f"[{i}{suffix}][{i+1}{suffix}]"
    else:
        input_stream = f"[0to{i}{suffix}][{i+1}{suffix}]"
    output_stream = f"[0to{i+1}{suffix}]"

    return input_stream, output_stream

def get_parts(input_video, detected, timestamp, options):
    from enum import Enum
    class SearchEdge(Enum):
        RISING = 0
        FALLING = 1

    min_part_length = options.min_part_length
    min_confidence = options.min_confidence
    min_area = options.min_area

    prev = False
    search_edge = SearchEdge.RISING
    rising_ts = 0
    max_confidence_in_part = 0
    max_area_in_part = 0
    parts = []

    def checker(part_length, max_confidence_in_part, max_area_in_part):
        return part_length >= min_part_length \
            and max_confidence_in_part >= min_confidence \
            and max_area_in_part >= min_area

    for (curr, conf, area), ts in zip(detected, timestamp):
        if search_edge == SearchEdge.RISING and prev == False and curr == True:
            search_edge = SearchEdge.FALLING
            rising_ts = ts
        if search_edge == SearchEdge.FALLING and prev == True and curr == False:
            search_edge = SearchEdge.RISING
            part_length = ts - rising_ts
            if checker(part_length, max_confidence_in_part, max_area_in_part):
                prev_offset = 0 if len(parts) == 0 else parts[-1][3]
                xfade_offset = prev_offset + part_length - DURATION
                parts.append((input_video, rising_ts, part_length, xfade_offset))
            else:
                print(f"# part not added {part_length} {max_confidence_in_part} {max_area_in_part}")

            max_confidence_in_part = 0
            max_area_in_part = 0

        if search_edge == SearchEdge.FALLING:
            max_confidence_in_part = max([max_confidence_in_part, conf])
            max_area_in_part = max([max_area_in_part, area])

        prev = curr

    return parts

def get_crossfade_commands(parts, encoder, name, ext):
    # https://stackoverflow.com/questions/64696381/ffmpeg-desynced-audio-when-using-xfade-and-acrossfade-together
    filters_video = []
    filters_audio = []

    command_video = ["ffmpeg"]
    command_audio = ["ffmpeg"]

    for i, (_, _, t, _) in enumerate(parts):
        # scale=720:-1
        filters_video.append(f"[{i}:v] tpad=stop=300:stop_mode=clone,trim=0:{t} [{i}v]")
        filters_audio.append(f"[{i}:a] apad=pad_dur=10,atrim=0:{t} [{i}a]")

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
            command_video += ["-filter_complex"] + [";".join(filters_video)]
            command_video += ["-map", ov, "-y", "-an", "-pix_fmt", "nv12"]
            command_audio += ["-filter_complex"] + [";".join(filters_audio)]
            command_audio += ["-map", oa, "-y", "-vn"]
            if encoder is not None:
                command_video += ["-c:v", encoder]
            command_audio += ["-c:a", "aac"]
            command_video += ["-f", "mp4", f"{name}.m4v"]
            command_audio += [f"{name}.m4a"]

    command_mux = ["ffmpeg",
                   "-i", f"{name}.m4v", "-i", f"{name}.m4a",
                   "-movflags", "faststart",
                   "-c", "copy",
                   "-f", "mp4",
                   "-y", f"{name}{ext}"]
    command_rm = ["rm", f"{name}.m4v", f"{name}.m4a"]

    return [command_video, command_audio, command_mux, command_rm]

def reinterpret_detected(detected, target_cls):
    unique_names = np.array([{name for name, _, _ in d} for d in detected])
    have_target = np.array([bool(target_cls & n) for n in unique_names])
    have_target = blur_bool_1d(have_target, 1)
    max_confidence = np.array([max([0] + [c if n in target_cls else 0 for (n, c, _) in d]) for d in detected])
    max_area = np.array([max([0] + [w * h if n in target_cls else 0 for (n, _, (_, _, w, h)) in d]) for d in detected])

    return zip(have_target, max_confidence, max_area)

def main(args):
    commands = []
    temp_dir = 'temp_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    commands.append(["mkdir", "-p", temp_dir])

    # video_lengths = {}
    print(f"# Analyzing {args.input_video}")
    detected, timestamp = analyze_video(args.input_video, args.input_json)
    detected = reinterpret_detected(detected, args.target_cls)

    parts = get_parts(args.input_video, detected, timestamp, args)
    print(f"# {args.input_video} has {len(parts)} parts")
    if len(parts) > 0:
        output, output_ext = os.path.splitext(args.output)
        commands += get_crossfade_commands(parts, args.ffmpeg_video_encoder, output, output_ext)
        # temp_name = os.path.join(temp_dir, str(i))
        # video_lengths[i] = parts[-1][3] + DURATION

        # copy an encoded video from the temporary directory
        # commands.append(["mv", f"{temp_dir}/0.mp4", args.output])

        # else:
        #     parts = []
        #     offset = 0
        #     for i, length in video_lengths.items():
        #         temp_name = os.path.join(temp_dir, f"{i}.mp4")
        #         offset += length - DURATION
        #         parts.append((temp_name, 0, length, offset))
        #     if len(parts) > 0:
        #         output, output_ext = os.path.splitext(args.output)
        #         commands += get_crossfade_commands(parts, args.ffmpeg_video_encoder, output, output_ext)
    else:
        # No target classes are detected
        input = ["-f", "lavfi", "-i", "color=black:s=1920x1080:r=1:d=1"]
        message = "No target detected"
        filter = ["-vf", f"drawtext=fontsize=64:text={message}:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2"]
        commands.append(["ffmpeg"] + input +  filter + ["-y", args.output])

    if args.remove_temp_dir:
        commands.append(["rm", "-rf", temp_dir])

    for c in commands:
        print(" ".join(c))
        if args.execute_commands:
            ph = subprocess.Popen(c, shell=False)
            ph.communicate()
            if ph.returncode != 0:
                raise SystemError(f"Command '{c}' returned {ph.returncode}")

if __name__ == "__main__":
    args = parse_args()
    main(args)