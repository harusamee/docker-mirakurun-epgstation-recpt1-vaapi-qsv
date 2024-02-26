import os
import subprocess
import argparse
from multiprocessing.pool import ThreadPool

import numpy as np
import json
import cv2
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input")
    parser.add_argument("-o", "--output")
    parser.add_argument("-f", "--force", action="store_true")

    return parser.parse_args()

def get_timestamp(input_video):
    command = ["ffprobe",
               "-select_streams", "v",
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
            tsstr, pict_type = line.split(b",")[:2]
            if pict_type.startswith(b"I"):
                timestamp.append(float(tsstr))
        except:
            pass

    if len(timestamp) > 0:
        return (np.array(timestamp) - timestamp[0]).tolist()
    return []

def detect_objects(input_video):
    model = YOLO("yolov8n.pt", task="detect")
    if not os.path.exists("yolov8n_openvino_model"):
        model.export(format="openvino")
    ov_model = YOLO('yolov8n_openvino_model/', task="detect")

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
        objects = []
        for result in results:
            for cls, conf, xywhn in zip(result.boxes.cls, result.boxes.conf, result.boxes.xywhn):
                name = result.names[cls.item()]
                objects.append((name, float(conf), xywhn.tolist()))
        detected.append(objects)
        count += 1

    ph.stdout.close()
    ph.kill()

    return detected

def analyze_video(input_video, output_json, force=False):
    if not os.path.exists(output_json) or force:
        pool = ThreadPool()
        detect_result = pool.apply_async(detect_objects, (input_video,))
        timestamp_result = pool.apply_async(get_timestamp, (input_video,))
        detected = detect_result.get()
        timestamp = timestamp_result.get()
        pool.close()
        pool.terminate()

        result = {"detected": detected, "timestamp": timestamp}
    
        with open(output_json, "w") as fp:
            json.dump(result, fp)
    else:
        print(f"# Using previous detection result from {output_json}")
        with open(output_json, "r") as fp:
            json_file = json.load(fp)
        detected = json_file["detected"]
        timestamp = json_file["timestamp"]

    return detected, timestamp

def main(args):
    print(f"# Analyzing {args.input}")
    analyze_video(args.input, args.output, args.force)

if __name__ == "__main__":
    args = parse_args()
    main(args)