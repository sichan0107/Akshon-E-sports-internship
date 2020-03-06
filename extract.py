from __future__ import print_function

from multiprocessing import Process, Pipe, Event
from threading import Thread
from argparse import ArgumentParser
import collections
import os
import time
import math
import json

import cv2


def _extract(video, dest, pipe, event, target_framerate=10., target_size=(1280, 720), updates=True):
    cap = cv2.VideoCapture(video)
    video_size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_framerate = cap.get(cv2.CAP_PROP_FPS)
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    resize = None
    if video_size[0] != target_size[0] and video_size[1] != target_size[1]:
        resize = target_size
    
    interval = int(math.ceil(video_framerate / target_framerate))
    
    metadata = {
        "frame_size": target_size,
        "frame_rate": video_framerate,
        "frame_count": video_frames,
        "frame_interval": interval,
    }
    
    with open(os.path.join(dest, "metadata.json"), "w") as w:
        json.dump(metadata, w, separators=(',', ':'))
    
    pipe.send(metadata)
    pipe.send([
        os.path.join(dest, "%d.jpg" % i) for i in range(0, int(video_frames), interval)
    ])
    
    extracted_count = 0
    for i in range(video_frames):
        if event.is_set():
            break
    
        if not cap.grab():
            break
        
        if i % interval > 0:
            continue
        
        out_file = os.path.join(dest, "%d.jpg" % i)
        if os.path.exists(out_file):
            if updates:
                pipe.send((i + 1, video_frames))
            continue
        
        ok, img = cap.retrieve()
        if not ok:
            break
        
        if resize is not None:
            img = cv2.resize(img, resize)
        
        with open(out_file, 'wb') as w:
            ok, buf = cv2.imencode(out_file, img)
            if ok:
                w.write(buf)
            else:
                print("failed to write frame %d", i)
        
        extracted_count += 1
        
        if updates:
            pipe.send((i + 1, video_frames))
    
    if updates:
        pipe.send((video_frames, video_frames))
    
    pipe.close()


def _send_progress(pipe, progress):
    try:
        while True:
            progress(*pipe.recv())
    except EOFError:
        pass

def extract(video, dest=None, async=True, progress=None, verbose=False, force=False, **kwargs):
    if dest is None:
        dest = os.path.splitext(video)[0]

    if not force and os.path.exists(dest):
        raise ValueError("output directory already exists")
    
    os.makedirs(dest, exist_ok=True)
    
    parent, child = Pipe()
    
    e = Event()
    p = Process(
        target=_extract,
        args=(video, dest, child, e),
        kwargs=kwargs,
    )
    
    p.start()
    
    if progress is None:
        if verbose:
            def progress(a, b):
                print("\r%.2f%%" % (float(a) / b * 100.,), end="")
        else:
            def progress(a, b):
                pass
    
    metadata = parent.recv()
    frame_paths = parent.recv()
    
    Thread(
        target=_send_progress,
        args=(parent, progress),
    ).start()
    
    ret = [collections.namedtuple("Metadata", metadata.keys())(*metadata.values()), frame_paths]
    
    if not async:
        p.join()
        
        return tuple(ret)
    
    ret.append(e)
    
    return tuple(ret)


def _size(str):
    parts = str.split("x")
    if len(parts) != 2:
        raise ValueError("invalid size format")
    
    return (int(parts[0]), int(parts[1]))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--dest", dest="dest", default=None, required=False)
    parser.add_argument("--framerate", dest="target_framerate", default=10., type=float)
    parser.add_argument("--size", dest="target_size", type=_size, default=(1280, 720))
    args = parser.parse_args()
    
    try:
        start = time.clock()
        
        _, frame_paths = extract(args.video, dest=args.dest, target_framerate=args.target_framerate, target_size=args.target_size, verbose=True, async=False)
        
        print("\rextracted %d frames in %.2f seconds" % (len(frame_paths), time.clock() - start,))
    except ValueError as e:
        print(str(e))
    except KeyboardInterrupt:
        pass
