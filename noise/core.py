import queue
import sys
import threading

import numpy as np
import sounddevice as sd
from scipy import signal

rng = np.random.default_rng(1234)


class SoundDeviceError(Exception):
    """Custom exception for sound device errors."""


def generate_noise_block(
    block_size: int, samplerate: float, filter_coeffs: tuple, initial_state: np.ndarray  # noqa: ARG001
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a block of filtered noise."""
    b, a = filter_coeffs
    data = rng.normal(size=block_size)
    data, z = signal.lfilter(b, a, data, zi=initial_state)
    data = np.vstack([data] * 2).T
    return data, z


def audio_callback(
    outdata: np.ndarray, frames: int, time: float, status: sd.CallbackFlags, block_size: int, q: queue.Queue  # noqa: ARG001
) -> None:
    """Define callback function for audio output."""
    if frames != block_size:
        msg = f"Requested frames ({frames}) do not match block size ({block_size})"
        raise ValueError(msg)

    if status.output_underflow:
        print("Output underflow: increase blocksize?", file=sys.stderr)
        raise sd.CallbackAbort

    if status:
        msg = f"Error from sound device: {status}"
        raise SoundDeviceError(msg)

    try:
        data = q.get_nowait()
    except queue.Empty as e:
        print("Buffer is empty: increase buffer_size?", file=sys.stderr)
        raise sd.CallbackAbort from e

    if len(data) < len(outdata):
        outdata[: len(data)] = data
        outdata[len(data) :].fill(0)
        raise sd.CallbackStop

    outdata[:] = data


def initialize_noise_buffer(buffer_size: int, block_size: int, filter_coeffs: tuple) -> queue.Queue:
    """Initialize the noise buffer with pre-generated noise blocks."""
    q = queue.Queue(maxsize=buffer_size)
    b, a = filter_coeffs
    z = signal.lfilter_zi(b, a) * 0.0

    for _ in range(buffer_size):
        data, z = generate_noise_block(block_size, 44100, filter_coeffs, z)
        q.put_nowait(data)

    return q


def main():
    device_index = 3  # Adjust this to your specific device index
    block_size = 1024
    buffer_size = 200

    device = sd.query_devices(device=device_index)
    samplerate = device["default_samplerate"]
    filter_coeffs = signal.butter(2, 0.01)

    q = initialize_noise_buffer(buffer_size, block_size, filter_coeffs)
    event = threading.Event()

    try:
        stream = sd.OutputStream(
            samplerate=samplerate,
            blocksize=block_size,
            device=device["index"],
            channels=device["max_output_channels"],
            callback=lambda outdata, frames, time, status: audio_callback(
                outdata, frames, time, status, block_size, q
            ),
            finished_callback=event.set,
        )

        with stream:
            timeout = block_size * buffer_size / samplerate
            z = signal.lfilter_zi(*filter_coeffs) * 0.0

            while True:
                data, z = generate_noise_block(block_size, samplerate, filter_coeffs, z)
                q.put(data, timeout=timeout)

            event.wait()  # Wait until playback is finished

    except KeyboardInterrupt:
        print("\nInterrupted by user")


if __name__ == "__main__":
    main()
