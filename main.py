from fourier.visualizer import FourierTransformScene
from manim import config
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--frequency', nargs='+', help='Frequency values', required=True)
    args = parser.parse_args()

    config.quality = "medium_quality"
    config.preview = False
    config.output_file = "fourier.mp4"  # saves to media/videos/720p30/filename.mp4

    scene = FourierTransformScene()
    scene.SIGNAL_FREQS = args.frequency
    scene.RUN_TIME = 15
    scene.render()