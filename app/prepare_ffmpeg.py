# app/prepare_ffmpeg.py
import os, shutil, imageio_ffmpeg

def main():
    ffmpeg_src = imageio_ffmpeg.get_ffmpeg_exe()
    dst_dir = "/tmp/bin"
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, "ffmpeg")
    if not os.path.exists(dst):
        shutil.copy2(ffmpeg_src, dst)
        os.chmod(dst, 0o755)
    print("ffmpeg ready at", dst)

if __name__ == "__main__":
    main()
