import os 
import argparse

def softlink(src_dir, dst_dir, src_prefix, dst_prefix):
    src_dir=os.path.abspath(src_dir)
    dst_dir=os.path.abspath(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)

    for name in ["expert", "antiexpert", "antiexpert_hidden", "expert_hidden"]:
        src = os.path.join(src_dir, f"{name}_{src_prefix}.bin")
        dst = os.path.join(dst_dir, f"{name}_{dst_prefix}.bin")
        os.symlink(src, dst)

    if not os.path.exists(os.path.join(dst_dir, f"emb.pt")):
        src = os.path.join(src_dir, f"emb.pt")
        dst = os.path.join(dst_dir, f"emb.pt")
        os.symlink(src, dst)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_dir",
        type=str,
        default=None
    )
    parser.add_argument(
        "--dst_dir",
        type=str,
        default=None
    )
    parser.add_argument(
        "--src_prefix",
        type=str,
        default=None
    )
    parser.add_argument(
        "--dst_prefix",
        type=str,
        default=None
    )
    args = parser.parse_args()

    softlink(args.src_dir, args.dst_dir, args.src_prefix, args.dst_prefix)