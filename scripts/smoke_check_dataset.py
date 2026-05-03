import sys

sys.path.insert(0, r"D:\clone\workspace\facial_tongue_health")

import train  # noqa: E402


def main():
    train_ds, train_steps = train.create_paired_dataset(train.FACE_DIR, train.TONGUE_DIR, subset="train", batch_size=8)
    val_ds, val_steps = train.create_paired_dataset(train.FACE_DIR, train.TONGUE_DIR, subset="val", batch_size=8)

    print("train_steps", train_steps, "val_steps", val_steps)
    print("train element_spec:", train_ds.element_spec)
    print("val element_spec:", val_ds.element_spec)

    (face_x, tongue_x), y = next(iter(val_ds))
    print("val batch shapes:", face_x.shape, tongue_x.shape, y.shape)


if __name__ == "__main__":
    main()

