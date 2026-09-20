import os
import numpy as np
from PIL import Image
from scipy.fft import dct

BASE = os.path.expanduser(
    "~/Downloads/iccad-official/iccad1/train"
)

folders = {
    "HS": "train_hs",
    "NHS": "train_nhs"
}


def get_dct(path):

    image = Image.open(path).convert("L")
    image = image.resize((224, 224))

    image = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    # True 2-D DCT
    coeff = dct(
        dct(
            image,
            axis=0,
            norm="ortho"
        ),
        axis=1,
        norm="ortho"
    )

    return coeff


for label, folder in folders.items():

    directory = os.path.join(
        BASE,
        folder
    )

    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(".png")
    ]

    print("\n================================")
    print(label)
    print("================================")

    for path in files[:5]:

        coeff = get_dct(path)

        magnitude = np.abs(coeff)
        log_mag = np.log1p(magnitude)

        print("\n", os.path.basename(path))

        print(
            "DC coefficient:",
            coeff[0, 0]
        )

        print(
            "DCT min:",
            coeff.min()
        )

        print(
            "DCT max:",
            coeff.max()
        )

        print(
            "DCT mean:",
            coeff.mean()
        )

        print(
            "DCT std:",
            coeff.std()
        )

        print(
            "Magnitude mean:",
            magnitude.mean()
        )

        print(
            "Magnitude std:",
            magnitude.std()
        )

        print(
            "Log magnitude mean:",
            log_mag.mean()
        )

        # Energy distribution
        total_energy = np.sum(
            coeff ** 2
        )

        low_energy = np.sum(
            coeff[:32, :32] ** 2
        )

        mid_energy = np.sum(
            coeff[32:112, 32:112] ** 2
        )

        high_energy = (
            total_energy
            - low_energy
            - mid_energy
        )

        print(
            "Total DCT energy:",
            total_energy
        )

        print(
            "Low-frequency energy ratio:",
            low_energy / total_energy
        )

        print(
            "Mid-frequency energy ratio:",
            mid_energy / total_energy
        )

        print(
            "Remaining energy ratio:",
            high_energy / total_energy
        )
