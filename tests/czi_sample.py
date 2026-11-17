"""A real Zeiss .czi file for the database tests (3 scenes, 3 channels, 5 planes, 20x objective).

The file is the public test image s_3_t_1_c_3_z_5.czi of the bioio-czi project
(https://github.com/bioio-devs/bioio-czi). It is not part of this repository: it is downloaded once
into tests/data/ (ignored by git) and checked against its SHA-256. Set ARBORIST_TEST_CZI to use a
copy that is already on disk.
"""
import hashlib
import os
import urllib.request

NAME = "s_3_t_1_c_3_z_5.czi"
SHA256 = "9f6a1a154385de9c5655f14df997d2d2460a7655714cc7248d11160331429c27"
URL = ("https://media.githubusercontent.com/media/bioio-devs/bioio-czi/main/"
       "bioio_czi/tests/resources/" + NAME)
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _ok(path):
    if not os.path.isfile(path):
        return False
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest() == SHA256


def get():
    """Path of the sample, downloading it if needed; None if it can't be had (offline)."""
    for p in (os.environ.get("ARBORIST_TEST_CZI", ""), os.path.join(DATA, NAME)):
        if p and _ok(p):
            return p
    os.makedirs(DATA, exist_ok=True)
    dst = os.path.join(DATA, NAME)
    try:
        print("downloading", URL)
        urllib.request.urlretrieve(URL, dst + ".part")
        os.replace(dst + ".part", dst)
    except Exception as ex:
        print("could not download the CZI test file:", ex)
        return None
    if not _ok(dst):
        print("downloaded CZI test file has the wrong checksum")
        os.remove(dst)
        return None
    return dst
