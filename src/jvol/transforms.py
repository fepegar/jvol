import numpy as np
import numpy.typing as npt
import transforms3d


def create_ijk_to_ras_from_itk_image(image) -> npt.NDArray[np.float64]:
    ijk_to_lps = transforms3d.affines.compose(
        tuple(image.GetOrigin()),
        np.array(image.GetDirection()).reshape(3, 3),
        tuple(image.GetSpacing()),
    )
    lps_to_ras = np.diag((-1, -1, 1, 1))
    ijk_to_ras = lps_to_ras @ ijk_to_lps
    return ijk_to_ras


def get_itk_metadata_from_ijk_to_ras(ijk_to_ras: npt.NDArray[np.float64]):
    ras_to_lps = np.diag((-1, -1, 1, 1))
    ijk_to_lps = ras_to_lps @ ijk_to_ras
    origin, rotation, spacing, _ = transforms3d.affines.decompose(ijk_to_lps)
    return origin, rotation, spacing
